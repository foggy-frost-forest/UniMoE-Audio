import copy
import glob
import json
import math
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import dac
import datasets
import numpy as np
import torch
import torchaudio
import transformers
from audiotools import AudioSignal
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessor, LogitsProcessorList

try:
    import torch_npu

    IS_CUDA = False

    IS_CUDA = False
except:
    IS_CUDA = True


class Dac:
    def __init__(self):
        # Check if model exists in ./utils/dac_model, if not, download it
        base_dir = os.path.dirname(__file__)
        dac_model_dir = os.path.join(base_dir, "dac_model")
        model_path = os.path.join(dac_model_dir, "weights_16khz.pth")
        
        # Check if model exists in the expected location
        if not os.path.isfile(model_path):
            print(f"DAC model not found at {model_path}, downloading...")
            # Create directory if it doesn't exist
            os.makedirs(dac_model_dir, exist_ok=True)
            # Download the model
            downloaded_path = dac.utils.download(model_type="16khz")
            # Move the downloaded model to our target location
            shutil.move(downloaded_path, model_path)
            print(f"DAC model downloaded and saved to {model_path}")
        
        # Fallback to environment variable or other locations if needed
        env_path = os.environ.get("DAC_WEIGHTS")
        candidates = []
        if env_path:
            candidates.append(env_path)
        
        candidates.extend([
            model_path,  # Our primary location
            os.path.join(base_dir, "weights_16khz.pth"),
            os.path.join(os.getcwd(), "utils", "dac_model", "weights_16khz.pth"),
            os.path.join(os.getcwd(), "dac_model", "weights_16khz.pth"),
        ])
        
        final_model_path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if not final_model_path:
            searched = "\n - " + "\n - ".join(candidates)
            raise FileNotFoundError(
                "DAC weights not found. Please place weights_16khz.pth in one of the following locations or set DAC_WEIGHTS to an absolute path:" + searched
            )
            
        self.model = dac.DAC.load(final_model_path)
        self.resampler = dict()
        if IS_CUDA:
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("npu")

    def encode(self, audio_path):
        signal = AudioSignal(audio_path)
        if signal.audio_data.shape[1] == 2:
            signal.audio_data = 0.5 * (signal.audio_data[:, :1, :] + signal.audio_data[:, 1:, :])
        signal.to(self.model.device)

        if signal.sample_rate != 16000:
            if not str(signal.sample_rate) in self.resampler:
                self.resampler[str(signal.sample_rate)] = torchaudio.transforms.Resample(signal.sample_rate, 16000)
                if IS_CUDA:
                    self.resampler[str(signal.sample_rate)] = self.resampler[str(signal.sample_rate)].cuda()
                else:
                    self.resampler[str(signal.sample_rate)] = self.resampler[str(signal.sample_rate)].npu()

            signal.audio_data = self.resampler[str(signal.sample_rate)](signal.audio_data)
            signal.sample_rate = 16000

        x = self.model.preprocess(signal.audio_data.to(self.model.device), signal.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)

        # codes (1, 12, len)

        # codes = torch.tensor(codes[0]).transpose(0, 1)
        codes = codes[0].clone().detach().transpose(0, 1)
        assert codes.shape[1] == 12 and len(codes.shape) == 2
        codes = codes.tolist()

        return codes  # length, channel

    def decode(self, codes, save_path, min_duration=None):
        """
        codes : (1, channel, length)
        """
        assert codes.shape[0] == 1 and codes.shape[1] == 12
        z, _, _ = self.model.quantizer.from_codes(codes.to(self.model.device))
        audio_out = self.model.decode(z)[0].detach().cpu()

        sample_rate = 16000
        duration = audio_out.size(1) / sample_rate
        if min_duration is not None and duration < min_duration:
            padding_duration = min_duration - duration
            padding_samples = int(padding_duration * sample_rate)
            padding = torch.zeros((audio_out.size(0), padding_samples), dtype=audio_out.dtype, device=audio_out.device)
            audio_out = torch.cat((audio_out, padding), dim=1)

        torchaudio.save(save_path, audio_out.detach().cpu(), sample_rate=16000, encoding="PCM_S", bits_per_sample=16)


def build_delay_indices(B: int, T: int, C: int, delay_pattern: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    Negative t_idx => BOS; t_idx >= T => PAD.
    """
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)

    t_idx_BxT = torch.broadcast_to(
        torch.arange(T, dtype=torch.int32)[None, :],
        [B, T],
    )
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)

    b_idx_BxTxC = torch.broadcast_to(
        torch.arange(B, dtype=torch.int32).view(B, 1, 1),
        [B, T, C],
    )
    c_idx_BxTxC = torch.broadcast_to(
        torch.arange(C, dtype=torch.int32).view(1, 1, C),
        [B, T, C],
    )

    # We must clamp time indices to [0..T-1] so gather_nd equivalent won't fail
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        dim=1,
    ).long()  # Ensure indices are long type for indexing

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(audio_BxTxC: torch.Tensor, pad_value: int, bos_value: int, precomp: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Applies the delay pattern to batched audio tokens using precomputed indices,
    inserting BOS where t_idx < 0 and PAD where t_idx >= T.

    Args:
        audio_BxTxC: [B, T, C] int16 audio tokens (or int32/float)
        pad_value: the padding token
        bos_value: the BOS token
        precomp:  (t_idx_BxTxC, indices_BTCx3) from build_delay_indices

    Returns:
        result_BxTxC: [B, T, C] delayed audio tokens
    """
    device = audio_BxTxC.device  # Get device from input tensor
    t_idx_BxTxC, indices_BTCx3 = precomp
    t_idx_BxTxC = t_idx_BxTxC.to(device)  # Move precomputed indices to device
    indices_BTCx3 = indices_BTCx3.to(device)

    # Equivalent of tf.gather_nd using advanced indexing
    # Ensure indices are long type if not already (build_delay_indices should handle this)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)

    # Create masks on the correct device
    mask_bos = t_idx_BxTxC < 0  # => place bos_value
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]  # => place pad_value

    # Create scalar tensors on the correct device
    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)

    # If mask_bos, BOS; else if mask_pad, PAD; else original gather
    # All tensors should now be on the same device
    result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))

    return result_BxTxC


def build_revert_indices(B: int, T: int, C: int, delay_pattern: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute indices for the revert operation using PyTorch.

    Returns:
        A tuple (t_idx_BxTxC, indices_BTCx3) where:
            - t_idx_BxTxC is a tensor of shape [B, T, C] computed as time indices plus the delay.
            - indices_BTCx3 is a tensor of shape [B*T*C, 3] used for gathering, computed from:
                batch indices, clamped time indices, and channel indices.
    """
    # Use default device unless specified otherwise; assumes inputs might define device later
    device = None  # Or determine dynamically if needed, e.g., from a model parameter

    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device)

    t_idx_BT1 = torch.broadcast_to(torch.arange(T, device=device).unsqueeze(0), [B, T])
    t_idx_BT1 = t_idx_BT1.unsqueeze(-1)

    t_idx_BxTxC = torch.minimum(
        t_idx_BT1 + delay_arr.view(1, 1, C),
        torch.tensor(T - 1, device=device),
    )
    b_idx_BxTxC = torch.broadcast_to(torch.arange(B, device=device).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C, device=device).view(1, 1, C), [B, T, C])

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_idx_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        axis=1,
    ).long()  # Ensure indices are long type

    return t_idx_BxTxC, indices_BTCx3


def revert_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    precomp: Tuple[torch.Tensor, torch.Tensor],
    T: int,
) -> torch.Tensor:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices (PyTorch version).

    Args:
        audio_BxTxC: Input delayed audio tensor
        pad_value: Padding value for out-of-bounds indices
        precomp: Precomputed revert indices tuple containing:
            - t_idx_BxTxC: Time offset indices tensor
            - indices_BTCx3: Gather indices tensor for original audio
        T: Original sequence length before padding

    Returns:
        Reverted audio tensor with same shape as input
    """
    t_idx_BxTxC, indices_BTCx3 = precomp
    device = audio_BxTxC.device  # Get device from input tensor

    # Move precomputed indices to the same device as audio_BxTxC if they aren't already
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)

    # Using PyTorch advanced indexing (equivalent to tf.gather_nd or np equivalent)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.size())  # Use .size() for robust reshaping

    # Create pad_tensor on the correct device
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
    # Create T tensor on the correct device for comparison
    T_tensor = torch.tensor(T, device=device)

    result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)  # Changed np.where to torch.where

    return result_BxTxC


def _prepare_audio_prompt(model, audio_prompts: list[torch.Tensor]):
    """Prepares the audio prompt tensor for the decoder.
    Handles padding, adds the beginning-of-sequence (BOS) token, applies the
    delay pattern, and determines the number of prefill steps for each item
    in the batch.
    Args:
        audio_prompts: A list of audio prompt tensors (encoded DAC frames) or None.
                       Each tensor should have shape [T, C].
    Returns:
        A tuple containing:
            - delayed_batch (torch.Tensor): The prepared audio prompt tensor with
              delays applied, shape [B, T_max_padded, C].
            - prefill_steps (list[int]): A list containing the number of valid
              tokens (including BOS) for each prompt in the batch.
    """
    num_channels = model.config.codec_channels
    audio_bos_value = model.config.codec_bos_value
    delay_pattern = model.config.codec_delay_pattern
    max_delay_pattern = max(delay_pattern)
    batch_size = len(audio_prompts)
    max_len = max(p.shape[0] if p is not None else 0 for p in audio_prompts) + max_delay_pattern + 1
    prefill_steps = []
    prefill = torch.full(
        (batch_size, max_len, num_channels),
        fill_value=-1,
        dtype=torch.int,
        device=model.device,
    )
    prefill[:, 0, :] = audio_bos_value
    for i in range(batch_size):
        prompt = audio_prompts[i]
        if prompt is not None:
            prompt = prompt.to(device=model.device, dtype=torch.int)
            prefill[i, 1 : prompt.shape[0] + 1, :] = prompt
            prefill_steps.append(prompt.shape[0] + 1)
        else:
            prefill_steps.append(1)

    delay_precomp = build_delay_indices(
        B=batch_size,
        T=max_len,
        C=num_channels,
        delay_pattern=delay_pattern,
    )

    delayed_batch = apply_audio_delay(
        audio_BxTxC=prefill,
        pad_value=-1,
        bos_value=audio_bos_value,
        precomp=delay_precomp,
    )

    return delayed_batch, prefill_steps


class DecoderOutput:
    def __init__(self, prefill, prefill_steps, device: torch.device, labels_prefill=None):
        self.generated_tokens = prefill
        self.prefill_steps = prefill_steps
        self.labels_prefill = labels_prefill
        self.device = device

    def get_tokens_at(self, step_from: int, step_to: int = None) -> torch.Tensor:
        if step_to is None:
            step_to = step_from + 1
        return self.generated_tokens[:, step_from:step_to, :].to(self.device)

    def get_labels_at(self, step_from: int, step_to: int = None) -> torch.Tensor:
        if step_to is None:
            step_to = step_from + 1
        if self.labels_prefill is None:
            return None
        return self.labels_prefill[:, step_from:step_to, :].to(self.device)

    def update_one(self, dec_out: torch.Tensor, step: int, apply_mask: bool = False):
        dec_out = dec_out.to(self.generated_tokens.dtype).to(self.generated_tokens.device)
        if apply_mask:
            assert step < self.generated_tokens.shape[1]
            mask = self.generated_tokens[:, step, :] == -1
            self.generated_tokens[:, step, :] = torch.where(mask, dec_out, self.generated_tokens[:, step, :])
        else:
            assert step == self.generated_tokens.shape[1]
            self.generated_tokens = torch.cat((self.generated_tokens, dec_out[:, None, :]), dim=1)


def _generate_output(model, generated_codes: torch.Tensor, lengths_Bx: torch.Tensor) -> list[np.ndarray]:
    """Converts generated delayed codes into audio waveforms.
    Reverts the delay pattern applied during generation, decodes the resulting
    codebook using the DAC model (if loaded), and returns a list of audio
    waveforms as NumPy arrays. If DAC is not loaded, returns the raw codebook indices.
    Args:
        generated_codes: The tensor of generated audio codes with delays,
                         shape [B, T_gen, C].
        lengths_Bx: A tensor containing the valid length of generated codes
                    (excluding padding and BOS/EOS markers) for each item
                    in the batch, shape [B].
    Returns:
        A list of NumPy arrays, where each array represents the generated audio
        waveform for one item in the batch. If DAC is not loaded, returns the
        raw, reverted codebook indices as NumPy arrays.
    """
    num_channels = model.config.codec_channels
    batch_size = generated_codes.shape[0]
    seq_length = generated_codes.shape[1]
    delay_pattern = model.config.codec_delay_pattern
    audio_pad_value = model.config.codec_pad_value
    max_delay_pattern = max(delay_pattern)
    revert_precomp = build_revert_indices(
        B=batch_size,
        T=seq_length,
        C=num_channels,
        delay_pattern=delay_pattern,
    )
    codebook = revert_audio_delay(
        audio_BxTxC=generated_codes,
        pad_value=audio_pad_value,
        precomp=revert_precomp,
        T=seq_length,
    )[:, :-max_delay_pattern, :]

    # min_valid_index = 0
    # max_valid_index = 1023
    # invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
    # codebook[invalid_mask] = 0

    audios = []
    for i in range(batch_size):
        audios.append(codebook[i, : lengths_Bx[i], :].cpu())
    return audios


if __name__ == "__main__":
    dac = Dac()
    # data = datasets.load_from_disk("/workspace/40103/orizyliu/A_Sonic/A_workspace/Data/dac20wHrs_pretraining/all-sampleing_2M-time_1to20-VC_prompt_generated")
    # data.shuffle(233)
    # for i in range(10):
    #     instance = data[i]
    #     prompt_path = f'./{i}_prompt.wav'
    #     cur_path = f'./{i}_output.wav'
    #     print(f"[1] {instance['prompt_transcription']}")
    #     dac.decode(torch.tensor(instance['prompt_codec']).transpose(0,1).unsqueeze(0), save_path=prompt_path)
    #     print(f"[2] {prompt_path}")
    #     print(f"[3] {instance['transcription']}")
    #     dac.decode(torch.tensor(instance['codec']).transpose(0,1).unsqueeze(0), save_path=cur_path)
    #     print(f"[4] {cur_path}")
    #     print('-' * 10)
    
    codec1 = [[1, 532, 754, 532, 956, 882, 336, 638, 408, 397, 473, 89], [661, 532, 767, 184, 211, 571, 539, 404, 48, 34, 127, 525], [170, 548, 127, 196, 90, 345, 142, 437, 503, 730, 853, 462], [1, 745, 443, 1019, 668, 588, 1011, 1006, 179, 561, 335, 748], [140, 134, 177, 61, 180, 455, 169, 47, 534, 304, 40, 138], [1, 532, 266, 232, 602, 502, 714, 608, 279, 310, 505, 1019], [140, 875, 458, 161, 451, 931, 156, 605, 727, 693, 759, 672], [1, 591, 397, 110, 39, 148, 807, 277, 527, 166, 604, 758], [1, 972, 1010, 808, 515, 97, 534, 97, 406, 284, 141, 210], [140, 134, 644, 993, 561, 683, 396, 40, 169, 236, 791, 558], [170, 92, 283, 65, 1023, 449, 687, 523, 317, 908, 699, 532], [832, 629, 975, 838, 126, 371, 1010, 354, 375, 778, 344, 96], [787, 1, 583, 378, 28, 701, 155, 988, 526, 679, 916, 318], [726, 151, 532, 100, 66, 51, 888, 99, 610, 256, 455, 254], [262, 962, 913, 436, 248, 611, 929, 307, 941, 883, 559, 278], [139, 262, 746, 902, 208, 299, 112, 871, 598, 787, 152, 625], [41, 908, 179, 870, 80, 808, 40, 156, 572, 344, 597, 781], [206, 510, 633, 1000, 779, 339, 125, 872, 675, 987, 595, 580], [954, 408, 711, 984, 113, 30, 250, 1003, 765, 982, 51, 996], [769, 744, 983, 612, 437, 892, 30, 492, 396, 841, 294, 17], [566, 290, 365, 1006, 913, 231, 9, 467, 1010, 649, 927, 662], [154, 603, 923, 229, 935, 356, 164, 377, 0, 674, 428, 247], [286, 467, 758, 528, 1003, 443, 461, 226, 691, 59, 34, 611], [976, 568, 459, 968, 527, 209, 982, 855, 76, 492, 277, 179], [620, 853, 262, 606, 645, 201, 382, 1002, 284, 874, 940, 193], [550, 154, 840, 724, 570, 478, 527, 483, 31, 923, 157, 514], [123, 524, 403, 233, 362, 434, 161, 110, 858, 31, 921, 944], [781, 432, 961, 257, 316, 715, 303, 547, 557, 837, 780, 377], [90, 618, 105, 990, 421, 857, 769, 564, 458, 721, 949, 1012], [50, 573, 271, 166, 539, 589, 897, 808, 111, 512, 858, 572], [959, 413, 21, 184, 144, 558, 331, 135, 70, 57, 684, 572], [566, 551, 731, 232, 523, 271, 313, 232, 18, 257, 322, 170], [669, 766, 1015, 474, 96, 608, 155, 968, 651, 392, 668, 909], [132, 984, 342, 581, 635, 801, 372, 70, 342, 983, 204, 883], [258, 669, 398, 235, 663, 1019, 921, 74, 643, 764, 150, 592], [72, 226, 941, 958, 453, 317, 942, 463, 928, 580, 546, 192], [322, 182, 250, 506, 338, 626, 851, 338, 362, 711, 367, 317], [233, 437, 318, 950, 431, 27, 612, 224, 303, 326, 223, 832], [668, 615, 554, 464, 537, 226, 254, 707, 504, 657, 167, 922], [374, 11, 631, 746, 454, 944, 518, 465, 313, 284, 848, 27], [795, 168, 744, 468, 721, 510, 722, 778, 521, 45, 730, 822], [102, 701, 796, 992, 415, 183, 674, 52, 781, 435, 1020, 351], [442, 727, 399, 859, 895, 621, 511, 650, 871, 557, 647, 651], [295, 978, 853, 37, 597, 793, 120, 672, 834, 10, 635, 658], [738, 146, 259, 839, 10, 639, 585, 80, 135, 221, 669, 183], [690, 601, 583, 493, 389, 989, 906, 440, 508, 504, 1011, 828], [450, 9, 183, 127, 566, 67, 754, 840, 358, 191, 816, 175], [317, 875, 1015, 129, 412, 364, 165, 945, 668, 491, 351, 1004], [772, 410, 630, 729, 51, 316, 718, 464, 955, 190, 625, 282], [385, 865, 350, 483, 331, 549, 833, 30, 327, 295, 37, 1017], [693, 758, 650, 144, 874, 370, 5, 1015, 740, 68, 888, 899], [938, 485, 120, 848, 843, 76, 730, 562, 431, 451, 316, 658], [381, 256, 218, 436, 43, 75, 633, 220, 1009, 504, 996, 635], [128, 709, 620, 143, 207, 75, 574, 324, 744, 751, 1020, 522], [292, 897, 626, 622, 396, 46, 518, 253, 748, 39, 810, 954], [308, 874, 415, 817, 927, 873, 642, 778, 260, 418, 323, 299], [975, 290, 392, 798, 739, 625, 147, 935, 882, 510, 323, 991], [911, 225, 75, 889, 806, 186, 918, 82, 719, 872, 377, 555], [470, 756, 724, 524, 824, 989, 176, 412, 34, 406, 61, 386], [753, 538, 653, 848, 8, 481, 298, 737, 178, 227, 451, 895], [987, 50, 222, 204, 872, 114, 182, 719, 745, 648, 23, 750], [961, 881, 680, 766, 4, 348, 804, 177, 934, 82, 92, 288], [550, 473, 865, 588, 74, 705, 1010, 645, 156, 827, 485, 844], [322, 338, 903, 77, 404, 425, 689, 412, 868, 623, 750, 190], [636, 927, 964, 740, 49, 228, 747, 250, 411, 12, 317, 140], [711, 719, 152, 1007, 82, 266, 563, 578, 908, 147, 435, 982], [967, 269, 253, 118, 842, 483, 576, 553, 951, 90, 861, 723], [1018, 225, 363, 630, 856, 876, 978, 888, 685, 476, 714, 608], [806, 850, 527, 354, 51, 897, 511, 234, 425, 89, 262, 542], [802, 992, 576, 959, 635, 798, 372, 486, 437, 468, 1000, 804], [233, 746, 366, 834, 503, 566, 483, 766, 936, 968, 836, 613], [481, 91, 971, 746, 567, 351, 1013, 886, 1007, 159, 687, 580], [216, 723, 717, 553, 932, 834, 476, 754, 464, 827, 32, 275], [40, 782, 116, 902, 207, 667, 839, 613, 262, 343, 94, 774], [870, 514, 971, 253, 241, 363, 721, 806, 258, 745, 107, 57], [740, 311, 755, 322, 691, 707, 698, 877, 907, 450, 99, 371], [567, 986, 1009, 881, 631, 133, 361, 942, 810, 219, 79, 485], [49, 159, 817, 422, 536, 738, 158, 950, 367, 297, 671, 552], [506, 660, 507, 736, 662, 1018, 639, 482, 883, 422, 537, 430], [592, 0, 218, 720, 28, 302, 350, 1022, 515, 868, 587, 116], [469, 867, 782, 826, 878, 180, 278, 834, 496, 35, 770, 144], [560, 978, 294, 254, 348, 401, 380, 997, 558, 357, 894, 583], [134, 23, 148, 410, 883, 693, 585, 10, 904, 216, 422, 132], [853, 239, 532, 241, 360, 903, 753, 694, 419, 360, 573, 183], [764, 645, 383, 487, 874, 968, 169, 989, 484, 290, 915, 848], [765, 610, 349, 277, 444, 250, 924, 191, 538, 224, 548, 953], [577, 683, 836, 402, 549, 993, 630, 582, 831, 707, 122, 962], [308, 602, 645, 285, 525, 609, 121, 165, 824, 1008, 995, 669], [51, 263, 373, 539, 273, 31, 587, 204, 225, 246, 602, 313], [587, 116, 284, 328, 477, 390, 468, 393, 359, 278, 544, 636], [751, 854, 365, 877, 925, 932, 80, 418, 260, 953, 76, 122], [993, 192, 980, 172, 968, 880, 815, 245, 150, 764, 282, 365], [511, 679, 823, 732, 84, 460, 604, 285, 470, 11, 169, 437], [1000, 972, 1008, 146, 91, 197, 353, 74, 330, 700, 49, 270], [896, 145, 511, 830, 138, 869, 163, 630, 327, 438, 247, 267], [747, 68, 503, 277, 605, 382, 481, 657, 94, 79, 426, 814], [879, 171, 625, 234, 8, 843, 367, 1019, 721, 183, 694, 889], [428, 500, 609, 938, 922, 935, 375, 60, 809, 892, 853, 283], [649, 498, 230, 431, 946, 446, 329, 242, 805, 896, 672, 48], [67, 297, 569, 532, 981, 197, 776, 688, 800, 608, 65, 65], [132, 271, 409, 672, 331, 470, 719, 908, 209, 759, 997, 877], [789, 541, 651, 371, 955, 152, 111, 733, 393, 861, 325, 512], [655, 347, 122, 488, 840, 509, 574, 631, 107, 173, 640, 998], [304, 546, 644, 548, 503, 132, 258, 90, 919, 567, 944, 282], [344, 219, 167, 279, 701, 356, 316, 290, 676, 399, 5, 149], [430, 788, 90, 342, 184, 270, 966, 411, 500, 820, 976, 164], [367, 789, 53, 738, 193, 812, 96, 115, 56, 934, 61, 120], [954, 263, 328, 989, 237, 306, 804, 691, 125, 863, 491, 704], [221, 583, 942, 525, 151, 344, 895, 457, 288, 883, 458, 670], [364, 47, 291, 709, 886, 307, 159, 650, 825, 336, 442, 183], [716, 711, 838, 632, 587, 7, 44, 560, 461, 919, 721, 73], [636, 376, 1003, 582, 926, 401, 974, 684, 528, 524, 348, 12], [673, 359, 592, 496, 1010, 751, 701, 188, 432, 990, 404, 319], [548, 129, 108, 981, 327, 756, 635, 699, 559, 823, 949, 235], [380, 611, 100, 891, 516, 1012, 155, 157, 493, 158, 696, 96], [1007, 972, 160, 745, 766, 654, 242, 211, 473, 184, 994, 439], [829, 190, 184, 458, 235, 415, 16, 829, 246, 626, 983, 840], [438, 885, 397, 83, 306, 987, 886, 429, 686, 1001, 243, 151], [511, 844, 428, 124, 271, 558, 202, 595, 266, 229, 719, 400], [972, 821, 539, 205, 71, 893, 262, 923, 975, 162, 47, 290], [548, 308, 714, 438, 453, 776, 99, 964, 71, 9, 311, 52], [717, 938, 923, 49, 459, 521, 396, 723, 195, 121, 79, 202], [677, 408, 110, 720, 549, 967, 922, 751, 543, 57, 732, 365], [779, 605, 431, 701, 123, 481, 262, 61, 391, 59, 616, 8], [11, 578, 295, 76, 575, 904, 336, 149, 76, 936, 78, 334], [771, 895, 133, 836, 83, 797, 17, 959, 908, 302, 162, 273], [69, 478, 354, 602, 730, 763, 179, 538, 689, 214, 323, 183], [712, 262, 802, 785, 563, 745, 12, 464, 356, 728, 820, 814], [1007, 437, 494, 902, 11, 623, 78, 179, 513, 959, 667, 292], [418, 989, 826, 485, 142, 115, 601, 191, 640, 162, 526, 24], [573, 223, 578, 199, 994, 49, 930, 205, 842, 755, 916, 405], [800, 842, 724, 770, 947, 256, 532, 805, 183, 198, 886, 125], [273, 166, 131, 808, 575, 986, 321, 692, 200, 343, 589, 347], [668, 380, 119, 74, 140, 710, 475, 550, 868, 827, 41, 962], [335, 860, 277, 357, 958, 972, 949, 741, 925, 731, 549, 107], [118, 249, 750, 720, 943, 940, 636, 351, 319, 903, 158, 221], [706, 312, 783, 170, 255, 159, 217, 1022, 490, 353, 280, 503], [330, 11, 241, 869, 786, 82, 805, 900, 886, 229, 97, 934], [958, 731, 943, 836, 139, 403, 988, 8, 81, 744, 433, 118], [878, 674, 59, 96, 87, 998, 962, 715, 202, 816, 394, 314], [752, 903, 228, 227, 474, 382, 836, 728, 654, 325, 689, 123], [124, 319, 206, 895, 970, 348, 698, 529, 508, 697, 141, 181], [130, 739, 496, 913, 628, 612, 670, 650, 62, 902, 148, 118], [859, 1, 684, 374, 806, 767, 402, 286, 142, 172, 771, 135], [703, 490, 651, 936, 32, 307, 62, 212, 290, 702, 966, 539], [409, 667, 160, 823, 679, 143, 430, 63, 422, 996, 273, 657], [292, 957, 241, 201, 943, 199, 785, 705, 837, 89, 522, 516], [472, 660, 975, 670, 991, 248, 323, 569, 439, 859, 1010, 615], [901, 285, 741, 827, 94, 1009, 943, 51, 898, 422, 280, 141], [641, 513, 400, 1016, 921, 267, 967, 367, 511, 137, 290, 905], [931, 319, 1014, 664, 881, 282, 522, 965, 745, 446, 735, 448], [810, 176, 10, 923, 528, 886, 890, 855, 126, 726, 862, 417], [200, 248, 116, 1, 864, 110, 16, 73, 566, 469, 910, 945], [95, 107, 849, 22, 905, 493, 793, 983, 733, 556, 253, 292], [267, 661, 209, 553, 100, 962, 636, 245, 298, 168, 295, 593], [107, 463, 233, 753, 19, 285, 353, 242, 886, 846, 43, 689], [100, 914, 94, 580, 789, 424, 646, 610, 213, 422, 707, 165], [192, 698, 268, 437, 53, 401, 542, 397, 37, 79, 314, 51], [665, 203, 187, 412, 577, 278, 671, 658, 781, 112, 861, 879], [66, 185, 195, 652, 858, 960, 735, 244, 382, 259, 80, 656], [887, 713, 511, 922, 43, 982, 917, 301, 385, 641, 114, 631], [602, 807, 859, 242, 577, 653, 990, 753, 618, 582, 918, 692], [41, 511, 882, 881, 1001, 604, 349, 374, 309, 253, 425, 732], [225, 778, 853, 962, 152, 31, 199, 736, 1018, 900, 180, 233], [0, 677, 355, 831, 641, 790, 478, 482, 39, 179, 435, 753], [634, 738, 678, 116, 968, 777, 909, 400, 598, 515, 557, 308], [892, 552, 1013, 532, 124, 706, 199, 288, 890, 856, 895, 112], [919, 802, 48, 966, 186, 308, 97, 321, 457, 931, 870, 496], [580, 733, 803, 518, 140, 583, 660, 684, 972, 700, 546, 381], [571, 337, 796, 1021, 192, 201, 478, 585, 644, 426, 916, 489], [1012, 220, 533, 559, 172, 972, 872, 83, 870, 40, 317, 600], [752, 700, 220, 235, 382, 282, 63, 543, 157, 385, 265, 995], [562, 223, 377, 507, 961, 249, 852, 120, 836, 900, 654, 210], [768, 205, 774, 802, 98, 663, 292, 580, 48, 315, 389, 696], [803, 392, 672, 694, 256, 148, 837, 542, 361, 831, 998, 112], [787, 332, 245, 392, 423, 161, 651, 433, 758, 998, 384, 922], [488, 275, 167, 840, 317, 243, 454, 350, 404, 138, 301, 815], [221, 944, 802, 308, 425, 561, 688, 721, 509, 81, 691, 553], [483, 298, 427, 596, 106, 96, 17, 702, 263, 82, 301, 763], [293, 350, 807, 431, 998, 266, 379, 175, 195, 749, 711, 798], [568, 713, 1013, 161, 652, 617, 403, 427, 59, 537, 870, 12], [498, 7, 611, 332, 259, 443, 358, 156, 264, 795, 792, 465], [331, 166, 582, 604, 23, 626, 839, 399, 629, 183, 971, 728], [488, 478, 882, 437, 104, 641, 770, 923, 420, 404, 937, 843], [718, 528, 895, 786, 560, 616, 657, 647, 944, 681, 60, 410], [12, 698, 266, 696, 244, 965, 10, 240, 128, 640, 385, 557], [99, 297, 811, 713, 651, 753, 701, 399, 496, 940, 369, 710], [641, 940, 15, 320, 144, 316, 192, 399, 641, 526, 539, 213], [799, 285, 448, 176, 478, 492, 213, 368, 945, 880, 846, 427], [636, 306, 456, 15, 163, 25, 182, 554, 326, 707, 211, 519], [144, 76, 899, 86, 719, 185, 129, 861, 884, 518, 430, 613], [133, 66, 288, 86, 1007, 859, 590, 60, 907, 721, 918, 503], [859, 859, 269, 256, 388, 323, 484, 968, 672, 662, 95, 902], [421, 592, 772, 72, 908, 349, 915, 140, 788, 221, 548, 30], [450, 653, 958, 348, 192, 913, 790, 806, 918, 829, 591, 858], [694, 178, 644, 78, 808, 608, 202, 754, 479, 978, 461, 683], [172, 897, 383, 261, 758, 42, 543, 165, 416, 463, 210, 315], [339, 137, 879, 845, 237, 614, 840, 260, 100, 823, 146, 250], [679, 101, 352, 810, 156, 565, 105, 703, 797, 149, 561, 603], [574, 156, 866, 480, 25, 75, 175, 135, 828, 112, 759, 135], [822, 758, 982, 13, 323, 73, 285, 146, 989, 74, 23, 82], [562, 984, 63, 339, 646, 963, 87, 874, 610, 228, 780, 669], [913, 968, 24, 26, 620, 603, 306, 713, 549, 126, 240, 744], [142, 13, 683, 810, 393, 156, 89, 674, 137, 206, 661, 740], [933, 898, 976, 554, 69, 729, 502, 91, 239, 373, 581, 965], [859, 978, 337, 56, 133, 19, 593, 686, 899, 745, 252, 363], [479, 120, 19, 456, 686, 456, 835, 934, 379, 372, 448, 677], [859, 383, 1018, 262, 681, 612, 246, 719, 428, 585, 591, 118]]
    codec2 = [[222, 663, 860, 382, 932, 840, 889, 358, 208, 704, 566, 695], [666, 183, 803, 537, 44, 471, 415, 22, 889, 31, 610, 422], [658, 281, 767, 312, 541, 210, 172, 270, 382, 6, 97, 524], [371, 308, 132, 911, 598, 460, 316, 728, 664, 469, 224, 725], [809, 949, 197, 1020, 246, 339, 392, 465, 564, 451, 117, 281], [51, 354, 585, 158, 974, 364, 9, 969, 412, 336, 610, 244], [89, 827, 97, 338, 842, 10, 60, 83, 830, 241, 103, 724], [496, 226, 619, 855, 632, 28, 636, 670, 615, 562, 135, 733], [620, 44, 674, 992, 110, 60, 594, 247, 30, 583, 148, 476], [316, 1015, 858, 869, 177, 49, 184, 494, 112, 932, 754, 538], [913, 404, 223, 1008, 226, 249, 407, 186, 991, 972, 586, 452], [74, 422, 824, 1011, 59, 281, 258, 246, 461, 233, 594, 14], [254, 713, 530, 9, 116, 797, 647, 534, 964, 696, 650, 881], [742, 334, 920, 135, 632, 342, 1021, 465, 958, 148, 374, 910], [247, 354, 468, 172, 380, 736, 478, 579, 972, 969, 642, 387], [375, 982, 232, 311, 543, 797, 100, 248, 950, 889, 268, 1004], [426, 783, 312, 900, 617, 1003, 533, 317, 125, 440, 219, 747], [57, 812, 886, 403, 1009, 995, 636, 457, 75, 996, 440, 1020], [23, 850, 808, 324, 724, 518, 714, 515, 125, 3, 937, 590], [647, 1010, 654, 279, 477, 321, 578, 754, 978, 145, 510, 171], [145, 69, 238, 802, 8, 536, 869, 581, 401, 285, 123, 800], [619, 564, 98, 27, 894, 661, 734, 547, 702, 901, 270, 255], [418, 698, 504, 967, 536, 283, 993, 665, 985, 820, 312, 653], [12, 7, 976, 757, 717, 94, 683, 549, 165, 794, 651, 665], [42, 10, 91, 116, 259, 918, 763, 618, 378, 852, 397, 195], [954, 487, 993, 552, 346, 884, 387, 879, 100, 117, 366, 456], [960, 752, 347, 627, 921, 246, 616, 241, 26, 379, 516, 762], [423, 258, 921, 482, 83, 572, 413, 576, 20, 265, 70, 857], [506, 89, 120, 959, 52, 608, 532, 137, 307, 167, 632, 377], [747, 33, 123, 476, 399, 556, 199, 676, 720, 517, 611, 647], [818, 585, 36, 434, 50, 835, 524, 260, 318, 838, 104, 297], [817, 697, 515, 493, 505, 1018, 993, 999, 234, 43, 764, 12], [226, 445, 983, 526, 942, 236, 566, 953, 723, 981, 692, 524], [145, 974, 430, 750, 40, 952, 837, 527, 437, 427, 37, 474], [72, 213, 374, 1005, 429, 361, 435, 900, 865, 20, 361, 79], [217, 805, 194, 748, 975, 30, 988, 170, 1020, 937, 195, 171], [932, 687, 886, 26, 590, 232, 339, 282, 872, 895, 396, 9], [827, 258, 725, 362, 916, 205, 665, 684, 331, 979, 712, 22], [103, 199, 956, 127, 681, 370, 195, 674, 589, 386, 536, 116], [308, 221, 262, 766, 689, 659, 444, 337, 151, 377, 922, 435], [247, 180, 382, 570, 360, 605, 205, 561, 551, 495, 613, 199], [253, 996, 292, 989, 1006, 188, 497, 242, 930, 452, 57, 588], [809, 273, 802, 919, 727, 700, 565, 406, 12, 292, 997, 862], [680, 613, 59, 914, 909, 375, 541, 27, 181, 33, 615, 630], [791, 167, 930, 397, 233, 89, 267, 191, 347, 591, 429, 838], [768, 658, 672, 169, 744, 845, 412, 191, 180, 694, 719, 940], [455, 947, 932, 963, 35, 548, 105, 736, 41, 229, 723, 472], [322, 256, 485, 893, 921, 405, 728, 448, 212, 868, 727, 661], [914, 320, 639, 799, 767, 64, 529, 267, 504, 17, 711, 151], [57, 947, 492, 431, 141, 337, 716, 482, 234, 517, 746, 127], [868, 942, 711, 127, 550, 884, 454, 901, 238, 133, 518, 167], [154, 364, 413, 78, 645, 1002, 975, 508, 692, 979, 341, 31], [292, 356, 778, 708, 215, 786, 299, 763, 63, 968, 19, 818], [672, 200, 566, 223, 146, 652, 114, 179, 908, 792, 375, 103], [267, 200, 644, 560, 561, 649, 1023, 844, 447, 42, 302, 552], [31, 711, 60, 187, 55, 817, 520, 293, 263, 800, 285, 306], [999, 473, 349, 36, 788, 182, 655, 1015, 771, 743, 95, 385], [251, 697, 343, 187, 909, 866, 544, 848, 149, 843, 518, 467], [77, 39, 459, 689, 759, 651, 91, 143, 188, 993, 693, 533], [245, 113, 481, 234, 849, 726, 99, 323, 409, 758, 838, 912], [898, 435, 480, 471, 771, 123, 822, 575, 908, 881, 950, 721], [34, 1005, 805, 283, 685, 794, 966, 285, 964, 693, 123, 726], [326, 122, 594, 837, 135, 698, 329, 760, 159, 405, 122, 958], [529, 894, 940, 427, 840, 33, 691, 217, 182, 660, 755, 800], [641, 461, 903, 446, 199, 361, 973, 277, 1015, 791, 713, 801], [904, 843, 653, 257, 1013, 368, 173, 790, 615, 801, 972, 894], [796, 478, 323, 265, 701, 436, 701, 403, 60, 408, 80, 793], [919, 949, 62, 1007, 972, 719, 341, 761, 367, 562, 989, 301], [800, 452, 784, 642, 545, 106, 528, 288, 47, 852, 885, 800], [273, 437, 110, 499, 84, 853, 422, 209, 415, 576, 165, 806], [768, 798, 144, 58, 286, 817, 204, 548, 812, 419, 652, 793], [804, 251, 229, 846, 202, 299, 931, 37, 877, 3, 602, 545], [691, 28, 711, 930, 89, 723, 229, 367, 370, 363, 808, 530], [77, 670, 478, 152, 223, 611, 320, 906, 479, 419, 409, 925], [278, 452, 1017, 654, 356, 822, 742, 522, 837, 484, 664, 934], [791, 294, 617, 323, 46, 88, 946, 319, 448, 132, 525, 263], [119, 843, 482, 236, 335, 631, 898, 193, 244, 195, 740, 590], [697, 778, 694, 918, 182, 557, 239, 679, 605, 257, 899, 1017], [31, 3, 478, 810, 173, 6, 29, 52, 652, 767, 157, 192], [11, 551, 557, 20, 594, 284, 136, 437, 837, 429, 951, 789], [142, 818, 369, 77, 165, 531, 520, 745, 510, 617, 482, 569], [816, 279, 229, 74, 84, 350, 775, 572, 243, 499, 5, 604], [470, 365, 388, 202, 437, 549, 535, 448, 346, 741, 569, 272], [924, 191, 544, 42, 140, 783, 386, 604, 657, 916, 92, 484], [578, 32, 505, 958, 109, 679, 985, 649, 931, 550, 464, 334], [892, 1019, 931, 234, 684, 924, 29, 209, 702, 708, 802, 953], [991, 431, 559, 193, 673, 252, 403, 76, 750, 205, 632, 541], [826, 454, 193, 52, 275, 658, 810, 634, 150, 368, 1008, 443], [576, 21, 39, 800, 175, 640, 119, 587, 122, 982, 1022, 244], [67, 136, 168, 277, 611, 838, 605, 659, 267, 811, 0, 712], [11, 765, 432, 77, 154, 463, 439, 867, 127, 436, 480, 937], [768, 896, 705, 381, 48, 94, 943, 65, 561, 685, 654, 89], [232, 788, 48, 644, 908, 524, 154, 703, 165, 332, 454, 657], [136, 30, 799, 967, 481, 696, 296, 758, 722, 397, 500, 865], [453, 37, 714, 318, 355, 773, 648, 507, 571, 608, 858, 103], [393, 563, 133, 198, 55, 1001, 988, 937, 394, 699, 496, 21], [716, 94, 852, 968, 627, 89, 827, 469, 203, 766, 354, 284], [550, 106, 560, 572, 94, 281, 428, 340, 790, 332, 214, 568], [208, 899, 44, 561, 485, 264, 888, 632, 117, 495, 133, 171], [340, 360, 318, 496, 602, 642, 29, 521, 10, 954, 705, 338], [377, 881, 924, 308, 35, 926, 974, 141, 1000, 531, 1000, 615], [217, 794, 222, 33, 385, 319, 617, 129, 995, 249, 979, 886], [365, 347, 594, 163, 824, 134, 237, 825, 197, 402, 238, 441], [918, 196, 148, 295, 769, 143, 961, 788, 933, 934, 1001, 365], [254, 320, 669, 332, 455, 136, 246, 414, 929, 355, 895, 412], [79, 583, 255, 47, 393, 809, 207, 945, 962, 910, 339, 352], [583, 454, 552, 352, 563, 913, 687, 522, 673, 849, 635, 954], [768, 698, 748, 350, 162, 80, 140, 825, 858, 980, 626, 73], [892, 390, 398, 145, 402, 111, 602, 437, 1020, 599, 530, 669], [789, 311, 533, 290, 746, 701, 732, 734, 803, 628, 54, 525], [517, 935, 613, 228, 693, 146, 22, 977, 692, 143, 146, 139], [52, 297, 522, 668, 162, 990, 646, 954, 563, 1015, 881, 842], [330, 689, 941, 688, 80, 54, 620, 607, 698, 952, 109, 688], [124, 988, 1007, 731, 957, 852, 751, 1004, 993, 1004, 106, 422], [655, 563, 488, 472, 708, 454, 742, 662, 214, 159, 454, 0], [192, 560, 75, 557, 595, 983, 153, 495, 375, 859, 1021, 951], [660, 717, 83, 485, 902, 129, 62, 383, 163, 149, 751, 45], [405, 715, 381, 747, 858, 519, 199, 487, 896, 830, 23, 262], [397, 463, 640, 423, 859, 845, 32, 345, 262, 813, 556, 665], [200, 109, 654, 40, 227, 459, 211, 22, 169, 482, 748, 856], [477, 345, 553, 127, 53, 396, 81, 347, 774, 910, 695, 659], [799, 348, 635, 155, 744, 648, 712, 13, 292, 417, 754, 338], [534, 12, 323, 366, 75, 264, 294, 473, 515, 6, 387, 64], [237, 70, 618, 847, 437, 304, 341, 251, 677, 965, 124, 854], [431, 368, 712, 966, 196, 766, 31, 67, 408, 665, 746, 450], [907, 212, 109, 72, 1023, 201, 434, 749, 420, 64, 575, 957], [959, 269, 104, 150, 292, 284, 552, 807, 775, 200, 675, 519], [988, 851, 637, 53, 292, 613, 306, 731, 288, 940, 184, 316], [102, 199, 884, 744, 968, 56, 385, 970, 471, 235, 311, 378], [17, 299, 313, 321, 712, 149, 609, 360, 270, 137, 212, 633], [375, 975, 653, 635, 355, 430, 801, 337, 407, 551, 997, 908], [690, 817, 32, 364, 34, 506, 983, 196, 532, 980, 503, 746], [253, 275, 606, 89, 704, 746, 239, 817, 583, 220, 266, 648], [946, 914, 108, 248, 646, 941, 358, 778, 816, 342, 514, 385], [579, 469, 569, 419, 793, 142, 659, 729, 177, 534, 715, 437], [524, 168, 819, 355, 286, 220, 927, 660, 248, 1010, 77, 858], [859, 827, 228, 556, 928, 550, 278, 806, 203, 167, 202, 102], [42, 166, 936, 3, 648, 539, 316, 412, 822, 189, 455, 870], [181, 567, 46, 148, 488, 411, 844, 595, 40, 1004, 653, 335], [865, 303, 678, 512, 849, 303, 1013, 624, 971, 871, 595, 378], [896, 467, 470, 650, 622, 727, 737, 679, 720, 106, 803, 430], [182, 434, 836, 362, 941, 30, 298, 15, 276, 552, 585, 35], [182, 622, 424, 592, 234, 285, 491, 89, 180, 155, 797, 373], [612, 848, 262, 982, 983, 96, 771, 216, 700, 1002, 129, 782], [808, 311, 158, 894, 186, 736, 519, 566, 180, 265, 8, 506], [338, 125, 711, 625, 447, 812, 201, 325, 480, 564, 516, 905], [782, 897, 62, 744, 217, 627, 140, 528, 109, 721, 374, 980], [273, 537, 586, 79, 655, 930, 10, 812, 775, 199, 886, 705], [991, 797, 931, 839, 248, 990, 128, 947, 713, 545, 854, 179], [514, 996, 802, 646, 609, 354, 315, 497, 80, 518, 503, 41], [118, 701, 603, 420, 496, 125, 598, 782, 887, 973, 621, 529], [315, 380, 712, 309, 654, 99, 0, 169, 643, 260, 582, 856], [556, 630, 886, 417, 718, 323, 736, 743, 693, 356, 416, 19], [333, 1004, 472, 870, 129, 737, 805, 287, 511, 746, 762, 737], [151, 728, 711, 399, 884, 202, 243, 63, 13, 168, 359, 891], [175, 264, 742, 630, 326, 52, 245, 82, 114, 453, 241, 1012], [73, 598, 600, 284, 396, 156, 81, 559, 422, 182, 712, 53], [804, 875, 367, 501, 110, 334, 808, 178, 79, 825, 177, 482], [653, 346, 898, 787, 384, 736, 513, 284, 740, 692, 131, 776], [245, 972, 188, 136, 471, 311, 118, 424, 466, 340, 634, 76], [716, 442, 169, 466, 779, 582, 542, 475, 69, 871, 916, 389], [385, 325, 132, 592, 929, 257, 99, 868, 814, 997, 694, 993], [528, 821, 989, 970, 234, 782, 227, 239, 704, 36, 882, 604], [896, 827, 1022, 702, 680, 274, 1022, 724, 960, 501, 354, 838], [319, 373, 5, 256, 111, 185, 53, 854, 871, 84, 41, 472], [321, 16, 669, 417, 255, 346, 978, 1004, 501, 975, 183, 417], [532, 328, 957, 451, 436, 239, 181, 718, 769, 372, 561, 203], [942, 576, 786, 93, 134, 349, 609, 262, 883, 755, 993, 184], [718, 454, 314, 998, 363, 247, 920, 742, 57, 772, 844, 14], [666, 1006, 410, 20, 402, 543, 505, 252, 12, 981, 331, 979], [229, 591, 4, 488, 45, 790, 267, 552, 892, 293, 893, 108], [442, 106, 700, 714, 299, 227, 307, 463, 310, 501, 992, 144], [244, 783, 889, 244, 148, 309, 943, 766, 346, 998, 360, 836], [829, 590, 533, 539, 733, 258, 39, 560, 989, 308, 838, 833], [350, 829, 222, 369, 600, 492, 983, 103, 681, 103, 873, 158], [795, 871, 837, 745, 150, 281, 532, 995, 873, 138, 369, 120], [428, 476, 519, 950, 171, 325, 922, 995, 161, 1000, 664, 493], [504, 756, 941, 598, 472, 138, 455, 826, 367, 316, 538, 780], [326, 635, 41, 244, 959, 417, 690, 522, 400, 228, 404, 388], [901, 975, 569, 511, 107, 773, 263, 705, 100, 605, 722, 580], [853, 943, 859, 878, 692, 56, 186, 460, 503, 118, 2, 901], [563, 248, 979, 783, 249, 17, 119, 389, 49, 312, 187, 473], [166, 601, 678, 682, 350, 192, 548, 775, 509, 639, 426, 807], [182, 921, 197, 598, 420, 1022, 772, 909, 69, 483, 750, 518], [589, 337, 959, 295, 673, 416, 836, 754, 795, 251, 472, 658], [989, 599, 921, 606, 249, 217, 935, 296, 206, 376, 230, 200], [171, 492, 783, 192, 131, 838, 629, 520, 172, 55, 324, 898], [50, 55, 876, 235, 539, 358, 860, 805, 292, 764, 814, 62], [841, 603, 483, 596, 411, 777, 387, 337, 237, 926, 827, 148], [864, 227, 720, 291, 105, 120, 114, 372, 185, 943, 96, 545], [292, 805, 456, 406, 9, 286, 487, 734, 912, 278, 993, 51], [944, 668, 1000, 869, 428, 1016, 614, 900, 886, 474, 6, 657], [731, 432, 122, 297, 840, 57, 370, 775, 568, 2, 28, 56], [308, 961, 1005, 825, 868, 638, 287, 623, 656, 693, 703, 762], [367, 512, 460, 461, 884, 238, 752, 414, 206, 619, 344, 949], [167, 63, 621, 743, 63, 969, 269, 33, 663, 684, 293, 516], [98, 574, 965, 976, 734, 1007, 158, 347, 48, 296, 607, 831], [130, 311, 157, 172, 648, 339, 456, 395, 924, 128, 1023, 473], [768, 535, 572, 924, 827, 620, 878, 314, 97, 404, 325, 849], [664, 871, 583, 210, 365, 808, 109, 52, 337, 677, 965, 298], [1, 275, 607, 471, 816, 783, 98, 14, 350, 631, 894, 329], [330, 848, 644, 585, 1003, 124, 952, 345, 303, 587, 948, 429], [256, 784, 853, 995, 679, 897, 255, 54, 557, 87, 0, 982], [1, 42, 990, 132, 519, 135, 627, 158, 504, 549, 914, 309], [1, 449, 667, 913, 501, 418, 869, 471, 886, 131, 77, 705], [1, 409, 862, 995, 846, 371, 497, 74, 228, 369, 254, 697], [958, 449, 681, 240, 956, 824, 33, 429, 839, 195, 742, 1012], [473, 771, 884, 736, 475, 91, 769, 603, 929, 764, 589, 195], [1, 273, 685, 38, 482, 848, 254, 165, 602, 539, 422, 108], [256, 813, 837, 305, 760, 667, 9, 393, 828, 881, 375, 797], [140, 178, 39, 60, 383, 552, 945, 548, 251, 466, 671, 452], [163, 134, 812, 461, 528, 812, 248, 717, 676, 953, 994, 971], [1, 409, 437, 196, 392, 805, 94, 388, 894, 383, 693, 209], [256, 273, 481, 172, 333, 656, 826, 535, 214, 8, 1003, 446], [151, 548, 230, 644, 994, 885, 799, 971, 158, 96, 1015, 267], [1, 972, 765, 315, 1004, 857, 664, 490, 395, 174, 736, 40], [1, 493, 701, 117, 231, 364, 505, 544, 108, 843, 528, 791], [473, 273, 927, 502, 435, 938, 769, 188, 1022, 582, 778, 936], [256, 134, 170, 481, 999, 239, 561, 355, 161, 105, 1019, 809], [256, 356, 993, 325, 769, 222, 324, 332, 824, 374, 106, 541]]
    dac.decode(torch.tensor(codec1).transpose(0,1).unsqueeze(0), save_path='./test1.wav')
    dac.decode(torch.tensor(codec2).transpose(0,1).unsqueeze(0), save_path='./test2.wav')
