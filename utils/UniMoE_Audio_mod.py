# -*- coding: utf-8 -*-
"""
UniMoE Audio Module
"""

import sys
import os
import torch
import torchaudio
from transformers import AutoTokenizer
import numpy as np
import tempfile
import time
import shutil
from pathlib import Path
import json
from functools import lru_cache
import threading
import itertools

# Import from our merged modules
from .UniMoE_Audio_utils import (
    Dac,
    _prepare_audio_prompt,
    DecoderOutput,
    _generate_output,
)
from .UniMoE_Audio_model import (
    UniAudioRVQQwen2_5VLMoEForConditionalGeneration,
    UniAudioRVQQwen2_5VLMoEConfig,
)


class UniMoEAudio:
    """UniMoE Audio generation class for text-to-music and text-to-speech."""
    
    def __init__(self, model_path, device_id=0):
        """Initialize the UniMoE Audio model.
        
        Args:
            model_path (str): Path to the model directory
            device_id (int): CUDA device ID to use
        """
        # Configuration parameters
        self.TORCH_DTYPE = torch.bfloat16
        self.MAX_TOKENS = 1500
        
        # Templates and constants
        self.SYSTEM_MESSAGE = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"""
        self.INPUT_FORMAT = """<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""
        self.AUDIO_START = "<|AUDIO_START|>"
        
        # Create temp directory
        self.TEMP_DIR = tempfile.mkdtemp()
        
        # Initialize model components
        self._initialize_model(model_path, device_id)
    
    def _initialize_model(self, model_path, device_id):
        """Initialize model, DAC, and tokenizer."""
        if not torch.cuda.is_available():
            raise RuntimeError("This application requires an NVIDIA GPU and CUDA environment.")
        
        torch.cuda.set_device(device_id)
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print("Loading UniMoE Audio model...")
        self.model = UniAudioRVQQwen2_5VLMoEForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=self.TORCH_DTYPE, 
            attn_implementation=None
        ).to(self.device)
        self.model.eval()

        print("Loading DAC...")
        self.dac = Dac()
        self._move_dac_to_device(self.dac, self.device)

        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=False)
        special_tokens = [
            "<|AUDIO_PLACEHOLDER|>", "<|AUDIO_START|>", "<|AUDIO_END|>",
            "<|SPEECH_START|>", "<|SPEECH_END|>", "<|VOICE_PROMPT_START|>",
            "<|VOICE_PROMPT_END|>", "<|SPEECH_PROMPT_START|>", "<|SPEECH_PROMPT_END|>",
            "<|MUSIC_START|>", "<|MUSIC_END|>"
        ]
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        print("Model initialization complete!")

    def _move_dac_to_device(self, dac_instance, target_device):
        """Move DAC components to target device."""
        if hasattr(dac_instance, 'model') and dac_instance.model is not None:
            dac_instance.model.to(target_device)
        if hasattr(dac_instance, 'resampler'):
            for key in dac_instance.resampler:
                dac_instance.resampler[key].to(target_device)

    def __del__(self):
        """Cleanup temporary directory."""
        try:
            if hasattr(self, 'TEMP_DIR') and self.TEMP_DIR is not None and os.path.exists(self.TEMP_DIR):
                shutil.rmtree(self.TEMP_DIR, ignore_errors=True)
        except (AttributeError, TypeError):
            # Ignore errors during cleanup in case attributes are already None
            pass

    def _preprocess_codec(self, codec, codec_delay_pattern, codec_channels, codec_bos_value, codec_eos_value, codec_pad_value):
        """Preprocess codec tokens with delay patterns."""
        codec_token = codec.clone().detach().to(torch.long)
        codec_token_len = codec_token.shape[0]
        max_delay_pattern = max(codec_delay_pattern)
        codec_input_ids = torch.zeros((codec_token_len + max_delay_pattern + 1, codec_channels), dtype=torch.long)
        for c in range(codec_channels):
            start = codec_delay_pattern[c] + 1
            codec_input_ids[:start, c] = codec_bos_value
            codec_input_ids[start : start + codec_token_len, c] = codec_token[:, c]
            codec_input_ids[start + codec_token_len :, c] = codec_pad_value
            if start + codec_token_len < codec_input_ids.shape[0]:
                codec_input_ids[start + codec_token_len, c] = codec_eos_value
        return codec_input_ids

    def _safe_encode_audio(self, audio_path):
        """Safely encode audio file to codec tokens."""
        print(f"Encoding audio: {audio_path}")
        with torch.cuda.device(self.device):
            try:
                codes = self.dac.encode(audio_path)
                return torch.tensor(codes, dtype=torch.long) if isinstance(codes, list) else codes
            except Exception as e1:
                try:
                    wav, sr = torchaudio.load(audio_path)
                    if wav.shape[0] > 1:
                        wav = torch.mean(wav, dim=0, keepdim=True)
                    wav = wav.to(self.device)
                    temp_wav_path = os.path.join(self.TEMP_DIR, f"temp_audio_{int(time.time())}.wav")
                    torchaudio.save(temp_wav_path, wav.cpu(), sr)
                    try:
                        result = self.dac.encode(temp_wav_path)
                        return torch.tensor(result, dtype=torch.long) if isinstance(result, list) else result
                    finally:
                        if os.path.exists(temp_wav_path):
                            os.unlink(temp_wav_path)
                except Exception as e2:
                    raise Exception(f"Audio encoding failed: {e1}, {e2}")

    def _generate_audio_core(self, source_input, codec_input_ids, cfg_scale=1.0, temperature=1.0,
                            max_audio_seconds=10, min_audio_seconds=1, top_p=1.0,
                            cfg_filter_top_k=45, eos_prob_mul_factor=1.0, do_sample=True):
        """Core audio generation function."""
        batch_size = len(source_input['input_ids']) // 2
        input_ids = source_input.input_ids.to(self.device)
        attention_mask = source_input.attention_mask.to(self.device)
        if codec_input_ids is not None:
            codec_input_ids = codec_input_ids.to(self.device)

        calculated_max_tokens = min(int(max_audio_seconds * 50), self.MAX_TOKENS)

        prefill, prefill_steps = _prepare_audio_prompt(self.model, audio_prompts=[None] * batch_size)
        dec_output = DecoderOutput(prefill, prefill_steps, self.model.device, labels_prefill=None)

        with torch.no_grad():
            generated_codes, lengths = self.model.generate(
                input_ids=input_ids,
                codec_input_ids=codec_input_ids,
                attention_mask=attention_mask,
                dec_output=dec_output,
                max_tokens=calculated_max_tokens,
                min_tokens=3*50,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                eos_prob_mul_factor=eos_prob_mul_factor,
                do_sample=do_sample,
                use_cache=True
            )
        return _generate_output(self.model, generated_codes, lengths)

    def text_to_music(self, caption, output_path, cfg_scale=10.0, temperature=1.0, max_audio_seconds=15,
                      top_p=1.0, cfg_filter_top_k=45, eos_prob_mul_factor=0.6, do_sample=True):
        """Generate music from text description.
        
        Args:
            caption (str): Text description of the music to generate
            output_path (str): Directory path to save the generated audio
            cfg_scale (float): Classifier-free guidance scale
            temperature (float): Sampling temperature
            max_audio_seconds (int): Maximum audio duration in seconds
            top_p (float): Top-p sampling parameter
            cfg_filter_top_k (int): Top-k filtering for CFG
            eos_prob_mul_factor (float): EOS probability multiplier
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Path to generated audio file, or None if failed
        """
        try:
            if not caption.strip():
                print("Please enter a music description.")
                return None

            print("Preparing text prompt...")
            neg_text = self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format("") + self.AUDIO_START
            pos_text = self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format(caption) + self.AUDIO_START
            source_input = self.tokenizer([neg_text, pos_text], add_special_tokens=False, return_tensors="pt", padding=True)

            result = {}

            def generation_target():
                try:
                    result['value'] = self._generate_audio_core(
                        source_input, None, cfg_scale=cfg_scale, temperature=temperature,
                        max_audio_seconds=max_audio_seconds, min_audio_seconds=10, top_p=top_p,
                        cfg_filter_top_k=cfg_filter_top_k, eos_prob_mul_factor=eos_prob_mul_factor,
                        do_sample=do_sample
                    )
                except Exception as e:
                    result['error'] = e

            thread = threading.Thread(target=generation_target)
            thread.start()

            animation_frames = itertools.cycle(["-", "\\", "|", "/"])
            start_time = time.time()
            while thread.is_alive():
                elapsed = time.time() - start_time
                print(f"Generating music... {next(animation_frames)} (Time: {elapsed:.1f}s)", end="\r")
                time.sleep(0.2)
            thread.join()

            if 'error' in result:
                raise result['error']
            audios = result['value']
            generation_time = time.time() - start_time

            print("\nSaving generated audio file...")
            output_file = os.path.join(output_path, f"generated_music_{int(time.time())}.wav")
            self.dac.decode(audios[0].transpose(0, 1).unsqueeze(0), save_path=output_file, min_duration=1)

            print(f"Music generation successful! Time: {generation_time:.2f}s File: {output_file}")
            return output_file

        except Exception as e:
            print(f"Generation failed: {str(e)}")
            return None

    def text_to_speech(self, target_text, reference_audio, reference_text, output_path, cfg_scale=1.0,
                       temperature=1.0, max_audio_seconds=30, top_p=1.0,
                       cfg_filter_top_k=45, eos_prob_mul_factor=1.0, do_sample=True):
        """Generate speech from text using reference audio for voice cloning.
        
        Args:
            target_text (str): Text to be spoken
            reference_audio (str): Path to reference audio file
            reference_text (str): Transcript of the reference audio
            output_path (str): Directory path to save the generated audio
            cfg_scale (float): Classifier-free guidance scale
            temperature (float): Sampling temperature
            max_audio_seconds (int): Maximum audio duration in seconds
            top_p (float): Top-p sampling parameter
            cfg_filter_top_k (int): Top-k filtering for CFG
            eos_prob_mul_factor (float): EOS probability multiplier
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Path to generated audio file, or None if failed
        """
        try:
            if not target_text.strip():
                print("Please enter the target text.")
                return None
            if reference_audio is None:
                print("Please provide a reference audio file.")
                return None
            if not reference_text.strip():
                print("Please enter the reference audio transcript.")
                return None

            if reference_audio and os.path.exists(reference_audio):
                safe_audio_path = os.path.join(self.TEMP_DIR, f"ref_audio_{int(time.time())}.wav")
                shutil.copy2(reference_audio, safe_audio_path)
                reference_audio = safe_audio_path

            print("Encoding reference audio...")
            prompt_codec = self._safe_encode_audio(reference_audio)
            prompt_codec_input_ids = self._preprocess_codec(
                codec=prompt_codec, codec_delay_pattern=self.model.config.codec_delay_pattern,
                codec_channels=self.model.num_channels, codec_bos_value=self.model.config.codec_bos_value,
                codec_eos_value=self.model.config.codec_eos_value, codec_pad_value=self.model.config.codec_pad_value
            )

            print("Preparing text prompt...")
            prompt_caption = f"<|SPEECH_PROMPT_START|>{reference_text}<|SPEECH_PROMPT_END|><|VOICE_PROMPT_START|>{'<|AUDIO_PLACEHOLDER|>' * prompt_codec_input_ids.shape[0]}<|VOICE_PROMPT_END|>"
            prompt_caption_fn = lambda x: f"{prompt_caption}<|SPEECH_START|>{x}<|SPEECH_END|>"
            neg_text = self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format(prompt_caption_fn("")) + self.AUDIO_START
            pos_text = self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format(prompt_caption_fn(target_text)) + self.AUDIO_START
            source_input = self.tokenizer([neg_text, pos_text], add_special_tokens=False, return_tensors="pt", padding=True)
            expanded_codec_ids = prompt_codec_input_ids.unsqueeze(0).expand(2, -1, -1).reshape(-1, prompt_codec_input_ids.shape[1])

            result = {}

            def generation_target():
                try:
                    result['value'] = self._generate_audio_core(
                        source_input, expanded_codec_ids, cfg_scale=cfg_scale, temperature=temperature,
                        max_audio_seconds=max_audio_seconds, top_p=top_p,
                        cfg_filter_top_k=cfg_filter_top_k, eos_prob_mul_factor=eos_prob_mul_factor,
                        do_sample=do_sample
                    )
                except Exception as e:
                    result['error'] = e

            thread = threading.Thread(target=generation_target)
            thread.start()

            animation_frames = itertools.cycle(["-", "\\", "|", "/"])
            start_time = time.time()
            while thread.is_alive():
                elapsed = time.time() - start_time
                print(f"Generating voice clone... {next(animation_frames)} (Time: {elapsed:.1f}s)", end="\r")
                time.sleep(0.2)
            thread.join()

            if 'error' in result:
                raise result['error']
            audios = result['value']
            generation_time = time.time() - start_time

            print("\nSaving cloned voice...")
            output_file = os.path.join(output_path, f"cloned_voice_{int(time.time())}.wav")
            self.dac.decode(audios[0].transpose(0, 1).unsqueeze(0), save_path=output_file, min_duration=1)

            print(f"Voice cloning successful! Time: {generation_time:.2f}s File: {output_file}")
            return output_file

        except Exception as e:
            print(f"Cloning failed: {str(e)}")
            return None

