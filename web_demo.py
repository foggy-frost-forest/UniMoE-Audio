import sys
import os
import torch
import torchaudio
from transformers import AutoTokenizer
import numpy as np
import gradio as gr
import tempfile
import time
import shutil
from pathlib import Path
import json
from functools import lru_cache
from loguru import logger
import threading
import itertools
import argparse

from utils.mod import UniMoEAudio

# Global variables
audio_model = None

# Configuration parameters
MODEL_PATH = "./models/UniMoE-Audio-preview"
DEVICE_ID = 0

# Create output directories
OUTPUT_DIR = "./gradio_outputs"
TEMP_DIR = "./gradio_outputs/temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Set temporary directory environment variables
os.environ["GRADIO_TEMP_DIR"] = TEMP_DIR
os.environ["TMPDIR"] = TEMP_DIR



# Predefined examples
PREDEFINED_EXAMPLES = {
    "music-jazz": {
        "type": "music",
        "description": "Generate upbeat jazz music",
        "text": "A vibrant swing jazz tune featuring a walking bassline, rhythmic ride cymbals, and an improvised saxophone solo, full of fun and energy."
    },
    "music-lofi-hiphop": {
        "type": "music",
        "description": "Generate chill lo-fi hip hop beats",
        "text": "A chill lo-fi hip hop beat with a dusty vinyl crackle, mellow rhodes piano chords, a simple boom-bap drum loop, and a deep, relaxed bassline, perfect for studying or relaxing."
    },
    "voice-clone-greeting": {
        "type": "voice",
        "description": "Clone voice for friendly greeting",
        "text": "Hello! It's a pleasure to meet you. Please use my voice to create a digital version. Have a wonderful day!",
        "reference_text": "This is the audio I want to clone."
    },
    "voice-clone-storytelling": {
        "type": "voice",
        "description": "Clone voice for storytelling",
        "text": "Beyond the mountains and across the deep sea, there lay a forgotten magical kingdom where a great adventure was about to unfold.",
        "reference_text": "This is the audio I want to clone."
    }
}


def create_theme():
    """Creates a custom Gradio theme."""
    try:
        theme = gr.Theme.load("theme.json")
    except:
        theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="gray", font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"])
    return theme


def initialize_model():
    """Initialize UniMoE Audio model."""
    global audio_model
    if audio_model is not None: return
    if not torch.cuda.is_available(): raise RuntimeError("This application requires an NVIDIA GPU and CUDA environment.")
    logger.info("Initializing UniMoE Audio model...")
    audio_model = UniMoEAudio(MODEL_PATH, device_id=DEVICE_ID)
    logger.info("Model initialization complete!")



def generate_music(caption, cfg_scale=10.0, temperature=1.0, max_audio_seconds=10,
                   top_p=1.0, cfg_filter_top_k=45, eos_prob_mul_factor=1.0, do_sample=True):
    """Music generation function with animation."""
    try:
        if not caption.strip():
            yield gr.update(), gr.update(), "Please enter a music description."
            return
        
        yield gr.update(), gr.update(), "Generating music..."
        start_time = time.time()
        
        # Use UniMoEAudio class for generation
        output_path = audio_model.text_to_music(
            caption=caption,
            output_path=OUTPUT_DIR,
            cfg_scale=cfg_scale,
            temperature=temperature,
            max_audio_seconds=max_audio_seconds,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            eos_prob_mul_factor=eos_prob_mul_factor,
            do_sample=do_sample
        )
        
        generation_time = time.time() - start_time
        
        if output_path:
            success_msg = f"Music generation successful!\nTime taken: {generation_time:.2f}s\nFile: {os.path.basename(output_path)}"
            yield output_path, output_path, success_msg
        else:
            yield gr.update(), gr.update(), "Music generation failed."
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        yield gr.update(), gr.update(), f"Generation failed:\n{str(e)}"

def generate_voice_clone(target_text, reference_audio, reference_text, cfg_scale=1.0,
                         temperature=1.0, max_audio_seconds=30, top_p=1.0,
                         cfg_filter_top_k=45, eos_prob_mul_factor=1.0, do_sample=True):
    """Voice cloning function with animation."""
    try:
        if not target_text.strip():
            yield gr.update(), gr.update(), "Please enter the target text."
            return
        if reference_audio is None:
            yield gr.update(), gr.update(), "Please upload a reference audio file."
            return
        if not reference_text.strip():
            yield gr.update(), gr.update(), "Please enter the reference audio transcript."
            return

        yield gr.update(), gr.update(), "Generating voice clone..."
        start_time = time.time()
        
        # Use UniMoEAudio class for generation
        output_path = audio_model.text_to_speech(
            target_text=target_text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            output_path=OUTPUT_DIR,
            cfg_scale=cfg_scale,
            temperature=temperature,
            max_audio_seconds=max_audio_seconds,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            eos_prob_mul_factor=eos_prob_mul_factor,
            do_sample=do_sample
        )
        
        generation_time = time.time() - start_time
        
        if output_path:
            success_msg = f"Voice cloning successful!\nTime taken: {generation_time:.2f}s\nFile: {os.path.basename(output_path)}"
            yield output_path, output_path, success_msg
        else:
            yield gr.update(), gr.update(), "Voice cloning failed."

    except Exception as e:
        logger.error(f"Cloning error: {str(e)}")
        yield gr.update(), gr.update(), f"Cloning failed:\n{str(e)}"

def create_demo():
    """Create the Gradio demo interface with enforced left/right layout."""
    logger.info("Initializing model...")
    initialize_model()
    theme = create_theme()

    # CSS for left/right layout
    enhanced_css = """
    /* Force left/right split layout */
    .main-row {
        display: flex !important;
        flex-direction: row !important;
        gap: 30px !important;
        align-items: flex-start !important;
    }
    
    .left-column {
        flex: 3 !important;
        min-width: 600px !important;
        max-width: 800px !important;
        /* border-right: 3px solid #e0e0e0 !important; */
        padding-right: 20px !important;
    }
    
    .right-column {
        flex: 2 !important;
        min-width: 400px !important;
        background: #f8f9fa !important;
        padding: 25px !important;
        border-radius: 15px !important;
        border: 1px solid #e0e0e0 !important;
        position: sticky !important;
        top: 20px !important;
    }
    
    /* Main container style */
    .gradio-container { 
        max-width: 1600px !important; 
        margin: auto !important; 
        padding: 20px !important; 
    }
    
    /* Title style */
    .main-title { 
        text-align: center; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        background-clip: text; 
        font-size: 2.5em !important; 
        font-weight: bold !important; 
        margin-bottom: 0.5em !important; 
    }
    
    .subtitle { 
        text-align: center; 
        color: var(--body-text-color-subdued); 
        font-size: 1.2em; 
        margin-bottom: 2em; 
    }
    
    /* Button styles */
    .primary-button { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
        border: none !important; 
        color: white !important; 
        font-weight: bold !important; 
        border-radius: 8px !important; 
        padding: 12px 24px !important;
    }
    
    .secondary-button { 
        background: var(--background-fill-secondary) !important; 
        border: 1px solid var(--border-color-primary) !important; 
        color: var(--body-text-color) !important; 
        font-weight: 500 !important; 
        border-radius: 8px !important; 
    }
    
    /* Example row style */
    .example-row { 
        padding: 15px !important; 
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
        background: var(--background-fill-primary) !important;
    }
    
    /* Component spacing optimization */
    .input-group {
        margin-bottom: 20px !important;
        padding: 20px !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        background: white !important;
    }
    
    .output-section {
        background: white !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin-bottom: 15px !important;
        border: 1px solid #e0e0e0 !important;
    }

    .download-file-group .wrapper {
        padding-top: 5px !important;
        padding-bottom: 5px !important;
    }
    .download-file-group {
        padding-top: 5px !important;
        padding-bottom: 5px !important;
    }
    
    /* Mobile adaptation */
    @media (max-width: 1024px) { 
        .main-row {
            flex-direction: column !important;
        }
        
        .left-column {
            border-right: none !important;
            padding-right: 0 !important;
            margin-bottom: 20px !important;
            min-width: auto !important;
        }
        
        .right-column {
            position: static !important;
        }
    }
    """

    with gr.Blocks(css=enhanced_css, theme=theme, title="UniMoE Audio Studio") as demo:
        gr.HTML('<h1 class="main-title">UniMoE Audio Studio</h1>')
        # Main layout with forced left/right split
        with gr.Row(elem_classes=["main-row"]):
            
            with gr.Column(elem_classes=["left-column"]):
                # gr.HTML('<div style="border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-bottom: 20px;"><h3>Input Controls</h3></div>')
                gr.HTML('<div style="border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-bottom: 20px;"><h3>Input Controls</h3></div>')
                
                mode_switch = gr.Radio(
                    ["Voice Cloning", "Music Generation"],
                    label="Select Mode",
                    value="Voice Cloning",
                    container=True
                )

                with gr.Group(elem_classes=["input-group"]):
                    gr.Markdown("### Generation Inputs")
                    
                    main_text_input = gr.Textbox(
                        label="Target Text to Generate", 
                        placeholder="Enter the text you want the cloned voice to speak...", 
                        lines=3
                    )
                    
                    reference_audio_input = gr.Audio(
                        label="Reference Audio (Upload voice sample)", 
                        type="filepath"
                    )
                    
                    reference_text_input = gr.Textbox(
                        label="Reference Audio Transcript", 
                        placeholder="Enter exactly what is said in the reference audio...", 
                        lines=2
                    )

                with gr.Accordion("Advanced Settings (Voice Cloning)", open=False, visible=True) as voice_accordion:
                    with gr.Row():
                        voice_cfg_scale = gr.Slider(minimum=0.5, maximum=5.0, value=1.0, step=0.1, label="CFG Scale")
                        voice_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                    with gr.Row():
                        voice_max_seconds = gr.Slider(minimum=10, maximum=60, value=30, step=5, label="Max Audio Length (s)")
                        voice_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p Sampling")
                    with gr.Row():
                        voice_cfg_filter_top_k = gr.Slider(minimum=10, maximum=100, value=45, step=5, label="CFG Filter Top-k")
                        voice_eos_prob_mul = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="EOS Probability Factor")
                    voice_do_sample = gr.Checkbox(value=True, label="Enable Sampling")

                with gr.Accordion("Advanced Settings (Music Generation)", open=False, visible=False) as music_accordion:
                    with gr.Row():
                        music_cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, value=10.0, step=0.5, label="CFG Scale")
                        music_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                    with gr.Row():
                        music_max_seconds = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Max Audio Length (s)")
                        music_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p Sampling")
                    with gr.Row():
                        music_cfg_filter_top_k = gr.Slider(minimum=10, maximum=100, value=45, step=5, label="CFG Filter Top-k")
                        music_eos_prob_mul = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="EOS Probability Factor")
                    music_do_sample = gr.Checkbox(value=True, label="Enable Sampling")

                with gr.Row():
                    clear_btn = gr.Button("Clear All", elem_classes=["secondary-button"])
                    generate_btn = gr.Button("Generate", variant="primary", elem_classes=["primary-button"])

                with gr.Group(visible=True) as vc_examples_group:
                    gr.Markdown("### Example Templates (Voice Cloning)")
                    voice_examples_buttons = []
                    for key, example in PREDEFINED_EXAMPLES.items():
                        if example["type"] == "voice":
                            with gr.Row(variant="panel", elem_classes=["example-row"]):
                                with gr.Column(scale=4): 
                                    gr.Markdown(f"**{key.replace('-', ' ').title()}**\n\n*{example['description']}*")
                                with gr.Column(scale=1, min_width=80):
                                    btn = gr.Button("Use", size="sm", elem_classes=["secondary-button"])
                                    voice_examples_buttons.append((btn, key, example))

                with gr.Group(visible=False) as music_examples_group:
                    gr.Markdown("### Example Templates (Music Generation)")
                    music_examples_buttons = []
                    for key, example in PREDEFINED_EXAMPLES.items():
                        if example["type"] == "music":
                            with gr.Row(variant="panel", elem_classes=["example-row"]):
                                with gr.Column(scale=4): 
                                    gr.Markdown(f"**{key.replace('-', ' ').title()}**\n\n*{example['description']}*")
                                with gr.Column(scale=1, min_width=80):
                                    btn = gr.Button("Use", size="sm", elem_classes=["secondary-button"])
                                    music_examples_buttons.append((btn, key, example))
            
            with gr.Column(elem_classes=["right-column"]):
                gr.HTML('<div style="border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-bottom: 20px;"><h3>Output Results</h3></div>')
                
                with gr.Group(elem_classes=["output-section"]):
                    output_audio = gr.Audio(
                        label="Generated Audio", 
                        interactive=False, 
                        autoplay=False
                    )
                
                with gr.Group(elem_classes=["output-section", "download-file-group"]):
                    output_download = gr.File(
                        label="Download File", 
                        interactive=False
                    )
                
                with gr.Group(elem_classes=["output-section"]):
                    status_textbox = gr.Textbox(
                        label="Generation Status", 
                        interactive=False, 
                        lines=8, 
                        value="Ready to clone voice..."
                    )

        
        def update_ui_for_mode(mode):
            is_vc = mode == "Voice Cloning"
            if is_vc:
                return {
                    main_text_input: gr.update(label="Target Text to Generate", placeholder="Enter the text you want the cloned voice to speak..."),
                    reference_audio_input: gr.update(interactive=True),
                    reference_text_input: gr.update(interactive=True, placeholder="Enter exactly what is said in the reference audio..."),
                    voice_accordion: gr.update(visible=True),
                    music_accordion: gr.update(visible=False),
                    vc_examples_group: gr.update(visible=True),
                    music_examples_group: gr.update(visible=False),
                    status_textbox: gr.update(value="Ready to clone voice..."),
                }
            else: # Music Generation
                return {
                    main_text_input: gr.update(label="Music Description", placeholder="e.g., A vibrant swing jazz tune featuring a walking bassline..."),
                    reference_audio_input: gr.update(value=None, interactive=False),
                    reference_text_input: gr.update(value="", interactive=False, placeholder="-- Not used for Music Generation --"),
                    voice_accordion: gr.update(visible=False),
                    music_accordion: gr.update(visible=True),
                    vc_examples_group: gr.update(visible=False),
                    music_examples_group: gr.update(visible=True),
                    status_textbox: gr.update(value="Ready to generate music..."),
                }
        
        mode_switch.change(
            fn=update_ui_for_mode,
            inputs=mode_switch,
            outputs=[
                main_text_input, reference_audio_input, reference_text_input,
                voice_accordion, music_accordion,
                vc_examples_group, music_examples_group,
                status_textbox
            ]
        )

        def clear_all_inputs():
            return ("", None, "", None, None, "Ready...")

        clear_btn.click(
            fn=clear_all_inputs, 
            outputs=[main_text_input, reference_audio_input, reference_text_input, output_audio, output_download, status_textbox]
        )
        
        def on_generate_click(mode, main_text, ref_audio, ref_text, 
                              vc_cfg, vc_temp, vc_sec, vc_p, vc_k, vc_eos, vc_sample,
                              m_cfg, m_temp, m_sec, m_p, m_k, m_eos, m_sample):
            if mode == "Voice Cloning":
                yield from generate_voice_clone(main_text, ref_audio, ref_text, vc_cfg, vc_temp, vc_sec, vc_p, vc_k, vc_eos, vc_sample)
            else: 
                yield from generate_music(main_text, m_cfg, m_temp, m_sec, m_p, m_k, m_eos, m_sample)

        generate_btn.click(
            fn=on_generate_click, 
            inputs=[
                mode_switch, main_text_input, reference_audio_input, reference_text_input,
                voice_cfg_scale, voice_temperature, voice_max_seconds, voice_top_p, voice_cfg_filter_top_k, voice_eos_prob_mul, voice_do_sample,
                music_cfg_scale, music_temperature, music_max_seconds, music_top_p, music_cfg_filter_top_k, music_eos_prob_mul, music_do_sample
            ], 
            outputs=[output_audio, output_download, status_textbox], 
            show_progress="hidden"
        )
        
        def load_voice_example(text, ref_text, key):
             return {
                 main_text_input: gr.update(value=text),
                 reference_text_input: gr.update(value=ref_text),
                 output_audio: gr.update(value=None),
                 output_download: gr.update(value=None),
                 status_textbox: gr.update(value=f"Template loaded: {key.replace('-', ' ').title()}")
             }

        def load_music_example(text, key):
            return {
                main_text_input: gr.update(value=text),
                output_audio: gr.update(value=None),
                output_download: gr.update(value=None),
                status_textbox: gr.update(value=f"Template loaded: {key.replace('-', ' ').title()}")
            }

        for btn, key, example in voice_examples_buttons:
            btn.click(
                fn=load_voice_example, 
                inputs=[gr.Textbox(value=example["text"], visible=False), gr.Textbox(value=example.get("reference_text", ""), visible=False), gr.Textbox(value=key, visible=False)], 
                outputs=[main_text_input, reference_text_input, output_audio, output_download, status_textbox]
            )
        
        for btn, key, example in music_examples_buttons:
            btn.click(
                fn=load_music_example, 
                inputs=[gr.Textbox(value=example["text"], visible=False), gr.Textbox(value=key, visible=False)], 
                outputs=[main_text_input, output_audio, output_download, status_textbox]
            )

    return demo

def main():
    """Main function to parse arguments and launch the demo."""
    global MODEL_PATH, DEVICE_ID
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="UniMoE Audio Web Demo")
    parser.add_argument("--model", type=str, default="./models/UniMoE-Audio-preview", 
                        help="Path to the model directory (default: ./models/UniMoE-Audio-preview)")
    parser.add_argument("--device", type=int, default=0, 
                        help="CUDA device ID (default: 0)")
    parser.add_argument("--port", type=int, default=7860, 
                        help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", default=True, 
                        help="Enable Gradio sharing (default: True)")
    
    args = parser.parse_args()
    
    # Update global configuration with command line arguments
    MODEL_PATH = args.model
    DEVICE_ID = args.device
    
    print(f"Gradio version: {gr.__version__}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Device ID: {DEVICE_ID}")
    print(f"Server: {args.host}:{args.port}")
    
    demo = create_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, debug=False, show_api=False)


if __name__ == "__main__":
    main()