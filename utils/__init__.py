# -*- coding: utf-8 -*-
"""
UniMoE Audio Utils Package

This package contains the merged utilities for UniMoE Audio model:
- UniMoE_Audio_utils: DAC utilities, DeepSpeed MoE inference utilities, and matrix compression utilities
- UniMoE_Audio_core: MoE core components including GRINMoESparseMoeBlock and related classes
- UniMoE_Audio_model: Main model classes including UniAudioRVQQwen2_5VLMoEForConditionalGeneration
- UniMoE_Audio_mod: High-level interface and convenience functions

Usage:
    # Quick start with high-level interface
    from utils import UniMoEAudio, load_unimoe_audio
    
    # Load model
    model = load_unimoe_audio("path/to/model")
    
    # Generate music
    audio = model.generate_music("A peaceful piano melody")
    
    # Generate speech
    audio = model.generate_speech("Hello world", voice_description="friendly female voice")

    # Low-level access to components
    from utils import GRINMoESparseMoeBlock, UniAudioRVQQwen2_5VLMoEForConditionalGeneration
"""

import warnings
from typing import Optional, List, Dict, Any

# Version information
__version__ = "1.0.0"
__author__ = "UniMoE Audio Team"
__description__ = "UniMoE Audio: Unified Multimodal Expert Audio Generation"

# Import utilities with error handling
try:
    from .UniMoE_Audio_utils import (
        # DAC utilities
        DAC,
        build_revert_indices,
        revert_audio_delay,
        _prepare_audio_prompt,
        
        # Delay pattern utilities
        apply_delay_pattern,
        revert_delay_pattern,
        
        # Matrix compression utilities
        compress_matrix,
        decompress_matrix,
        
        # DeepSpeed utilities
        DecoderOutput,
        _generate_output,
    )
except ImportError as e:
    warnings.warn(f"Failed to import from UniMoE_Audio_utils: {e}")
    # Define dummy classes to prevent import errors
    class DAC: pass
    def build_revert_indices(*args, **kwargs): pass
    def revert_audio_delay(*args, **kwargs): pass
    def apply_delay_pattern(*args, **kwargs): pass
    def revert_delay_pattern(*args, **kwargs): pass
    def compress_matrix(*args, **kwargs): pass
    def decompress_matrix(*args, **kwargs): pass

# Import core MoE components with error handling
try:
    from .UniMoE_Audio_core import (
        # MoE blocks
        GRINMoESparseMoeBlock,
        MoE,
        MOELayer,
        Experts,
        
        # Expert selection and routing
        sparsemixer,
        dynamic_expert_selection,
        cal_global_weight,
        
        # Loss functions
        load_balancing_loss_func,
        
        # MLP variants
        SharedExpertMLP,
        DynamicExpertMLP,
        NULLExpertMLP,
    )
except ImportError as e:
    warnings.warn(f"Failed to import from UniMoE_Audio_core: {e}")
    # Define dummy classes
    class GRINMoESparseMoeBlock: pass
    class MoE: pass
    def load_balancing_loss_func(*args, **kwargs): pass
    def sparsemixer(*args, **kwargs): pass
    def dynamic_expert_selection(*args, **kwargs): pass

# Import model classes with error handling
try:
    from .UniMoE_Audio_model import (
        # Configuration classes
        UniAudioRVQQwen2_5VLMoEConfig,
        Qwen2_5_VLMoETextConfig,
        Qwen2_5_VLConfig,
        Qwen2_5_VLVisionConfig,
        
        # Model classes
        UniAudioRVQQwen2_5VLMoEForConditionalGeneration,
        Qwen2_5_VLMoEDecoderLayer,
        Qwen2_5_VLMoETextModel,
        Qwen2_5_VLMoEPreTrainedModel,
        
        # Vision components
        Qwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VisionPatchEmbed,
        Conv3D,
        
        # Output classes
        MoEQwen2_5VLCausalLMOutputWithPast,
    )
except ImportError as e:
    warnings.warn(f"Failed to import from UniMoE_Audio_model: {e}")
    # Define dummy classes
    class UniAudioRVQQwen2_5VLMoEConfig: pass
    class UniAudioRVQQwen2_5VLMoEForConditionalGeneration: pass
    class Qwen2_5_VLMoETextConfig: pass
    class MoEQwen2_5VLCausalLMOutputWithPast: pass

# Import high-level interface with error handling
try:
    from .UniMoE_Audio_mod import (
        # Main interface class
        UniMoEAudio,
        
        # Convenience functions
        load_unimoe_audio,
        generate_music_from_text,
        generate_speech_from_text,
        
        # Demo functions
        demo_music_generation,
        demo_speech_generation,
    )
except ImportError as e:
    warnings.warn(f"Failed to import from UniMoE_Audio_mod: {e}")
    # Define dummy classes
    class UniMoEAudio: pass
    def load_unimoe_audio(*args, **kwargs): pass
    def generate_music_from_text(*args, **kwargs): pass
    def generate_speech_from_text(*args, **kwargs): pass

# Define comprehensive __all__ list
__all__ = [
    # Version and metadata
    "__version__",
    "__author__", 
    "__description__",
    
    # DAC and audio utilities
    "DAC",
    "build_revert_indices", 
    "revert_audio_delay",
    "_prepare_audio_prompt",
    "apply_delay_pattern",
    "revert_delay_pattern",
    
    # Matrix compression utilities
    "compress_matrix",
    "decompress_matrix",
    
    # DeepSpeed utilities
    "DecoderOutput",
    "_generate_output",
    
    # Core MoE components
    "GRINMoESparseMoeBlock",
    "MoE",
    "MOELayer", 
    "Experts",
    "load_balancing_loss_func",
    "sparsemixer",
    "dynamic_expert_selection",
    "cal_global_weight",
    
    # MLP variants
    "SharedExpertMLP",
    "DynamicExpertMLP", 
    "NULLExpertMLP",
    
    # Configuration classes
    "UniAudioRVQQwen2_5VLMoEConfig",
    "Qwen2_5_VLMoETextConfig",
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLVisionConfig",
    
    # Model classes
    "UniAudioRVQQwen2_5VLMoEForConditionalGeneration",
    "Qwen2_5_VLMoEDecoderLayer",
    "Qwen2_5_VLMoETextModel", 
    "Qwen2_5_VLMoEPreTrainedModel",
    
    # Vision components
    "Qwen2_5_VisionTransformerPretrainedModel",
    "Qwen2_5_VisionPatchEmbed",
    "Conv3D",
    
    # Output classes
    "MoEQwen2_5VLCausalLMOutputWithPast",
    
    # High-level interface
    "UniMoEAudio",
    "load_unimoe_audio",
    "generate_music_from_text", 
    "generate_speech_from_text",
    "demo_music_generation",
    "demo_speech_generation",
]

# Convenience aliases for backward compatibility
UnimoeAudio = UniMoEAudio  # Alternative naming
load_model = load_unimoe_audio  # Shorter alias

# Add aliases to __all__
__all__.extend(["UnimoeAudio", "load_model"])

def get_version() -> str:
    """Get the current version of UniMoE Audio."""
    return __version__

def list_available_models() -> List[str]:
    """List available model types."""
    return [
        "UniAudioRVQQwen2_5VLMoEForConditionalGeneration",
        "Qwen2_5_VLMoETextModel",
    ]

def get_model_info() -> Dict[str, Any]:
    """Get information about the UniMoE Audio package."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "available_models": list_available_models(),
        "components": {
            "utils": "DAC utilities, matrix compression, DeepSpeed utilities",
            "core": "MoE components, expert routing, load balancing",
            "model": "Main model classes and configurations", 
            "mod": "High-level interface and convenience functions"
        }
    }

# Add utility functions to __all__
__all__.extend(["get_version", "list_available_models", "get_model_info"])

# Package initialization message
def _show_welcome_message():
    """Show welcome message when package is imported."""
    try:
        print(f"🎵 UniMoE Audio v{__version__} loaded successfully!")
        print("📖 Quick start: from utils import load_unimoe_audio")
        print("🔗 Documentation: Use help(utils) for more information")
    except:
        pass  # Silently fail if print is not available

# Show welcome message on import (can be disabled by setting environment variable)
import os
if not os.environ.get("UNIMOE_AUDIO_QUIET", False):
    _show_welcome_message()