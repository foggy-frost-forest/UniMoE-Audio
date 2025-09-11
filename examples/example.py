"""
UniMoE Audio Usage Example

This example demonstrates how to use the UniMoEAudio class for:
1. Text-to-Music generation
3. Voice Cloning
2. Text-to-Speech
"""

from utils.mod import UniMoEAudio

# Initialize the UniMoE Audio model
audio_generator = UniMoEAudio(model_path="path/to/model", device_id=0)

# Generate music
music_file = audio_generator.text_to_music(
    caption="A caption for music generation",
    output_path="./output"
)

# Voice Cloning
speech_file = audio_generator.text_to_speech(
    target_text="Hello world",
    reference_audio="reference audio path",
    reference_text="Reference transcript",
    output_path="./output",
    max_audio_seconds= 30,
)

#TTS (Voice Cloning with default refrence voice)
tts_file = audio_generator.text_to_speech(
    target_text="Hello world",
    reference_audio="asset/reference_audio/path",
    reference_text="Reference_transcript",
    output_path="./output",
    max_audio_seconds= 30,
)
