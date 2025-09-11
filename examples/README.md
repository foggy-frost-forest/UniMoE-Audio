# UniMoE Audio Inference Interface

This project provides multiple ways to use the UniMoE Audio model for audio generation, including a simplified inference function and a complete batch processing framework.

## File Overview

- `inference.py` - Simplified inference function for quick single-task calls
- `inference_framework.py` - Complete batch processing framework with configuration files
- `config_example.json/yaml` - Example configuration files for the framework
- `tasks_example.json/yaml` - Example task configuration files for batch processing

## Quick Start

### 1. Using the Simplified Inference Function

#### Command-Line Usage

```bash
# Generate music
python inference.py --task text_to_music --input "A peaceful piano melody" --output ./music_output --model /path/to/your/model

# Voice cloning
python inference.py --task text_to_speech --input "Hello world" --ref-audio ref.wav --ref-text "Reference text" --output ./speech_output --model /path/to/your/model
```

#### Programmatic Usage in Python

```python
from inference import inference

# Generate music
music_file = inference(
    task="text_to_music",
    input_text="A peaceful piano melody",
    output_path="./output",
    model_path="/path/to/your/model"
)

# Voice cloning
speech_file = inference(
    task="text_to_speech",
    input_text="Hello world",
    ref_audio="reference.wav",
    ref_text="Reference transcript",
    output_path="./output",
    model_path="/path/to/your/model"
)
```

### 2. Using the Batch Processing Framework

#### Prepare Configuration Files

1. Copy and modify `config_example.json`:
```json
{
  "model_path": "/path/to/your/model",
  "device_id": 0,
  "output_base_dir": "./generated_audio",
  "log_level": "INFO",
  "log_file": "inference.log"
}
```

2. Copy and modify `tasks_example.json`:
```json
[
  {
    "task_type": "text_to_music",
    "task_id": "music_001",
    "caption": "A peaceful piano melody",
    "output_path": "./output/music"
  },
  {
    "task_type": "text_to_speech",
    "task_id": "speech_001",
    "target_text": "Hello world",
    "reference_audio": "reference.wav",
    "reference_text": "Reference transcript",
    "output_path": "./output/speech"
  }
]
```

#### Run Batch Processing

```bash
python inference_framework.py --config config.json --tasks tasks.json --output-results results.json
```

## Parameter Descriptions

### Inference Function Parameters

- `task`: Task type, either "text_to_music" or "text_to_speech"
- `input_text`: Input text for generation
- `ref_audio`: Reference audio file path (required for text_to_speech)
- `ref_text`: Reference text (required for text_to_speech)
- `output_path`: Output directory path
- `model_path`: Path to the model
- `device_id`: GPU device ID
- `reuse_model`: Whether to reuse the loaded model instance (default: True)

### Command-Line Parameters

```
--task, -t          Task type (text_to_music or text_to_speech)
--input, -i         Input text for generation
--ref-audio, -ra    Reference audio file path
--ref-text, -rt     Reference text
--output, -o        Output directory path
--model, -m         Path to the model
--device, -d        GPU device ID
--no-reuse          Do not reuse the model instance
```

## Examples

### Generate Multiple Music Tracks

```bash
python inference.py -t text_to_music -i "Classical symphony" -o ./music -m /path/to/model
python inference.py -t text_to_music -i "Jazz piano solo" -o ./music -m /path/to/model
python inference.py -t text_to_music -i "Electronic dance music" -o ./music -m /path/to/model
```

### Batch Voice Cloning

```bash
python inference.py -t text_to_speech -i "Hello, how are you?" -ra ref.wav -rt "Reference" -o ./speech -m /path/to/model
python inference.py -t text_to_speech -i "Welcome to our service" -ra ref.wav -rt "Reference" -o ./speech -m /path/to/model
```

## Features

- **Model Reuse**: Automatically caches the model instance to avoid redundant loading
- **Flexible Invocation**: Supports both command-line and Python code usage
- **Complete Parameters**: Covers all required parameters with additional configuration options
- **Error Handling**: Comprehensive error handling and exception capturing
- **Batch Processing**: Handles large-scale tasks with the batch processing framework
- **Logging**: Full logging and progress tracking

## Notes

1. Ensure the model path is correct
2. For text_to_speech tasks, both `ref_audio` and `ref_text` are required
3. Ensure the reference audio file exists and is accessible
4. Output directories will be created automatically
5. Use the `clear_model()` function to release model memory when done