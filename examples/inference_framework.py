#!/usr/bin/env python3
"""
UniMoE Audio Inference Framework

A flexible framework for batch audio generation with configurable parameters.
Supports both text-to-music and text-to-speech generation with voice cloning.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .utils.mod import UniMoEAudio


@dataclass
class TaskConfig:
    """Configuration for a single audio generation task"""
    task_type: str  # "text_to_music" or "text_to_speech"
    task_id: Optional[str] = None
    
    # Common parameters
    output_path: str = "./output"
    
    # Text-to-Music parameters
    caption: Optional[str] = None
    
    # Text-to-Speech parameters
    target_text: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None


@dataclass
class FrameworkConfig:
    """Configuration for the inference framework"""
    model_path: str
    device_id: int = 0
    output_base_dir: str = "./generated_audio"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    max_concurrent_tasks: int = 1
    

class InferenceFramework:
    """Main inference framework class"""
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.audio_generator = None
        self.logger = self._setup_logging()
        self.task_results = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('UniMoEAudio')
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def initialize_model(self) -> bool:
        """Initialize the UniMoE Audio model"""
        try:
            self.logger.info(f"Initializing UniMoE Audio model from {self.config.model_path}")
            self.audio_generator = UniMoEAudio(
                model_path=self.config.model_path,
                device_id=self.config.device_id
            )
            self.logger.info("Model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            return False
    
    def execute_task(self, task: TaskConfig) -> Dict:
        """Execute a single audio generation task"""
        if not self.audio_generator:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        task_id = task.task_id or f"{task.task_type}_{int(time.time())}"
        self.logger.info(f"Starting task {task_id}: {task.task_type}")
        
        result = {
            "task_id": task_id,
            "task_type": task.task_type,
            "status": "failed",
            "output_file": None,
            "error": None,
            "start_time": datetime.now().isoformat(),
            "end_time": None
        }
        
        try:
            # Ensure output directory exists
            os.makedirs(task.output_path, exist_ok=True)
            
            if task.task_type == "text_to_music":
                if not task.caption:
                    raise ValueError("Caption is required for text_to_music task")
                
                self.logger.info(f"Generating music: {task.caption}")
                output_file = self.audio_generator.text_to_music(
                    caption=task.caption,
                    output_path=task.output_path
                )
                
            elif task.task_type == "text_to_speech":
                if not all([task.target_text, task.reference_audio, task.reference_text]):
                    raise ValueError("target_text, reference_audio, and reference_text are required for text_to_speech task")
                
                if not os.path.exists(task.reference_audio):
                    raise FileNotFoundError(f"Reference audio file not found: {task.reference_audio}")
                
                self.logger.info(f"Generating speech: {task.target_text[:50]}...")
                output_file = self.audio_generator.text_to_speech(
                    target_text=task.target_text,
                    reference_audio=task.reference_audio,
                    reference_text=task.reference_text,
                    output_path=task.output_path
                )
                
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            if output_file:
                result["status"] = "success"
                result["output_file"] = output_file
                self.logger.info(f"Task {task_id} completed successfully: {output_file}")
            else:
                result["error"] = "Generation returned None"
                self.logger.error(f"Task {task_id} failed: Generation returned None")
                
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Task {task_id} failed: {e}")
        
        result["end_time"] = datetime.now().isoformat()
        self.task_results.append(result)
        return result
    
    def execute_batch(self, tasks: List[TaskConfig]) -> List[Dict]:
        """Execute a batch of audio generation tasks"""
        self.logger.info(f"Starting batch execution of {len(tasks)} tasks")
        
        results = []
        for i, task in enumerate(tasks, 1):
            self.logger.info(f"Processing task {i}/{len(tasks)}")
            result = self.execute_task(task)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        
        self.logger.info(f"Batch execution completed: {successful} successful, {failed} failed")
        return results
    
    def save_results(self, output_file: str) -> None:
        """Save task results to a JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.task_results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved to {output_file}")


def load_config_from_file(config_file: str) -> FrameworkConfig:
    """Load framework configuration from JSON or YAML file"""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        else:
            config_data = json.load(f)
    
    return FrameworkConfig(**config_data)


def load_tasks_from_file(tasks_file: str) -> List[TaskConfig]:
    """Load tasks from JSON or YAML file"""
    tasks_path = Path(tasks_file)
    
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
    
    with open(tasks_path, 'r', encoding='utf-8') as f:
        if tasks_path.suffix.lower() in ['.yaml', '.yml']:
            tasks_data = yaml.safe_load(f)
        else:
            tasks_data = json.load(f)
    
    return [TaskConfig(**task) for task in tasks_data]


def create_sample_config() -> None:
    """Create sample configuration files"""
    # Sample framework config
    framework_config = {
        "model_path": "path/to/your/model",
        "device_id": 0,
        "output_base_dir": "./generated_audio",
        "log_level": "INFO",
        "log_file": "inference.log",
        "max_concurrent_tasks": 1
    }
    
    with open("config_sample.json", 'w') as f:
        json.dump(framework_config, f, indent=2)
    
    # Sample tasks
    sample_tasks = [
        {
            "task_type": "text_to_music",
            "task_id": "music_001",
            "caption": "A peaceful piano melody with soft strings in the background",
            "output_path": "./output/music"
        },
        {
            "task_type": "text_to_speech",
            "task_id": "speech_001",
            "target_text": "Hello, this is a demonstration of voice cloning technology.",
            "reference_audio": "path/to/reference/audio.wav",
            "reference_text": "Original text from the reference audio",
            "output_path": "./output/speech"
        }
    ]
    
    with open("tasks_sample.json", 'w') as f:
        json.dump(sample_tasks, f, indent=2)
    
    print("Sample configuration files created:")
    print("- config_sample.json")
    print("- tasks_sample.json")


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description="UniMoE Audio Inference Framework")
    parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    parser.add_argument("--tasks", "-t", required=True, help="Path to tasks file")
    parser.add_argument("--output-results", "-o", help="Path to save results JSON file")
    parser.add_argument("--create-sample", action="store_true", help="Create sample configuration files")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config()
        return
    
    try:
        # Load configuration and tasks
        config = load_config_from_file(args.config)
        tasks = load_tasks_from_file(args.tasks)
        
        # Initialize framework
        framework = InferenceFramework(config)
        
        if not framework.initialize_model():
            return 1
        
        # Execute tasks
        results = framework.execute_batch(tasks)
        
        # Save results if specified
        if args.output_results:
            framework.save_results(args.output_results)
        
        # Print summary
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"\nExecution Summary:")
        print(f"Total tasks: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        return 0 if successful == len(results) else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())