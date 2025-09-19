
<!-- <p align="center">
    <img src="https://s21.ax1x.com/2024/05/14/pkmtDSS.png" width="250" style="margin-bottom: 0.2;"/>
<p> -->
<!-- <h2 align="center"> <a href="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/">Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts</a></h2> -->
<!-- <h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2>

<h4 align="center"> -->


# Welcome to the repo of **UniMoE-Audio**!

### UniMoE-Audio is An Unified Speech and Music Generation with Dynamic-Capacity Mixture of Experts

<!-- [![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-Uni_MoE-yellow)](https://huggingface.co/foggyforest/UniMoE-Audio-preview)
[![Project Page](https://img.shields.io/badge/Project_Page-Uni_MoE-blue)]()
[![Demo](https://img.shields.io/badge/Demo-Local-orange)]() 
[![Paper](https://img.shields.io/badge/Paper-arxiv-yellow)]() -->


## News

- [2025-09-10] 🔥 We released **UniMoE-Audio-preview**. 
<!-- Check out the [paper]() and [demo](). -->


## Performance show
<!-- 
🎵 **[🌐 Online Preview with Video Player](docs/music.html)** - Click to play MP4 files directly in your browser -->

| Prompt | Audio |
|:--:|:--:|
| This song contains several drum hits and percussive instruments playing a fast paced rhythm that motivates dancing along. An e-bass is bringing the low end supporting the drums. Cuatro guitars are strumming chords as a rhythmic addition. Trumpets are playing a loud and catchy melody. Some of the musical elements are slightly panned to the left and right side of the speakers. This song may be playing at a cheerful event. | <audio controls width="400" height="50"> <source src="https://github.com/foggy-frost-forest/UniMoE-Audio/raw/main/assets/audios/demo_1.mp3" type="audio/mpeg"> demo 1</audio> |
| This song contains a digital drum playing a simple pattern with a kick and a snare sound. Synthesizers are playing a repeating melody in the higher register. Another synth sound is playing a more aggressive lead sound with a countermelody. A string sample is being used to create a short hit. This song may be playing during a car ride. |<audio controls width="400" height="50"><source src="https://github.com/foggy-frost-forest/UniMoE-Audio/raw/main/assets/audios/demo_2.mp3" type="audio/mpeg"> demo 2</audio> |
| This is a four on the floor style of production. The song is a drum and bass type of song with a bright and fuzzy synth to add a melodic element. The first part of the song feels suspenseful. | <audio controls width="400" height="50"> <source src="https://github.com/foggy-frost-forest/UniMoE-Audio/raw/main/assets/audios/demo_3.mp3" type="audio/mpeg"> demo 3</audio>|
| This is a rock music piece. There is a medium-to-high pitched electric guitar solo at the forefront. In the melodic background, a keyboard and a bass guitar repeating the same pattern can be heard. The acoustic drums are playing a loud and slightly fast-paced rock drum beat. There is a rebellious atmosphere to this piece. It can be used in the soundtrack of a teenage drama or a crime shootout audio game. | <audio controls width="400" height="50"> <source src="https://github.com/foggy-frost-forest/UniMoE-Audio/raw/main/assets/audios/demo_4.mp3" type="audio/mpeg"> demo 4</audio> |
|||

https://github.com/user-attachments/assets/08a9ee44-f62b-465f-9793-7e2a45573d29

https://github.com/user-attachments/assets/83adee25-957e-4c81-bff2-108b78633db7

https://github.com/user-attachments/assets/910b1fb1-76c1-4743-9f83-36ecfcbf7f9b

https://github.com/user-attachments/assets/2881161e-f86f-463d-a910-06ab04799eae

## UniMoE-Audio

We propose UniMoE-Audio, a unified speech and music generation model based on a novel dynamic-capacity Mixture-of-Experts framework. By integrating a dynamic-capacity routing strategy for adaptive resource allocation and a hybrid expert design for functional decoupling, our architecture effectively mitigates the inherent task conflict between speech and music generation.

### Novelty and Contributions

---

### Dynamic-capacity MoE for Mitigating Task Conflict

**Key architectural optimizations:**

- **Dynamic-capacity Routing Strategy**  
  Dynamically adjusts the number of experts allocated to each token based on complexity, replacing conventional fixed-capacity routing with a Top-P sampling based approach.

- **Hybrid Expert Design**  
  - (1) Conditional dynamic experts for domain-specific knowledge  
  - (2) Constantly active shared experts for domain-agnostic features  
  - (3) Null experts enabling adaptive computation skipping  

<img src="assets/img/AudioLLM_model-MoE.png" alt="UniMoE-Audio Structure" style="max-width: 100%; width: 600px; height: auto; display: block; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(123, 179, 255, 0.15);">
<p align="center"><strong>Fig. 1</strong>  UniMoE-Audio Structure</p>

---

### Three-stage Training Curriculum for Data Imbalance

**Stages:**

- **Independent Specialist Training**  
  Pre-trains proto-experts on original, uncurated datasets to acquire domain-specific knowledge.

- **MoE Fusion and Warmup**  
  Integrates specialists into UniMoE-Audio using a curated, balanced dataset created via a data filtering pipeline.

- **Synergistic Joint Training**  
  End-to-end training on the curated balanced dataset, enabling effective cross-domain knowledge transfer.

<img src="assets/img/AudioLLM_training_moe.png" alt="UniMoE-Audio Training Pipeline" style="max-width: 40%; width: 600px; height: auto; display: block; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(123, 179, 255, 0.15);">
<p align="center"><strong>Fig. 2</strong>  UniMoE-Audio Training Pipeline</p>

## Installation
The following instructions are for Linux installation.

### 1. Clone this repository and navigate to the UniMoE Audio folder
```bash
git clone https://github.com/foggy-frost-forest/UniMoE-Audio.git
cd UniMoE-Audio 
```

### 2. Set up environment
We recommend using conda to install the environment.
```bash
conda env create -f configs/enviroment.yml      # add -n for your name
conda activate unimoe-audio                     # default name
```
then install the torch packages
  ```bash
   # Use the official index
   pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
   
   # Use Tsinghua mirror source
   pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --extra-index-url https://download.pytorch.org/whl/cu121
   
   # Use Alibaba Cloud mirror source
   pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cu121
   ```
A `dac model` is also required to be downloaded in '/path/to/UniMoE-Audio/utils/dac_model'.
It will be automatically downloaded when running the first time.

## UniMoE Audio Weights
`All weights` should be downloaded to ensure use.
After downloading all of them, organize the weights as follows in '/path/to/UniMoE-Audio-preview' folder:
```
models
└── UniMoE_Audio-preview
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── model-00003-of-00003.safetensors
```
## How to infer and deploy your demo

### 1.Make sure that all the weights are downloaded and the running environment is set correctly.

### 2.Run inference scripts:

`inference.py`: Simplified inference function for quick single-task calls.
```bash
conda activate unimoe-audio
cd examples

# Music Generating
python inference.py --task text_to_music --input "Caption about music" --output ./music_output --model /path/to/your/model

# Voice Cloning / TTS
python inference.py --task text_to_speech --input "Input text" --ref-audio ref.mp3 --ref-text "Reference text" --output ./speech_output --model /path/to/your/model
```

`inference_framework.py`: Complete batch processing framework with configuration files.
```bash
cd path/to/UniMoE-Audio
conda activate unimoe-audio
python inference_framework.py --config test_config.json --tasks test_tasks.json --output-results results.json
```
Details about json files can be found in the [examples/README.md](examples/README.md)

### To launch the online demo, run the following command:
Firstly, please install the web dependencies:
```bash
cd path/to/UniMoE-Audio
conda activate unimoe-audio
pip install -r configs/requirements_web.txt
```
```bash
python web_demo.py --model /path/to/your/model
```

<!-- ## How to evaluate on datasets
Evaluate the model on the datasets using the following command:
```bash

``` -->
