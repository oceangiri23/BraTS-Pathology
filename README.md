# BraTS 2025 Pathological Image Classification 

## Overview

This repository contains  multi-modal classification solution for the **BraTS 2025 Pathology Classification Task**, which achieved **5th place globally** in the competition . Our approach combines image and text modalities using  deep learning models to classify pathological images into 10 distinct classes.

## Competition Information

The Brain Tumor Segmentation (BraTS) 2025 Pathology Classification challenge focuses on automated classification of brain tumor pathology from medical images. This task is crucial for assisting pathologists in accurate diagnosis and treatment planning.

🔗 **Competition Link**: [BraTS 2025 LightHouse Challenge](https://www.synapse.org/Synapse:syn64153130/wiki/631458)

## Architecture Overview

Our solution employs a **multi-modal fusion architecture** that leverages both visual and textual information:

### Key Components

1. **Image Encoder**: Fine-tuned ProgigaPath model encoder
2. **Text Encoder**: MidGemma for caption generation
3. **Fusion Module**: Cross-attention transformer
4. **Classification Head**: Multi-head classifier for 10 classes

### Architecture Diagram

```
[Pathological Image] ──────────► [Fine-tuned ProgigaPath] ──────► [Image Embeddings]
    │                                                                     │
    ▼                                                                     ▼
[MidGemma model]                                                   [Cross-Attention Transformer] ──► [Multi-head Classifier] ──► [10 Classes]
     |                                                                    ▲
     ▼                                                                    │
[Generated Caption]  ─────────────────────────────────────────► [Image Description]
```


## Model Components

### 1. Fine-tuned ProgigaPath Encoder
- **Purpose**: Extracts high-level visual features from pathological images
- **Base Model**: ProgigaPath (specialized for pathology images)
- **Fine-tuning**: Adapted for BraTS 2025 dataset characteristics
- **Output**: Dense image embeddings (feature vectors)

### 2. MidGemma Text Encoder
- **Purpose**: Generates contextual captions and processes textual information
- **Architecture**: Transformer-based language model
- **Function**: Converts image descriptions into semantic text embeddings
- **Integration**: Provides complementary textual context to visual features

### 3. Cross-Attention Transformer
- **Purpose**: Fuses multi-modal information (image + text)
- **Mechanism**: Attention-based feature alignment and integration
- **Benefits**: Captures complex relationships between visual and textual modalities

### 4. Multi-head Classification Layer
- **Output Classes**: 10 pathological categories
- **Architecture**: Multiple classification heads for robust predictions
- **Activation**: Softmax for probability distribution across classes

## System Requirements

- **GPU**: NVIDIA A6000 (48GB VRAM) 

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/brats2025-pathology-classification.git


conda env create -f environment.yml

# Activate the environment
conda activate pathology
```

## Model Setup

### Downloading Pre-trained Weights

1. **Download model weights **:
   ```bash
   # Download the fine-tuned weights from the huggingface: 
   Link : https://huggingface.co/Sagar32/Pathology_classification
   ```

2. **Model file structure**:
   ```
   Files/
   ├── Data2.jsonl
   ├── multimodel_classifier.pt
   ├── provgigapath_finetuned.ckpt
   ├── provgigapath_pretrained.bin
   └── slide_encoder_finetuned.pth
   ```

3. **Place models in correct locations**:
   ```bash
   # Ensure models are in the following structure:

   medgemma
   │  └── Data2.jsonl
   │
   provgigapath
   │   └── model  
   │         ├──provgigapath_finetuned.ckpt
   │         ├── provgigapath_pretrained.bin
   │         └── slide_encoder_finetuned.pth
   │ 
   Wmultimodel
   └── data
         └── checkpoints
                  └── withfinetunedimageencoder
                             └── multimodel_classifier.pt
  
   ```




## Performance Metrics

### Competition Results

| Metric | Score |
|--------|-------|
| **Global Rank** | **5th Place**  |
| **Recall (Macro)** | 75.2% |



### Hardware Configuration
- **GPU**: NVIDIA A6000 (48GB VRAM)



⭐ **Star this repository if you found it helpful!**
