# Hybrid EffRes Attention Network Model

## Overview
The Hybrid EffRes Attention Network Model (HEANM) is a novel deep learning architecture combining EfficientNet, ResNet, and attention mechanisms to achieve superior performance on custom datasets. This repository contains the implementation of HEANM, along with training and evaluation scripts.

## Features
- Combines EfficientNet and ResNet architectures
- Incorporates attention mechanisms to enhance performance
- Customizable and extensible
- Provided with pre-trained weights and example notebooks

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Clone the repository:
    ```bash
    https://github.com/Ojasva-Goyal/HybridEff-Res-Attention-Network-Model.git
    cd Hybrid-EffRes-Attention-Network
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Model Architecture
The architecture combines EfficientNet and ResNet models with attention mechanisms to improve feature extraction and classification accuracy.

### Training the Model
1. Prepare your dataset as per the guidelines in the `data/` directory.
2. Configure the training parameters in `config/config.yaml`.
3. Run the training script:
    ```bash
    python training/train.py --config config/config.yaml
    ```

### Running Inference
To run inference using a trained model:
```bash
python inference/predict.py --model_path saved_models/your_model.pth --input_path path/to/your/input

