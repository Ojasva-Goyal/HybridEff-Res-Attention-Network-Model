# Hybrid EffRes Attention Network Model

## Overview
The Hybrid EffRes Attention Network Model (HEANM) is a novel deep learning architecture combining EfficientNet, ResNet, and attention mechanisms to achieve superior performance on custom datasets. This repository contains the implementation of HEANM, along with training and evaluation scripts.

## Features
- Combines EfficientNet and ResNet architectures
- Incorporates attention mechanisms to enhance performance
- Customizable and extensible
- Provided with pre-trained weights and example scripts.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Training](training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](contributing)
- [License](#license)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Ojasva-Goyal/HybridEff-Res-Attention-Network-Model.git
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
python predict.py --model_path path_to_your_model.pth --input_path path_to_test_images --output_path output --output_type csv
```

    - `--model_path`: Path to the saved model weights.
    - `--input_path`: Path to the folder containing test images.
    - `--output_path`: Path where the output CSV or Excel file will be saved. Default is "./output".
    - `--output_type`: Type of file to save the results. Choices are "csv" or "excel". Default is "csv".

#### Example

```sh
python predict.py --model_path ./models/model.pth --input_path ./test_images --output_path ./results --output_type excel
```

### Training
Detailed instructions for training the model on your custom dataset can be found in `training/README.md`.

### Evaluation
Evaluation scripts and metrics are provided in the `evaluation/` directory. Run the evaluation script to see the performance of the trained model:

``` bash
python evaluation/evaluate.py --model_path saved_models/your_model.pth --data_path path/to/your/test_data
```
## Results
Performance Comparison:

| Model         | Accuracy      |  Precision     |  Recall       |  F1-Score |
| ------------- | ------------- |  ------------- | ------------- | ------------- |
| Hybrid EffRes Attention Network Model  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B0  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B1  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B2  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B3  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B4  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B5  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B6  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B7  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B8  | Content Cell  | ------------- | ------------- | ------------- |
| EfficientNet B9  | Content Cell  | ------------- | ------------- | ------------- |
| MobileNet  | Content Cell  | ------------- | ------------- | ------------- |
| MobileNetv2  | Content Cell   | ------------- | ------------- | ------------- |
| MobileNetv3_Small  | Content Cell  | ------------- | ------------- | ------------- |
| MobileNetv3_Large  | Content Cell | ------------- | ------------- | ------------- |
| ResNet 50  | Content Cell  | ------------- | ------------- | ------------- |
| ResNet 152  | Content Cell  | ------------- | ------------- | ------------- |
| Inception v3  | Content Cell  | ------------- | ------------- | ------------- |
| DebseNet 121  | Content Cell  | ------------- | ------------- | ------------- |
| VGG 16  | Content Cell  | ------------- | ------------- | ------------- |


## Contributing
We welcome contributions to improve this project! Please see the CONTRIBUTING.md file for guidelines.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Include any acknowledgements, references to your paper, or external libraries used.
## Contact
For any questions or suggestions, feel free to reach out to your email.


