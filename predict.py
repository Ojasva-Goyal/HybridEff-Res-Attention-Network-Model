import torch
from torchvision import transforms
from PIL import Image
from model.hybrid_effres_attention import HybridModel
import argparse
import os
import pandas as pd

# Parse command-line arguments for test data and saved model weights path.
parser = argparse.ArgumentParser(description='Run the prediction script on your custom dataset.')
parser.add_argument('--model_path', type=str, required=True,
                    help='Relative or absolute path to the saved model weights')
parser.add_argument('--input_path', type=str, required=True,
                    help='Relative or absolute path to the test images folder.')
parser.add_argument('--output_path', type=str, default='./output',
                    help='Relative or absolute path to save the prediction results. Default is "./output".')
parser.add_argument('--output_type', type=str, choices=['csv', 'excel'], default='csv',
                    help='File type to save the results. Choices are "csv" or "excel". Default is "csv".')
parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu',
                    help='Device on which you are testing our model. Default is "cpu".')
args = parser.parse_args()

# Load model
model = HybridModel(num_classes=3)
model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Process all images in the input folder
input_folder = args.input_path
results = []

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add more extensions if needed
        image_path = os.path.join(input_folder, filename)
        prediction = predict(image_path)
        if prediction == 0:
            prediction_label = 'Brown-Rust'
        elif prediction == 1:
            prediction_label = 'Healthy Wheat Leaf'
        elif prediction == 2:
            prediction_label = 'Yellow-Rust'
        else:
            prediction_label = "ERROR !!!"
        print(f'Image: {filename}, Predicted class: {prediction_label}')
        results.append([filename, prediction_label])

# Save results to CSV or Excel
output_df = pd.DataFrame(results, columns=['Image', 'Predicted Class'])

if args.output_type == 'csv':
    output_df.to_csv(args.output_path + '.csv', index=False)
elif args.output_type == 'excel':
    output_df.to_excel(args.output_path + '.xlsx', index=False)

print(f'Results saved to {args.output_path}.{args.output_type}')
