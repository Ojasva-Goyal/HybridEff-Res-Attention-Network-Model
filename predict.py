import torch
from torchvision import transforms
from PIL import Image
from model.hybrid_effres_attention import HybridModel

# Load model
model = HybridModel(num_classes=3)
model.load_state_dict(torch.load('/path/to/saved/model.pth'))
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

# Example usage
image_path = '/path/to/image.jpg'
prediction = predict(image_path)
print(f'Predicted class: {prediction}')
