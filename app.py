import torch
import os
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Initialize and load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ResNet-18 model with updated weights parameter
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(in_features=512, out_features=4)
model = model.to(device)

# Save the entire model (once, when you have trained the model)
#torch.save(model, 'model.pth')

# Load the entire model
model = torch.load('model.pth', map_location=device)

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# API to handle image prediction
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    # Get image from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['file']
    
    # Preprocess the image
    img = Image.open(file.stream)
    img = img.convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # Move to device

    # Get prediction from the model
    with torch.no_grad():
        try:
            output = model(img_tensor)
            _, predicted_idx = torch.max(output, 1)
        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Return error details

    # Define your classes
    labels = ['Blight', 'Common Rust', 'Grey Leaf Spot', 'Healthy']
    predicted_label = labels[predicted_idx.item()]
    
    # Return the prediction result
    return jsonify({'prediction': predicted_label})


if __name__ == '__main__':
    # Set the port from the environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
