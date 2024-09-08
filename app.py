import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize and load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ResNet-18 model with pre-trained weights and modify the final layer
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(in_features=512, out_features=4)
model = model.to(device)
model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
model.eval()


# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# API to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get image from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['file']
    
    # Preprocess the image
    img = Image.open(file.stream)
    img = img.convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)

    # Get prediction from the model
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_idx = torch.max(output, 1)
    
    # Define your classes (replace this with your labels)
    labels = ['Blight', 'Common Rust', 'Grey Leaf Spot', 'Healthy']
    predicted_label = labels[predicted_idx.item()]
    
    # Return the prediction result
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8080)
