
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from torchvision import transforms
from PIL import Image
import base64
import io
import cv2
from io import BytesIO
image_height = 256
image_width = 256

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (image_height // 8) * (image_width // 8), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# Initialize Flask application
app = Flask(__name__)

# Load the trained models
denomination_model = CNN()
denomination_model.load_state_dict(torch.load('denomination.pth', map_location=torch.device('cpu')))
denomination_model.eval()

fake_real_model = load_model('real_or_fake.h5')

# Define class labels for denominations and fake/real classification
denomination_labels = ['10', '100', '1000', '20', '50', '500', '5000']
fake_real_labels = ['fake', 'real']

# Define a function to preprocess the image
def preprocess_image(base64_image, target_size=(150, 150)):
    try:
        # Decode base64 to bytes
        img_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None

def preprocess_image_pytorch(base64_image, target_size=(256, 256)):
    try:
        # Decode base64 to bytes
        img_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    except Exception as e:
        print(f"Error in preprocess_image_pytorch: {e}")
        return None


def preprocess_currency_image(image_data):
    try:
        # Decode base64 image data
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV BGR format

        # Backup the original image
        backup = img.copy()

        # Replace bright white pixels with neighboring pixels
        for i in range(len(img[:, 0, 0])):
            for j in range(len(img[0, :, 0])):
                R = int(img[i, j, 0])
                G = int(img[i, j, 1])
                B = int(img[i, j, 2])

                sum_col = R + G + B

                if (sum_col > 180) & (R > 200) & (G > 200) & (B > 200):
                    img[i, j, 0] = img[i - 1, j - 1, 0]
                    img[i, j, 1] = img[i - 1, j - 1, 1]
                    img[i, j, 2] = img[i - 1, j - 1, 2]

        # Define color range for masking
        lower = [np.mean(img[:, :, i] - np.std(img[:, :, i]) / 3) for i in range(3)]
        upper = [250, 250, 250]

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        # Threshold and find contours
        ret, thresh = cv2.threshold(mask, 40, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Crop the largest contour if any
        if len(contours) != 0:
            cv2.drawContours(output, contours, -1, 255, 3)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 5)

            # Crop the foreground
            cropped_image = backup[y:y + h, x:x + w]
        else:
            cropped_image = backup

        # Convert cropped image back to base64 for response
        retval, buffer = cv2.imencode('.jpg', cropped_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return base64_image

    except Exception as e:
        return str(e)


@app.route('/preProcess', methods=['POST'])
def preprocess():
    try:
        data = request.get_json()
        if 'base64Image' not in data:
            return jsonify({'error': 'No image provided'})

        base64_image = data['base64Image']
        processed_image = preprocess_currency_image(base64_image)

        return jsonify({'processedImage': processed_image})

    except Exception as e:
        return jsonify({'error': str(e)})
# Define a route for the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64 image from the request
        data = request.get_json()
        if 'base64Image' not in data:
            return jsonify({'error': 'No image provided'})

        base64_image = data['base64Image']

        # Preprocess the image
        img_array_real_fake = preprocess_image(base64_image)
        img_tensor_denomination = preprocess_image_pytorch(base64_image, target_size=(256, 256))

        if img_array_real_fake is None or img_tensor_denomination is None:
            return jsonify({'error': 'Error processing image'})

        # Make denomination prediction
        with torch.no_grad():
            denomination_prediction = denomination_model(img_tensor_denomination)
            denomination_index = torch.argmax(denomination_prediction, dim=1).item()
            predicted_denomination = denomination_labels[denomination_index]

        # Make fake/real prediction
        fake_real_prediction = fake_real_model.predict(img_array_real_fake)
        fake_real_class = fake_real_labels[int(np.round(fake_real_prediction[0]))]

        # Return the predictions as JSON response
        return jsonify({
            'denomination': predicted_denomination,
            'fake_or_real': fake_real_class
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5002)

