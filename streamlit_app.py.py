import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from PIL import Image
import time
import base64
from io import BytesIO


# Define the CNN model structure
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Load the trained model
model = CNN()
model.load_state_dict(
    torch.load("cnn_model.pkl", map_location=torch.device("cpu"))
)
model.eval()

# Define a transformation to preprocess the input image
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize
    ]
)


# Define a function to predict the label of an image
def predict_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
    return predicted.item()


# Streamlit app UI
st.markdown(
    "<h1 style='text-align: center; color: blue;'>Digit Recognition App</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Upload an image of a handwritten digit (0-9) from MNIST dataset, and the model will predict the label.</p>",
    unsafe_allow_html=True,
)

# Add a placeholder for loading animation
placeholder = st.empty()

def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Initialize the placeholder for the progress bar
    progress_bar = placeholder.progress(0)
    
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Simulate the loading process with a progress bar
    for percent_complete in range(100):
        time.sleep(0.01)  # Adjust time to control speed of the progress
        progress_bar.progress(percent_complete + 1)
    
    # After loading is complete, remove the progress bar
    placeholder.empty()
    
    # Get prediction
    label = predict_image(image)

    # Resize the uploaded image to a smaller dimension (e.g., 200x200)
    resized_image = image.resize((200, 200))
    
    # Convert the resized image to base64
    img_base64 = pil_to_base64(resized_image)

    # Display the resized image within a styled block
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid #ccc;
            background-color: #f2f2f2;
            padding: 15px;
            border-radius: 10px;
            width: 250px;
            margin: 20px auto;
        ">
            <img src="data:image/png;base64,{img_base64}" style="width: 200px; height: 200px; border-radius: 5px;" />
        </div>
        """,
        unsafe_allow_html=True
    )

    # Styled output box for the predicted label
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid #4CAF50;
            background-color: #f0f8ff;
            color: #4CAF50;
            padding: 10px;
            font-size: 24px;
            font-weight: bold;
            border-radius: 8px;
            width: 200px;
            margin: 20px auto;
        ">
            Predicted Label: {label}
        </div>
        """,
        unsafe_allow_html=True
    )
