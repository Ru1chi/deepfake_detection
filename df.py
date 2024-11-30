import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from transformers import ViTModel
import streamlit as st

# ---------------------------
# Define the DeepfakeModel
# ---------------------------
class DeepfakeModel(nn.Module):
    def __init__(self):
        super(DeepfakeModel, self).__init__()
        # Using MobileNetV2 for feature extraction
        self.mobilenet = models.mobilenet_v2(weights='DEFAULT')
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Identity()

        # Using ViT for additional feature extraction
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # Update LSTM input size
        self.lstm = nn.LSTM(input_size=num_ftrs * 2, hidden_size=256, num_layers=2, batch_first=True)

        # Final classification layer
        self.fc = nn.Linear(256, 1)

    def forward(self, x_frames, x_flows):
        batch_size = x_frames.size(0)
        mobilenet_outs_frames = []
        mobilenet_outs_flows = []

        # Process each frame and flow pair through MobileNetV2
        for t in range(x_frames.size(1)):
            mobilenet_out_frame = self.mobilenet(x_frames[:, t])
            mobilenet_out_flow = self.mobilenet(x_flows[:, t])
            mobilenet_outs_frames.append(mobilenet_out_frame.unsqueeze(1))
            mobilenet_outs_flows.append(mobilenet_out_flow.unsqueeze(1))

        # Concatenate outputs along the sequence dimension
        lstm_input = torch.cat((torch.cat(mobilenet_outs_frames, dim=1), torch.cat(mobilenet_outs_flows, dim=1)), dim=2)

        lstm_out, _ = self.lstm(lstm_input)
        final_output = self.fc(lstm_out[:, -1])
        return final_output

# ---------------------------
# Helper Functions
# ---------------------------

# Function to calculate a dummy optical flow
def calculate_dummy_optical_flow(image):
    dummy_flow = np.zeros_like(image, dtype=np.uint8)
    return dummy_flow

# Function to extract frames from a video
def extract_frames(video_path, frame_rate=1):
    """
    Extract frames from a video at the specified frame rate.
    :param video_path: Path to the video file.
    :param frame_rate: Extract 1 frame every 'frame_rate' seconds.
    :return: List of frames (images).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = int(fps * frame_rate)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

# Function to classify a single frame
def classify_frame(model, transform, frame):
    # Simulate optical flow using a dummy flow
    flow = calculate_dummy_optical_flow(frame)

    # Apply transformations
    img = transform(frame)
    flow = transform(flow)

    # Add batch dimension
    img = img.unsqueeze(0)
    flow = flow.unsqueeze(0)

    # Model expects inputs in a sequence format
    img = img.unsqueeze(1)  # Add sequence dimension
    flow = flow.unsqueeze(1)

    # Run inference
    with torch.no_grad():
        output = model(img, flow)
        prob = torch.sigmoid(output).item()  # Sigmoid applied to logits to get probability

    return prob

# Function to classify an entire video
def detect_deepfake_in_video(model, video_path):
    """
    Detect deepfake frames in a video and classify the video overall.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    # Extract frames from the video
    frames = extract_frames(video_path, frame_rate=1)

    # Run inference on each frame
    highest_prob = 0
    highest_label = None
    for frame in frames:
        prob = classify_frame(model, transform, frame)
        label = "FAKE" if prob > 0.5 else "REAL"

        # Keep track of the highest probability and its label
        if prob > highest_prob:
            highest_prob = prob
            highest_label = label

    return highest_label, highest_prob

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    # Set page title and description
    st.title("Deepfake Detection App")
    st.write("Upload a video to detect whether it's real or a deepfake.")

    # Upload video
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load the model
        st.write("Loading the model...")
        model_path = "C:/Users/nidhi/Desktop/DF/deepfake_detection_model_with_mobileNetV2_16.pth"  # Update this path
        model = DeepfakeModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Perform deepfake detection
        st.write("Processing the video...")
        label, prob = detect_deepfake_in_video(model, temp_video_path)

        # Display results
        st.write(f"**Video Classification Result:** {label}")
        st.write(f"**Confidence Score:** {prob:.2f}")

        # Clean up the temporary video file
        os.remove(temp_video_path)

if __name__ == "__main__":
    main()
