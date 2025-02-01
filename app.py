import os
import cv2
import tempfile
import streamlit as st
import numpy as np
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock
import re

# Set up Google API Key
os.environ["GOOGLE_API_KEY"] = ""

# Initialize the Gemini model
gemini_pro = Gemini(model_name="models/gemini-1.5-flash")

# Streamlit UI
st.title("Real-time Object Detection with Gemini")
st.sidebar.write("Click 'Capture' to analyze an image.")

# Webcam capture
img_file_buffer = st.camera_input("Take a picture")

def process_image(image):
    """Processes the image, sends it to Gemini, and returns the annotated image."""
    
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Resize the image
    img_resized = cv2.resize(img, (600, 500))
    image_height, image_width = img_resized.shape[:2]

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, img_resized)
        image_path = tmp.name
    
    # Create message for Gemini
    msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="Return bounding boxes for Detect and return bounding boxes for all objects in the image, including people with specific attributes (e.g., person with glasses, person wearing a red shirt, person carrying a backpack, etc.), and provide details like their clothing or other features if visible. Format the output as: [ymin, xmin, ymax, xmax, object_name]. The object names should include specific descriptions (e.g., 'person with glasses', 'person in a red shirt', etc.) in the format: [ymin, xmin, ymax, xmax, object_name]. Include all objects, such as animals, vehicles, people,products and any other visible objectsin the image in the format:"
                           " [ymin, xmin, ymax, xmax, object_name]. Return response in text."),
            ImageBlock(path=image_path, image_mimetype="image/jpeg"),
        ],
    )

    # Get response from Gemini
    response = gemini_pro.chat(messages=[msg])

    # Extract bounding boxes
    bounding_boxes = re.findall(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\w\s]+)\]', response.message.content)

    list1 = []
    for box in bounding_boxes:
        parts = box.split(',')
        numbers = list(map(int, parts[:-1]))
        label = parts[-1].strip()
        list1.append((numbers, label))

    # Draw bounding boxes
    for numbers, label in list1:
        ymin, xmin, ymax, xmax = numbers
        x1 = int(xmin / 1000 * image_width)
        y1 = int(ymin / 1000 * image_height)
        x2 = int(xmax / 1000 * image_width)
        y2 = int(ymax / 1000 * image_height)

        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_resized

# If an image is captured, process and display it
if img_file_buffer is not None:
    st.sidebar.write("Processing image... Please wait.")
    processed_img = process_image(img_file_buffer)
    
    # Convert processed OpenCV image to Streamlit format
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    st.image(processed_img_rgb, caption="Detected Objects", use_container_width=True)
