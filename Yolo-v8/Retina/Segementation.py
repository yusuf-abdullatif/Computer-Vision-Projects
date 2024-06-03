# from ultralytics import YOLO
#
# # Load the YOLOv8 segmentation model
# model = YOLO('best.pt')
#
# # Perform inference on an image
# results = model('Images/mooree012.jpg')
#
# for result in results:
#     boxes = result.boxes  # Bounding boxes
#     masks = result.masks  # Segmentation masks
#     class_ids = result.class_ids  # Class IDs for each detected object
#     scores = result.scores  # Confidence scores
#
# # Visualize the results
# results.show()
#
# # Print the results
# for result in results:
#     print(f"Detected {len(result.boxes)} objects")
#     for box, mask, class_id, score in zip(result.boxes, result.masks, result.class_ids, result.scores):
#         print(f"Class: {class_id}, Score: {score}, Box: {box}, Mask: {mask}")


import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load the trained YOLO model
model = YOLO("best.pt")

# Directory paths
input_dir = "Images"
output_dir = "run"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to calculate the area in square meters from pixels
def calculate_area(pixels):
    return pixels * 0.001


# Iterate through all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Predict with the model
        results = model.predict(input_path)

        # Process each result
        for i, result in enumerate(results):
            # Get the annotated image
            annotated_image = result.plot()
            annotated_image_pil = Image.fromarray(annotated_image)
            draw = ImageDraw.Draw(annotated_image_pil)
            font = ImageFont.load_default()

            # Process detection results and draw the annotations
            for detection in result.boxes:
                x1, y1, x2, y2 = detection.xyxy[0].tolist()
                conf = detection.conf[0].tolist()
                cls = detection.cls[0].tolist()

                # Calculate the area of the bounding box
                width = x2 - x1
                height = y2 - y1
                area_pixels = width * height
                area_meters = calculate_area(area_pixels)

                # Prepare text with confidence and area
                text = f"                                               Area: {area_meters:.4f} m²"

                # Get text size using textbbox
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

                # Draw the text on the image (replace existing text)
                draw.text((x1, y1 - text_height), text, fill="red", font=font)

            # Save the image
            annotated_image_pil.save(output_path)

        print(f"Processed and saved: {filename}")
