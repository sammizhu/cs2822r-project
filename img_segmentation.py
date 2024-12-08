import cv2
import os
import numpy as np

# Define input and output directories
input_folder = "input_images"  
output_folder = "output_images" 

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Load the image
        image_path = os.path.join(input_folder, file_name)
        original_face = cv2.imread(image_path)
        gray_face = cv2.cvtColor(original_face, cv2.COLOR_BGR2GRAY)

        # Initialize FAST feature detector
        fast = cv2.FastFeatureDetector_create()

        # Disable non-max suppression and detect keypoints
        fast.setNonmaxSuppression(False)
        keypoints_without_nonmax = fast.detect(gray_face, None)

        # Create blank images for keypoints
        blank_without_nonmax = np.zeros_like(original_face)

        # Draw keypoints on blank images
        cv2.drawKeypoints(blank_without_nonmax, keypoints_without_nonmax, blank_without_nonmax, color=(0, 255, 0),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the blank images with keypoints
        output_without_nonmax = os.path.join(output_folder, f"keypoints_without_nonmax_{file_name}")
        cv2.imwrite(output_without_nonmax, blank_without_nonmax)

        print(f"Processed and saved keypoints for: {file_name}")

print("Processing complete. Check the output folder for results.")
