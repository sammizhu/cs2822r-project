import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

# Define the folder containing the images
folder_path = "/cs2822r-project/finalized_topk_imgs"  

# Get a list of image files in the folder
image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Sort the files to maintain a consistent order
image_files.sort()

# Number of images per row and rows needed per group
images_per_row = 3
images_per_group = 9
rows_per_group = images_per_group // images_per_row

# Split the image files into chunks of 9
chunks = [image_files[i:i + images_per_group] for i in range(0, len(image_files), images_per_group)]

# Process each chunk and save it as a separate output image
for chunk_index, chunk in enumerate(chunks):
    # Create a figure for the current chunk
    fig, axs = plt.subplots(rows_per_group, images_per_row, figsize=(10, rows_per_group * 3))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    # Iterate through the images in the chunk
    for idx, file_name in enumerate(chunk):
        img_path = os.path.join(folder_path, file_name)
        img = imread(img_path)
        axs[idx].imshow(img)
        axs[idx].axis('off')
        axs[idx].set_title(file_name, fontsize=12)

    # Hide any unused subplots
    for idx in range(len(chunk), len(axs)):
        axs[idx].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    output_path = f"output_{chunk_index + 1}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")