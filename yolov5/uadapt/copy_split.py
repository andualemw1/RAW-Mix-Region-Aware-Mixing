import sys
# Add the parent directory of 'yolov5' to the Python path
sys.path.append('C:/Users/anduw/Desktop/my_project/ASST')

import os
import random
import shutil

def select_random_files_with_annotations(directory, num_files, output_directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter to include only image files with corresponding annotation files
    image_files = [f for f in files if f.endswith(('.jpg', '.png')) and os.path.exists(os.path.join(directory, os.path.splitext(f)[0] + '.txt'))]
    
    # Ensure there are enough image files with annotations
    if len(image_files) < num_files:
        raise ValueError(f"The directory contains fewer than {num_files} image files with annotations.")
    
    # Select random image files
    selected_images = random.sample(image_files, num_files)
    
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    for image in selected_images:
        # Get the base filename without extension
        base_name = os.path.splitext(image)[0]
        
        # Paths for the image and its annotation
        image_path = os.path.join(directory, image)
        annotation_path = os.path.join(directory, base_name + '.txt')
        
        # Copy the selected files to the output directory
        shutil.copy(image_path, os.path.join(output_directory, image))
        shutil.copy(annotation_path, os.path.join(output_directory, base_name + '.txt'))
        
    print(f"{num_files} image files and their annotations have been copied to {output_directory}.")



# Usage
source_directory = 'C:/Users/anduw/Desktop/my_project/Experiment/yolov5/data/synt/udax'
output_directory = 'C:/Users/anduw/Desktop/my_project/Experiment/yolov5/data/synt/uda'
number_of_files = 1180  # specify the number of files you want to select

select_random_files_with_annotations(source_directory, number_of_files, output_directory)

