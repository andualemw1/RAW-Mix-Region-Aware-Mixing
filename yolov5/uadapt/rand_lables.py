import os
import random

def generate_random_annotation(num_annotations=10):
    """
    Generate random YOLO format annotations with class and bounding box values.
    """
    annotations = []
    for _ in range(num_annotations):
        # Random class between 0 and 3
        class_id = random.randint(0, 3)

        # Random bounding box values: x_center, y_center, width, height
        x_center = round(random.uniform(0.1, 0.9), 6)
        y_center = round(random.uniform(0.1, 0.9), 6)
        width = round(random.uniform(0.01, 0.1), 6)
        height = round(random.uniform(0.01, 0.1), 6)

        annotation = f"{class_id} {x_center} {y_center} {width} {height}"
        annotations.append(annotation)

    return annotations

def create_annotation_file(image_filename, annotations, output_dir):
    """
    Creates a .txt annotation file with the same name as the image file in the specified directory.
    """
    base_name = os.path.splitext(image_filename)[0]  # Strip off the extension
    annotation_filename = os.path.join(output_dir, base_name + ".txt")

    with open(annotation_filename, 'w') as f:
        f.write("\n".join(annotations))

    print(f"Annotation file {annotation_filename} created with {len(annotations)} annotations.")

def generate_annotations_for_images(input_dir, output_dir, num_annotations=10):
    """
    Iterates through the images in the input directory and generates .txt annotation files
    in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image files in the directory (assuming typical image extensions)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        annotations = generate_random_annotation(num_annotations)
        create_annotation_file(image_file, annotations, output_dir)

    print(f"Processed {len(image_files)} images and created corresponding annotation files.")

# Example usage
input_directory = "C:/Users/anduw/Desktop/my_project/Experiment/yolov5/data/synt/uda"   # Replace with the folder containing your images
output_directory = "C:/Users/anduw/Desktop/my_project/Experiment/yolov5/data/synt/uda1"    # Replace with the folder where you want to save .txt files
generate_annotations_for_images(input_directory, output_directory, num_annotations=6)


