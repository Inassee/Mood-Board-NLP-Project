import os
from PIL import Image
import random

def retrieve_images(emotion):
    path = "C:/Users/ninai/OneDrive/Documents/MB/images_resized"
    images = []
    print(f"Searching for images with emotion: {emotion}")
    for filename in os.listdir(path):
        print(f"Checking file: {filename}")  # Debug: list files being checked
        # Adjust the matching logic here based on actual filename patterns
        if (emotion == 'positive' and 'pos' in filename.lower()) or \
           (emotion == 'negative' and 'neg' in filename.lower()):
            image_path = os.path.join(path, filename)
            images.append(Image.open(image_path))
            print(f"Image added: {filename}")  # Confirm which images are added
            if len(images) == 40: #devolver 5 randoms de las 40
                random.shuffle(images)
                images = images[:5]
                break
    print(f"Total images retrieved: {len(images)}")  # Debug: Count of images retrieved
    return images
