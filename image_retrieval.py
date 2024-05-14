import os
from PIL import Image
import random
from image_analysis import analyze_image_sentiment

def retrieve_images(emotion, sample_size=100, display_size=5):
    path = os.path.join(os.getcwd(), "images_resized")
    all_filenames = os.listdir(path)
    random.shuffle(all_filenames)
    sample_filenames = all_filenames[:sample_size]
    
    images = []
    print(f"Searching for images with emotion: {emotion}")

    for filename in sample_filenames:
        image_path = os.path.join(path, filename)
        try:
            sentiment = analyze_image_sentiment(image_path)
            print(f"Processed {filename}, detected sentiment: {sentiment}")
            if (emotion == 'positive' and sentiment == 0) or \
               (emotion == 'negative' and sentiment == 1) or \
               (emotion == 'neutral' and sentiment == 2):
                images.append(image_path)
                print(f"Image added: {filename}")
                if len(images) >= display_size:
                    break
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    print(f"Total images retrieved: {len(images)}")
    return images
