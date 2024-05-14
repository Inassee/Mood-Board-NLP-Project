# NLP Text-to-Image Mood Board Generator
## Project Overview
 
The NLP Text-to-Image Mood Board Generator is a web application designed to analyze user-provided text for emotional content and generate a corresponding mood board with images. The project utilizes a combination of natural language processing (NLP) and deep learning for sentiment analysis and image retrieval.

## Project Components
The project comprises several key components:

1. Text analysis: This module processes user input to determine the sentiment (positive, negative, or neutral) of the text.
2. Image retrieval: Based on the detected sentiment, this module retrieves images from a pre-labeled dataset.
3. Model training and prediction: A convolutional neural network (CNN) model is used to analyze and predict the sentiment of images in the dataset.
4. Streamlit application: A user-friendly web interface built with Streamlit for interacting with the tool.

## Detailed file descriptions

1. app.py
This is the main file for the Streamlit application. It includes the user interface and integrates the text analysis and image retrieval modules. Key functionalities include:

Displaying the app's logo and titles.
Accepting user input through a text area.
Displaying the resulting mood board with images.

2. text_analysis.py
This file contains the function to process and analyze the text input using the VADER sentiment analysis tool from the NLTK library. It determines the sentiment of the text as positive, negative, or neutral.

3. image_retrieval.py
This file handles the retrieval of images based on the sentiment detected from the user input. It randomly selects images from the dataset labeled with the corresponding sentiment.

4. image_analysis.py
This script includes the definition and training of the CNN model used for image sentiment analysis. It processes the image dataset, trains the model, and saves it for future use. Key components include:

Loading and preprocessing image data.
Defining and training the CNN model.
A function to predict the sentiment of individual images.

5. utils.py
A utility file containing the logging error function.

How to use?
-Setup: Ensure all required libraries are installed. You can use the provided requirements.txt file to install dependencies.
-Run the application: Execute app.py file using Streamlit:

```bash
streamlit run app.py 
```
-Input text: Enter any text in the provided text area. The application will analyze the text and generate a mood board based on the detected sentiment.
-View results: The mood board with corresponding images will be displayed below the text input area.

## Model details
The image sentiment analysis model is a convolutional neural network (CNN) built with TensorFlow and Keras and includes:

-Convolutional layers for feature extraction.
-MaxPooling layers for downsampling.
-Dropout layers for regularization.
-Dense layers for classification.

The model is trained on a dataset of images labeled as positive, negative, or neutral and the training process involves:

-Resizing images to 300x300 pixels.
-Normalizing pixel values.
-Splitting the dataset into training and testing sets.
-Training the mode over 10 epochs. 

## Limitations
-Text analysis: The accuracy of text sentiment analysis depends on the context and complexity of the input. Simple statements are more likely to be accurately classified.
-Image dataset: The quality and relevance of retrieved images depend on the pre-labeled dataset. The dataset may contain images that do not perfectly match the detected sentiment.


# DISCLAIMER
The dataset used may have inconsistencies, and the retrieved images are sometimes unexpected or unusual images.  

