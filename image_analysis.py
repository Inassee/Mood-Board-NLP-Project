import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image

img_rows, img_cols = 300, 300
nb_classes = 3  # Number of classes for categorization
model_path = 'image_sentiment_model.h5'

def load_data():
    path1 = "images"
    path2 = "images_resized"
    listing = os.listdir(path1)
    num_samples = len(listing)
    immatrix = []
    labels = []

    for i, file in enumerate(listing):
        img = Image.open(os.path.join(path1, file))
        img = img.resize((img_rows, img_cols))
        gray = img.convert('L')
        gray.save(os.path.join(path2, file), "JPEG")
        immatrix.append(np.array(gray).flatten())
        if 'neg' in file:
            labels.append(1)  # Negative
        elif 'pos' in file:
            labels.append(0)  # Positive
        elif 'neu' in file:
            labels.append(2)  # Neutral

    data = np.array(immatrix)
    labels = np.array(labels)
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)
    data = data.astype('float32')
    data /= 255
    labels = to_categorical(labels, nb_classes)
    return data, labels

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    data, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=4)
    model = create_model((img_rows, img_cols, 1))
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))
    model.save(model_path)

def analyze_image_sentiment(image_path):
    img = load_img(image_path, target_size=(img_rows, img_cols), color_mode="grayscale")
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    prediction = model.predict(img)
    return np.argmax(prediction)

def get_model():
    global model
    return model