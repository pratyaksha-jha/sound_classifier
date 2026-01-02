import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
DATA_PATH = "data2.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    #convert list into np arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    mapping = data["mapping"]
    return inputs, targets, mapping

 #create train validation and test sets
def prepare_dataset(testsize, validationsize):
    #load data
    x, y, mapping = load_data(DATA_PATH)

    #create train/test split
    x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=testsize)
    #create train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validationsize)

    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    #x_train is a 4D array -> (num_samples, 130, 13, 1)

    return x_train, x_validation, x_test, y_train, y_validation, y_test, mapping


def build_model(input_shape):
    #create model
    model = keras.Sequential()

    #1 conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape= input_shape))
    #maxpooling layer to downsample our layer
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())
    
    #2 conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu"))
    #maxpooling layer to downsample our layer
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())

    #3 conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation="relu"))
    #maxpooling layer to downsample our layer
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())

    #flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def predict(model,mapping,x,y):
    x = x[np.newaxis, ...] #x is a 3D array in this case, model.predict expects a 4d array
    predictions = model.predict(x)
    #prediction is a 2d array and we get 10 diff values as the percent chance of each of the genres
    #extract index with max value
    predicted_index = np.argmax(predictions, axis=1)[0] #1d array with a value - index of predicted genre
    expected_genre = mapping[y]
    predicted_genre = mapping[predicted_index]
    
    print("expected output is: {} , predicted index is : {}".format(expected_genre, predicted_genre))

def plot_confusion_matrix(model, x_test, y_test, mapping):
    prediction = model.predict(x_test)
    predicted_indices = np.argmax(prediction, axis=1)
    
    cm = confusion_matrix(y_test, predicted_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=mapping, yticklabels=mapping)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    #prepare data
    x_train, x_validation, x_test, y_train, y_validation, y_test, mapping = prepare_dataset(0.25, 0.2)

    # build model
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape)

    # compile and train
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=30)

    model.save_weights("music_genre_model.weights.h5")
    print("Model weights saved to music_genre_model.h5")
    
    plot_confusion_matrix(model, x_test, y_test, mapping)
