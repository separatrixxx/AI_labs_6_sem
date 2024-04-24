from keras import utils
from keras import layers
from keras.models import Sequential
from keras.metrics import SparseTopKCategoricalAccuracy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import pandas as pd
import time
import seaborn as sn
from pathlib import Path


# def predict_image(model, img_path, class_names):
#     img = image.load_img(img_path, target_size=(256, 256))
#     img_array = image.img_to_array(img)
#     img_batch = np.expand_dims(img_array, axis=0)
#     img_preprocessed = preprocess_input(img_batch)
#     prediction = model.predict(img_preprocessed)
#     predicted_class = np.argmax(prediction, axis=1)

#     return class_names[predicted_class[0]]

def graf(title, label1, label2, epochs, val1, val2):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), val1, label=label1)
    plt.plot(range(epochs), val2, label=label2)
    plt.legend()
    plt.grid()
    plt.title(title)

    plt.show()

def main():
    url = 'http://www.soshnikov.com/permanent/data/petfaces.tar.gz'
    dataset = Path(utils.get_file('petfaces', origin=url, untar=True))
    # dataset = r"C:\gm\petfaces"

    img_size = 256, 256
    SEED = 3000

    train_set = utils.image_dataset_from_directory(
        dataset, 
        validation_split=0.2, 
        subset='training',
        seed=SEED,
        image_size=img_size,
        batch_size=32)

    test_set = utils.image_dataset_from_directory(
        dataset, 
        validation_split=0.2, 
        subset='validation',
        seed=SEED, 
        image_size=img_size,
        batch_size=32)

    class_names = train_set.class_names

    class_cats = []
    class_dogs = []

    for i in range(len(class_names)):
        if class_names[i].startswith('cat'):
            class_cats.append(i)
        if class_names[i].startswith('dog'):
            class_dogs.append(i)

    train_set = train_set.cache().shuffle(SEED//2).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_set = test_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = Sequential()

    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))
    model.add(layers.Conv2D(8, 3, activation='relu'))
    model.add(layers.Conv2D(16, 3, activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(len(class_names)))
    model.add(layers.Softmax())

    top3_accuracy = SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    start_time = time.time()

    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', top3_accuracy])

    epochs = 30

    history = model.fit(train_set, validation_data=test_set, epochs=epochs)

    train_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    train_top3_accuracy = history.history['top3_accuracy']
    test_top3_accuracy = history.history['val_top3_accuracy']

    # graf('Accuracy', 'train acc', 'test acc', epochs, train_accuracy, test_accuracy)
    # graf('Loss', 'train loss', 'test loss', epochs, train_loss, test_loss)
    # graf('Top-3 Accuracy', 'train top-3 acc', 'test top-3 acc', epochs, train_top3_accuracy, test_top3_accuracy)

    correct, total = 0, 0

    for x, y in test_set:
        y_pred = np.argmax(model.predict(x), axis=1)
        correct_cats = sum([y_pred[i] in class_cats and y[i] in class_cats for i in range(len(y))])
        correct_dogs = sum([y_pred[i] in class_dogs and y[i] in class_dogs for i in range(len(y))])
        correct += correct_cats + correct_dogs
        total += len(y)

    y_real = np.array([])
    y_pred = np.array([])

    for x, y in test_set:
        y_real = np.concatenate((y_real, y), axis=None)
        y_pred = np.concatenate((y_pred, np.argmax(model.predict(x), axis=1)), axis=None)    

    confusion_matrix = tf.math.confusion_matrix(y_real, y_pred, num_classes=len(class_names))
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in class_names], columns = [i for i in class_names])

    plt.figure(figsize = (8, 8))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    
    print('\nFinal train accuracy:', train_accuracy[epochs - 1])
    print('Final test accuracy:', test_accuracy[epochs - 1])
    print('\nFinal train loss:', train_loss[epochs - 1])
    print('Final test loss:', test_loss[epochs - 1])
    print('\nBinary classification accuracy:', correct / total)
    print('\nFinal train top-3 accuracy:', train_top3_accuracy[epochs - 1])
    print('Final test top-3 accuracy:', test_top3_accuracy[epochs - 1])

    end_time = time.time()
    print("\ntime =", end_time - start_time, '\n')

    # img_path = r"C:\gm\img.jpg"
    # predicted_class_name = predict_image(model, img_path, class_names)

    # print('\nPredicted class:', predicted_class_name)

main()
