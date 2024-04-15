import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
import np_utils
import os
import cv2
import matplotlib.pyplot as plt
from numpy import asarray
from tqdm import tqdm
import shutil
# from sklearn.metrics import mean_squared_error
from util import mean_squared_error

class AutoEncoder:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, latent_size, model_path=None, history_path=None):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.latent_size = latent_size
        self.model_path = model_path
        self.history_path = history_path
        self.history = []
        self.autoencoder = self.create_model()
        
        if model_path and os.path.exists(model_path):
            self.autoencoder.load_weights(model_path)
        
        if history_path and os.path.exists(history_path):
            self.history = list(np.load(history_path))

    def create_model(self):
        x = Input(shape=(self.input_size,))
        hidden_1 = Dense(self.hidden_size_1, activation='relu')(x)
        h = Dense(self.latent_size, activation='relu')(hidden_1)
        hidden_2 = Dense(self.hidden_size_2, activation='relu')(h)
        r = Dense(self.input_size, activation='sigmoid')(hidden_2)
        autoencoder = Model(inputs=x, outputs=r)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def _train_step(self, images, val_images, epochs=100, batch_size=32):
        images = images.reshape((len(images), np.prod(images.shape[1:])))
        val_images = val_images.reshape((len(val_images), np.prod(val_images.shape[1:])))
        history = self.autoencoder.fit(images, images, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_images))
        loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        self.history.append([loss, val_loss])
        self.save()
        self.save_history()

    def get_latest_version(self):
        versions = np.sort([int(file.split("v")[-1].split(".")[0]) for file in os.listdir("models")])
        if len(versions) == 0:
            return 0
        return versions[-1]

    def clear_models(self):
        try:
            for file in os.listdir("models"):
                os.remove(f"models/{file}")
        except Exception as e:
            print("[-] Model clear failed!")
            print(e)

    def save(self):
        version = self.get_latest_version() + 1
        self.clear_models()
        model_path = f"models/model_v{version}.h5"
        self.autoencoder.save("temp.h5")
        shutil.move("temp.h5", model_path)
    
    def train(self, train_loader, validation_loader, epochs=100, batch_size=32):
        val_images = next(validation_loader)
        print(len(self.history))
        for images in tqdm(train_loader):
            self._train_step(images, val_images, epochs, batch_size)

    
    def save_history(self):
        np.save(self.history_path, self.history)
    
    def plot_history(self):
        train_history = [data[0] for data in self.history]
        val_history = [data[1] for data in self.history]
        plt.plot(train_history)
        plt.plot(val_history)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
    
    def predict(self, test_loader):
        images = next(test_loader)
        images = images.reshape((len(images), np.prod(images.shape[1:])))

        decoded_imgs = self.autoencoder.predict(images)

        n = 5
        plt.figure(figsize=(20, 6))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i+1)
            plt.imshow(images[i].reshape(256, 256))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


            # display reconstruction
            ax = plt.subplot(3, n, i+n+1)
            plt.imshow(decoded_imgs[i].reshape(256, 256))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # plt.imshow(decoded_imgs[1].reshape(256, 256))

        plt.show()
    
    def test_model(self, test_loader):
        images = next(test_loader)
        images = images.reshape((len(images), np.prod(images.shape[1:])))
        decoded_imgs = self.autoencoder.predict(images)

        mse = mean_squared_error(images, decoded_imgs)
        print(f"MSE: {mse}")

        return images, decoded_imgs
    
    # def save_history(self):
    #     np.save(self.history_path, self.history)