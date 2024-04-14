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

class AutoEncoder:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, latent_size):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.latent_size = latent_size
        self.autoencoder = self.create_model()

    def create_model(self):
        x = Input(shape=(self.input_size,))
        hidden_1 = Dense(self.hidden_size_1, activation='relu')(x)
        h = Dense(self.latent_size, activation='relu')(hidden_1)
        hidden_2 = Dense(self.hidden_size_2, activation='relu')(h)
        r = Dense(self.input_size, activation='sigmoid')(hidden_2)
        autoencoder = Model(inputs=x, outputs=r)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def train(self, images, epochs=100, batch_size=32):
        images = images.reshape((len(images), np.prod(images.shape[1:])))
        self.autoencoder.fit(images, images, epochs=epochs, batch_size=batch_size)