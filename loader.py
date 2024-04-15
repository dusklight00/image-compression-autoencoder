import os
import cv2
import numpy as np
from tqdm import tqdm

class DatasetLoader:
    def __init__(self, dataset_path, image_size=(256, 256), train_split=0.7, test_split=0.2, validation_split=0.1, batch_size=32, shuffle=False):
        self.dataset_path = dataset_path
        
        self.IMAGE_WIDTH = image_size[0]
        self.IMAGE_HEIGHT = image_size[1]
        
        self.train_split = train_split
        self.test_split = test_split
        self.validation_split = validation_split
        self.batch_size = batch_size
        
        self.train_image_paths = []
        self_test_image_paths = []
        self.validation_image_paths = []

        image_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path)]
        
        if shuffle:
            np.random.shuffle(image_paths)

        train_image_index = int(len(image_paths) * self.train_split)
        test_image_index = int(len(image_paths) * self.test_split)
        validation_image_index = int(len(image_paths) * self.validation_split)

        self.train_image_paths = image_paths[:train_image_index]
        self.test_image_paths = image_paths[train_image_index:train_image_index + test_image_index]
        self.validation_image_paths = image_paths[train_image_index + test_image_index:]
    
    def load_image_data(self, image_paths):
        images = []
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            np_image = np.asarray(image)
            np_image.reshape(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)
            image_arr = np_image.tolist()
            images.append(image_arr)
        
        images = np.array(images)
        images = images.reshape(images.shape[0], self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)
        images = images.astype("float32") / 255.
        return images
    
    def train_image_loader(self):
        for i in tqdm(range(0, len(self.train_image_paths), self.batch_size)):
            images = self.load_image_data(self.train_image_paths[i:i + self.batch_size])
            yield images
    
    def test_image_loader(self):
        for i in tqdm(range(0, len(self.test_image_paths), self.batch_size)):
            images = self.load_image_data(self.test_image_paths[i:i + self.batch_size])
            yield images
    
    def validation_image_loader(self):
        for i in tqdm(range(0, len(self.validation_image_paths), self.batch_size)):
            images = self.load_image_data(self.validation_image_paths[i:i + self.batch_size])
            yield images