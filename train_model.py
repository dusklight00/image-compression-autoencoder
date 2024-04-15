from loader import DatasetLoader
from model import AutoEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

loader = DatasetLoader(
    dataset_path="dataset", 
    image_size=(256, 256), 
    train_split=0.7, 
    test_split=0.2, 
    validation_split=0.1, 
    batch_size=32
)

train_loader = loader.train_image_loader()
test_loader = loader.test_image_loader()
validation_loader = loader.validation_image_loader()

model = AutoEncoder(
    input_size=256*256*3, 
    hidden_size_1=1024, 
    hidden_size_2=512, 
    latent_size=500,
    model_path="model.h5",
    history_path="history.npy"
)

# images = next(train_loader)

# plt.plot(images[0])
# plt.show()

# model.plot_history()

# model.test_model(test_loader)

model.train(train_loader, validation_loader, epochs=50, batch_size=32)
