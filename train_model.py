from loader import DatasetLoader
from model import AutoEncoder

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

model = AutoEncoder(
    input_size=256*256, 
    hidden_size_1=128, 
    hidden_size_2=64, 
    latent_size=32
)

for images in train_loader:
    # print(images.shape)
    # break
    model.train(images, epochs=100, batch_size=32)
    break