from gc import callbacks
import os
import cv2
from cats_dogs.data.data_loader import DataGenerator
from sklearn.model_selection import train_test_split
from cats_dogs.models.unet import construct_model
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import optimizers

image_dir = "cats_dogs/data/images/"
mask_dir = "cats_dogs/data/annotations/trimaps/"
dl = DataGenerator()
image_paths, mask_paths = dl.clean_directory(image_dir, mask_dir,
                                             pre_cleaned=True)
(train_images, test_images,
 train_masks, test_masks) = train_test_split(image_paths, mask_paths,
                                             test_size=0.2, random_state=100, 
                                             shuffle=True)
(train_images, cv_images,
 train_masks, cv_masks) = train_test_split(train_images, train_masks,
                                           test_size=0.2, random_state=100, 
                                           shuffle=True)

print("Train images: ", len(train_images))
print("Validation images: ", len(cv_images))
print("Test images: ", len(test_images))

# Parameters
image_shape = (256, 256, 3)
n_filters = 8
batch_size = 16
n_classes = 3
epochs = 15
loss = "sparse_categorical_crossentropy"
learning_rate = 1e-03
optimizer = optimizers.RMSprop(learning_rate)
callback = [callbacks.ModelCheckpoint("oxford_segmentation.h5",
            save_best_only=True)]

model = construct_model(image_shape, n_classes, n_filters)
train_gen = DataGenerator(batch_size, image_shape, train_images, train_masks)
valid_gen = DataGenerator(batch_size, image_shape, cv_images, cv_masks)
model.compile(optimizer=optimizer, loss=loss,
              metrics=["accuracy"])

model.fit(train_gen, epochs=epochs, validation_data=valid_gen,
          callbacks=callback)

