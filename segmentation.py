""" Script that accepts the arguments from the user and runs the appropriate
methods.

Author:
Anirudh Kakarlapudi
"""

import os
import cv2
import argparse
from tqdm import trange
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from cats_dogs.utils.image import Image
from sklearn.model_selection import train_test_split
from cats_dogs.models.unet import construct_model
from cats_dogs.data.data_loader import DataGenerator

from cats_dogs.utils.metrics import jaccard_coefficient, f1_loss, f1_score


IMG_DIR = "cats_dogs/data/images/"
MSK_DIR = "cats_dogs/data/annotations/trimaps/"
PRED_DIR = "cats_dogs/data/predictions/"

# Parameters
IMG_SHAPE = (256, 256, 3)
N_FILTERS = 16
BATCH_SIZE = 16
N_CLASSES = 3
EPOCHS = 40
LEARNING_RATE = 1e-03


def run_train(**kwargs):
    data_gen = DataGenerator()
    image_paths, mask_paths = data_gen.clean_directory(IMG_DIR, MSK_DIR,
                                                       pre_cleaned=True)
    (train_images, test_images,
     train_masks, test_masks) = train_test_split(image_paths, mask_paths,
                                                 test_size=0.2,
                                                 random_state=100, 
                                                 shuffle=True)
    (train_images, cv_images,
     train_masks, cv_masks) = train_test_split(train_images, train_masks,
                                               test_size=0.2, random_state=100,
                                               shuffle=True)

    print("Train images: ", len(train_images))
    print("Validation images: ", len(cv_images))
    print("Test images: ", len(test_images))

    params = {"image_shape": IMG_SHAPE, "n_filters": N_FILTERS,
              "batch_size": BATCH_SIZE, "n_classes": N_CLASSES,
              "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
              "optimizer": "adam", "loss": "crossentropy"}

    # Update the parameters
    for param, item in enumerate(kwargs.items()):
        if param in params:
            params[param] = item

    if params["optimizer"] == "adam":
        optimizer = optimizers.Adam(params["learning_rate"])
    elif params["optimizer"] == "rmsprop":
        optimizer = optimizers.RMSProp(params["learning_rate"])
    if params["loss"] == "crossentropy":
        loss = "categorical_crossentropy"
    elif params["loss"] == "f1":
        loss = f1_loss

    callback = [callbacks.ModelCheckpoint("model.h5", verbose=1,
                                          save_best_model=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3,
                                            factor=0.1, min_lr=1e-06,
                                            verbose=1),
                callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                        verbose=1)]

    model = construct_model(params["image_shape"], params["n_classes"],
                            params["n_filters"])
    train_gen = DataGenerator(params["batch_size"], params["image_shape"],
                              train_images, train_masks)
    valid_gen = DataGenerator(params["batch_size"], params["image_shape"],
                              cv_images, cv_masks)
    # test_gen = DataGenerator(params["batch_size"], params["img_shape"],
    #                          test_images, test_masks)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=["accuracy", f1_score, jaccard_coefficient])

    model.fit(train_gen, epochs=params["epochs"], validation_data=valid_gen,
              callbacks=callback)
    return model


def run_test(img_path=None, **kwargs):
    img_cl = Image()
    params = {"image_shape": IMG_SHAPE, "n_filters": N_FILTERS,
              "batch_size": BATCH_SIZE, "n_classes": N_CLASSES,
              "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
              "optimizer": "adam", "loss": "crossentropy"}

    # Update the parameters
    print(kwargs.items())
    for param, item in enumerate(kwargs.items()):
        if param in params:
            print(param)
            params[param] = item
    if not os.path.exists("model.h5"):
        # train the model
        model = run_train(**kwargs)
    else:
        dependencies = {'f1_score': f1_score,
                        'f1_loss': f1_loss,
                        'jaccard_coefficient': jaccard_coefficient}
        model = tf.keras.models.load_model("model.h5",
                                           custom_objects=dependencies)
    if img_path is None:
        data_gen = DataGenerator()
        image_paths, mask_paths = data_gen.clean_directory(IMG_DIR, MSK_DIR,
                                                           pre_cleaned=True)

        (_, test_images,
         _, test_masks) = train_test_split(image_paths, mask_paths,
                                           test_size=0.2,
                                           random_state=100, 
                                           shuffle=True)

        for i in trange(len(test_images[:])):
            name = test_images[i].split("/")[-1][:-4]
            test_image = img_cl.read_image(IMG_DIR+name+".jpg",
                                           params["image_shape"][:-1],
                                           normalize=True)
            test_image = np.expand_dims(test_image, [0])
            test_msk_true = img_cl.read_mask(MSK_DIR+name+".png")
            if test_msk_true is None:
                continue
            true_shape = test_msk_true.shape

            pred = model.predict(test_image)
            reconstructed_mask = img_cl.reconstruct_mask(pred,
                                                         true_shape[:-1][::-1])
            if not os.path.exists(PRED_DIR):
                os.makedirs(PRED_DIR)
            cv2.imwrite(PRED_DIR+name+".png", reconstructed_mask)
        img_cl.display_n_img_masks(IMG_DIR, MSK_DIR, 3, PRED_DIR)
    else:
        image = img_cl.read_image(img_path, params["image_shape"], True)
        pred = model.predict(image)
        reconstructed_pred = img_cl.reconstruct_mask(pred,
                                                     true_shape[:-1][::-1])
        img_cl.display_img_mask(image, reconstructed_pred)

def run():
    parser = argparse.ArgumentParser(description="U-net Image Segmentation")
    
    parser.add_argument('-f','--func', type=int, required=False, default=1,
                        help="Train(0) or test(1) the model")
    parser.add_argument("-p", "--path", type=str, required=False,
                        default="",
                        help="The path of the test or query image")
    parser.add_argument('-m','--model', type=str, required=False, default=0,
                        help="Mondel to be used {0 : Unet}")
    parser.add_argument('-is','--image_shape', type=str, required=False,
                        default="(256,256,3)",
                        help="The input shape of the image")
    parser.add_argument("-o", "--optimizer", type=str, required=False,
                        default="adam",
                        help="Optimizer to choose-{adam, rmsprop}")
    parser.add_argument("-e", "--epochs", type=int, required=False,
                        default="40",
                        help="Number of epochs to train the model")
    parser.add_argument("-l", "--loss", type=str, required=False,
                        default="crossentropy",
                        help="The loss function to be chosen {crossentropy," +
                             "f1_loss}")
    parser.add_argument("-b", "--batch_size", type=int, required=False,
                        default=16, help = "Size of the batch")
    args = parser.parse_args()
    #args = vars(args)
    if args.func == 0:
        run_train()
    else:
        run_test()

if __name__ == "__main__":
    run()