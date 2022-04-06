import argparse
from cats_dogs.data.data_loader import SegmentationDataLoader
from cats_dogs.models.unet import Unet
from cats_dogs.utils import metrics
import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        ReduceLROnPlateau,
                                        EarlyStopping)
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt


def segmentation_train(loss="categorical_crossentropy", metric="accuracy"):
    """ Trains the unet model for image segmentation and saves the best model

    Args:
        loss(function/str):
            A loss function or predefined string like categorical_crossentropy
        metric(function/str)
            A metric function or predefined string like accuracy
    """
    # Hyper parameters
    shape = (128, 128, 3)
    num_classes = 3
    learning_rate = 1e-03
    b_size = 8
    epochs = 50
    dropout = 0.3
    kernel_size = 3
    num_filters = 16

    sl = SegmentationDataLoader(width=shape[0], height=shape[1])
    un = Unet()
    data = sl.train_test_cv_files()
    img_train_names, mask_train_names = data[0]
    img_cv_names, mask_cv_names = data[2]

    # Model
    model = un.construct(shape, num_classes, num_filters=num_filters, dropout=dropout, kernel_size=kernel_size)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=metric)
    train_data = sl.segmentation_data(img_train_names,
                                      mask_train_names, batch_size=b_size)
    cv_data = sl.segmentation_data(img_cv_names,
                                   mask_cv_names, batch_size=b_size)

    train_steps = len(train_data)//b_size
    cv_steps = len(cv_data)//b_size

    callbacks = [
        ModelCheckpoint("model_seg.h5", verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3,
                          factor=0.1, verbose=1, min_lr=1e-5),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1)]
    final = model.fit(train_data, steps_per_epoch=train_steps,
                      validation_data=cv_data,
                      validation_steps=cv_steps,
                      epochs=epochs,
                      callbacks=callbacks)
    display_model_epochs(final.history["loss"],
                         final.history["val_loss"],
                         "Loss")
    display_model_epochs(final.history["accuracy"],
                         final.history["val_accuracy"],
                         "Accuracy")

def display_model_epochs(train_val, cv_val, name):
    plt.figure(figsize=(7,7))
    plt.plot(train_val, label='Training')
    plt.plot(cv_val, label='Validation')
    plt.title(f"Model {name} vs Epochs")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(f"{name}")
    plt.savefig(f"Epochs_vs_{name}.png")

def segmentation_test(image=None):
    """ Loads the unet model for image segmentation and predicts the mask
    for given image. If image is not given, the model predicts for all the
    test images and displays the output

    Args:
        image(arr):
            An image to be segmented
    """
    shape = (128, 128, 3)
    sl = SegmentationDataLoader(width=shape[0], height=shape[1])
    if not os.path.isfile("model_seg.h5"):
        print("There are no saved models.")
    else:
        # Load the model
        model = tf.keras.models.load_model("model_seg.h5")
        if image is None:
            # test against all the images
            data = sl.train_test_cv_files()
            img_test_names, mask_test_names = data[1]
            for i in range(len(img_test_names))[:10]:
                test_image = img_test_names[i]
                test_mask = mask_test_names[i]
                name = test_image.split("/")[-1]

                # Read image
                image = sl.read_image(test_image)
                true_mask = sl.read_mask(test_mask)
                num_classes = 3

                # prediction
                pred_mask = model.predict(np.expand_dims(image, axis=0))[0]
                pred_mask = np.argmax(pred_mask, axis=-1)
                pred_mask = np.expand_dims(pred_mask, axis=-1)
                pred_mask = pred_mask * (255/num_classes)
                pred_mask = pred_mask.astype(np.int32)
                pred_mask = np.concatenate([pred_mask, pred_mask, pred_mask],
                                           axis=2)
                res_path = "cats_dogs/data/predictions/"
                if not os.path.exists(res_path):
                    os.makedirs(res_path)
                cv2.imwrite(res_path+name, pred_mask)
            display_segmentation_predictions(num=3)
        else:
            img = cv2.resize(image, (shape[0], shape[1]))
            # Normalize the image
            img = img/255.0
            img = img.astype(np.float32)
            pred = model.predict(img)
            pred_mask = model.predict(np.expand_dims(image, axis=0))[0]
            pred_mask = np.argmax(pred_mask, axis=-1)
            pred_mask = np.expand_dims(pred_mask, axis=-1)
            pred_mask = pred_mask * (255/num_classes)
            pred_mask = pred_mask.astype(np.int32)
            pred_mask = np.concatenate([pred_mask, pred_mask, pred_mask],
                                        axis=2)
            display_segmentation_predictions(image, pred_mask)

def display_segmentation_predictions(image=None, mask=None, num=5):
    if image is None:
        _,ax = plt.subplots(3,num)
        res_path = "cats_dogs/data/predictions/"
        image_path = "cats_dogs/data/images/"
        mask_path = "cats_dogs/data/annotations/trimaps/"
        names = os.listdir(res_path)
        display_names = np.random.choice(names, num)
        for i in range(num):
            name = display_names[i]
            print(name)
            pred = cv2.imread(res_path+name, cv2.IMREAD_GRAYSCALE)
            shape = pred.shape
            image = cv2.imread(image_path+name, cv2.IMREAD_COLOR).astype(np.float32)
            image = image/255
            image = cv2.resize(image, shape)
            
            mask = cv2.imread(mask_path+name[:-4]+".png", cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, shape)

            ax[i][0].imshow(image)
            ax[i][1].imshow(mask)
            ax[i][2].imshow(pred)
        plt.show()
        plt.savefig("Prediction.jpg")
    else:
        _,ax = plt.subplots(1,2)
        ax[0].imshow(image)
        ax[1].imshow(mask)
        plt.savefig("Prediction.jpg")


def classification_train(loss, metric):
    pass


def classification_test():
    pass


if __name__ == "__main__":
    np.random.seed(40)
    tf.random.set_seed(40)
    parser = argparse.ArgumentParser()
    # Add the arguments to get the ground truth and prediction filenames
    parser.add_argument('-t', '--task', type=int, required=False, default=0,
                        help="Segmentation task -> 0,\
                              classification task -> 1")
    parser.add_argument('-m', '--metric', type=int, required=False, default=0,
                        help="{0 -> categorical_cross_entropy,\
                              1-> dice loss}")
    parser.add_argument('-f', '--function', type=str, required=True,
                        help="Train or test the model -> {train, test}")

    # Parse the arguments
    args = parser.parse_args()
    if args.metric == 0:
        loss = "categorical_crossentropy"
        metric = ["accuracy"]
    else:
        loss = metrics.iou_coef
        metric = ["accuracy"]
    if args.function == "train":
        if args.task == 0:
            segmentation_train(loss, metric)
        else:
            # classification_train(loss, metric)
            pass
    elif args.function == "test":
        if args.task == 0:
            segmentation_test()
        else:
            pass
            # classification_test()
    else:
        print("Wrong arguments chosen")
