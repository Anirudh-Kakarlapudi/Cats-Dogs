""" This module allows user to generate the batch segmentation data
when training the model. Since this module requires the Image class
import, this cannot be run without setting up the cats_dogs package.
It can be run from the main.py file

Author:
Anirudh Kakarlapudi
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from cats_dogs.utils.image import Image


class DataGenerator(keras.utils.Sequence, Image):
    """ A data generator that inherits Sequence class and
    Image class. It is used to generate training batch and
    validation batch.

    Attributes:
        batch_size
    """
    def __init__(self, batch_size=16, img_shape=(128, 128, 3),
                 img_paths=None, mask_paths=None):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.img_paths = img_paths
        self.mask_paths = mask_paths
    
    def __len__(self):
        """ Denotes number of batches per epoch"""
        return int(np.floor(len(self.img_paths)/self.batch_size))

    def data_gen(self, images, masks):
        """ Loads the images and masks and creates a batch

        Args:
            images(list):
                list of paths of images
            masks(list):
                list of paths of masks
        Returns:
            (list):
                A batch of images and masks
        """
        # The Input tensor should be of dim (batch_size, width, height, 3)
        batch_images = np.zeros((self.batch_size, *self.img_shape),
                                dtype=np.float32)

        # The Output tensor should be of dim (batch_size, width, height, 3)
        batch_masks = np.zeros((self.batch_size, *self.img_shape),
                               dtype=np.int32)
        for i, _ in enumerate(images):
            batch_images[i] = self.read_image(images[i],
                                              output_shape=self.img_shape[:-1])
            msk = self.read_mask(masks[i], output_shape=self.img_shape[:-1])
            # increase the dimesions of msk
            batch_masks[i] = msk  # np.expand_dims(msk, 2)
        return batch_images, batch_masks

    def apply_filter(self, filter_name="gaussian",
                     filter_path="cats_dogs/data/filter_images/",
                     kernel_size=3,
                     depth=5):
        """ Applies filter to the images and creates a new set of images

        Args:
            filter_name(str):
                Name of the filter among {box, gaussian, laplacian, bilateral}
            filter_path(str):
                The path for the filtered image
        """
        if not os.path.exists(filter_path):
            os.makedirs(filter_path)
        for image in tqdm(self.img_paths):
            name = image.split("/")[-1]
            if filter_name == "box":
                f_img = self.apply_box_filter(self.read_image(image),
                                              depth, kernel_size)
            elif filter_name == "gaussian":
                f_img = self.apply_gaussian_filter(self.read_image(image),
                                                   kernel_size)
            elif filter_name == "laplacian":
                f_img = self.apply_laplacian_filter(self.read_image(image),
                                                    kernel_size, depth)
            elif filter_name == "bilateral":
                f_img = self.apply_bilateral_filter(self.read_image(image))
            elif filter_name == "median":
                f_img = self.apply_median_filter(self.read_image(image),
                                                 kernel_size)
            else:
                continue
            cv2.imwrite(filter_path + name, f_img)

    def __getitem__(self, batch_num):
        """ Returns the list to inputs, targets corresponding to batch number

        Args:
            batch_num(int):
                The batch position in the sequence
        Returns:
            (list):
                A batch of images and masks
        """
        i = batch_num * self.batch_size
        batch_images = self.img_paths[i:i+self.batch_size]
        batch_masks = self.mask_paths[i:i+self.batch_size]

        return self.data_gen(batch_images, batch_masks)
