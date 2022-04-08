""" This module allows user to work with images. Mainly used to
read images, masks and apply filters. Also displays the images,
masks and predictions.

Author:
Anirudh Kakarlapudi
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Image:
    """ Reads images, applies filters and displays the images
    """
    def __init__(self, img_dir="cats_dogs/data/images/",
                 msk_dir="cats_dogs/data/annotation/trimaps/",
                 fil_dir="cats_dogs/data/predictions/"):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.fil_dir = fil_dir

    def read_mask(self, mask, output_shape=None, normalize=True):
        """ Reads the mask, subtracts -1 from the masks
        since the ground truth labels are 1,2,3

        Args:
            mask(arr):
                An mask represented as an array or matrix
            output_shape(list):
                The required output shape of mask. Resizes mask to the
                output shape if given
            normalize(bool):
                If truem subtracts -1 from the masks since the ground
                truth labels are 1,2,3
        """
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        if output_shape is not None:
            mask = cv2.resize(mask, output_shape)
        if normalize:
            mask = mask - 1
        mask = mask.astype(np.int32)
        return mask

    def read_image(self, img, output_shape=None,
                   normalize=True, orig_color=False):
        """ Reads the image and normalizes it by dividing it with 255

        Args:
            img(arr):
                An image represented as an array or matrix
            output_shape(list):
                The required output shape of image. Resizes image to the
                output shape if given
            normalize(bool):
                If True, divdes the color images by 255 to make the range
                of values in image array to be (0,1)
            orig_color(bool):
                open_cv considers images to be in BGR format. So convert into
                original color formal RGB if given
        """
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        if orig_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if output_shape is not None:
            img = cv2.resize(img, output_shape)
        if normalize:
            img = img/255
        img = img.astype(np.float32)
        return img

    def display_img(self, img):
        """ Displays the given image
        Args:
            img(arr):
                An image represented as an array or matrix
        """
        if type(img) == str:
            img = self.read_image(img)
        plt.imshow(img)
        plt.show()

    def display_img_mask(self, img, mask, pred=None,
                         save_fig=False, true_size=True):
        """ Displays the image, mask and predicted mask

        Args:
            img(str/arr):
                If the img is path(str) then image is read from path.
                If the img is arr, then the image is directly displayed
            mask(str/arr):
                If the mask is path(str) then mask is read from path.
                If the mask is arr, then the mask is directly displayed
            pred(str/arr):
                If the pred is path(str) then prediction is read from path.
                If the pred is arr, then the prediction is directly displayed
            save_fig(bool):
                Saves the figure with the name
            true_size(bool):
                Change the predictions to original size
        """
        num = 2 if pred is None else 3
        _, axes = plt.subplots(1, num)
        if type(img) == str:
            img = self.read_image(img)
        if type(mask) == str:
            img = self.read_image(mask)

        axes[0].imshow(img)
        axes[1].imshow(mask)

        if pred is not None:
            shape = img.shape[:-1][::-1] if true_size else None
            if type(pred) == str:
                pred = self.read_image(pred, output_shape=shape)
            axes[2].imshow(pred)

        if save_fig:
            plt.savefig("Display.jpg")

        plt.show()

    def display_n_img_masks(self, imgs_dir, mask_dir, n_imgs=2,
                            pred_dir=None, save_fig=True, true_size=True):
        """ Displays the number of images, masks and predicted masks

        Args:
            imgs_dir(str):
                The path of images directory
            mask_dir(str):
                The path of masks directory
            n_imgs(int):
                The number of images to be displayed
            pred_dir(str):
                The path of predictions directory. If given, the predictions
                are also displayed
            save_fig(bool):
                Saves the figure with the name
            true_size(bool):
                Change the predictions to original size
        """

        num = 2 if pred_dir is None else 3
        directory = mask_dir if pred_dir is None else pred_dir
        names = [x for x in os.listdir(directory) if x.endswith(".png")]
        names = np.random.choice(names, n_imgs)
        names = [x.split("/")[-1][:-4] for x in names]

        _, axis = plt.subplots(n_imgs, num)

        for i, elem in enumerate(names):

            image = self.read_image(imgs_dir + elem + ".jpg")
            mask = self.read_mask(mask_dir + elem + ".png")

            axis[i][0].imshow(image)
            axis[i][1].imshow(mask)

            axis[i][0].set_title(label="Image",
                                 fontdict={"fontsize": 10})
            axis[i][1].set_title(label="Mask",
                                 fontdict={"fontsize": 10})
            if pred_dir is not None:
                shape = image.shape[:-1][::-1] if true_size else None
                pred = self.read_mask(mask=(pred_dir + elem + ".jpg"),
                                      output_shape=shape, normalize=True)

                axis[i][2].imshow(pred)
                axis[i][2].set_title(label="Prediction",
                                     fontdict={"fontsize": 10})

            plt.setp(axis[i, 0], ylabel=elem)
        # Set fontsize the ticks from the plots
        for j in range(n_imgs):
            for k in range(num):
                for label in (axis[j][k].get_xticklabels() +
                              axis[j][k].get_yticklabels()):
                    label.set_fontsize(7)
        if save_fig:
            plt.savefig("Prediction.jpg")
        plt.show()

    def apply_gaussian_filter(self, image, kernel_size=3, display=False):
        """ Blurs the image by applying a Gaussian filter on the image.
        A gaussian filter is a low pass filter. Hence it reduces high frequency
        components and smooths the inage

        Args:
            image(arr):
                An image represented as an array or matrix
            kernel_size(int):
                Width and height of the filter mask
            display(bool):
                Displays the image after applying the filter if true

        Returns:
            filter_img(arr):
                Image after the application of bilateral filter
        """
        filter_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        filter_img = cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB)

        if display:
            self.display_img(filter_img)
        return filter_img

    def apply_laplacian_filter(self, image, kernel_size=3,
                               depth=5, display=False):
        """ A linear differential operator to find edges in an image. Computes
        the second derivatives of an image, measuring the rate at which the
        first derivatives change.

        Args:
            image(arr):
                An image represented as an array or matrix
            depth(int):
                Represents the depth of the output image
            kernel_size(int):
                Width and height of the filter mask
            display(bool):
                Displays the image after applying the filter if true

        Returns:
            filter_img(arr):
                Image after the application of laplacian filter
        """
        filter_img = cv2.Laplacian(image, depth, kernel_size)

        if display:
            self.display_img(filter_img)
        return filter_img

    def apply_bilateral_filter(self, image, dia=1, sigma_color=5,
                               sigma_space=5, display=False):
        """ Applies the bilateral filter to the image. Median filter is used to
        remove noise from an image. To achieve this, the bilateral filter
        replaces the each pixel with weighted mean of nearby pixels

        Args:
            image(arr):
                An image represented as an array or matrix
            dia(int):
                Diameter of each pixel neighborhood
            sigmaColor(int):
                sigma denotes spatial extent of kernel.The smaller the value
                of sigma, the sharper the edge. sigmaColor is the value of
                sigma in color space. The greater the value of sigmaColor
                the colors farther to each other will start to get mixed
            sigmaSpace(int):
                sigmaSpace is the value of sigma in coordinate space. The
                greater the value of sigmaSpace the more further pixels
                will mix
            display(bool):
                Displays the image after applying the filter if true

        Returns:
            filter_img(arr):
                Image after the application of bilateral filter
        """
        filter_img = cv2.bilateralFilter(image, dia, sigma_color, sigma_space)
        filter_img = cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB)

        if display:
            self.display_img(filter_img)
        return filter_img

    def apply_median_filter(self, image, kernel_size=3, display=False):
        """ Applies the median blur to the image. Replaces each pixel with
        median of nearby pixels

        Args:
            image(arr):
                An image represented as an array or matrix
            kernel_size(int):
                Width and height of the filter mask
            display(bool):
                Displays the image after applying the filter if true

        Returns:
            filter_img(arr):
                Image after the application of median filter
        """
        # filter_img = cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB)
        filter_img = cv2.medianBlur(image, kernel_size)
        filter_img = cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB)
        if display:
            self.display_img(filter_img)
        return filter_img

    def apply_box_filter(self, image, depth, kernel_size, display=False):
        """ Replaces each pixel with average of nearby pixels. Hence smoothens
        the image by reducing noise and removing the edges.

        Args:
            image(arr):
                An image represented as an array or matrix
            depth(int):
                Represents the depth of the output image
            kernel_size(int):
                Width and height of the filter mask
            display(bool):
                Displays the image after applying the filter if true

        Returns:
            filter_img(arr):
                Image after the application of box filter
        """
        filter_img = cv2.boxFilter(image, ddepth=depth, ksize=kernel_size)
        filter_img = cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB)
        if display:
            self.display_img(filter_img)
        return filter_img

    def clean_directory(self, image_dir, mask_dir,
                        imgs_save_dir="cats_dogs/data/imgs_paths.txt",
                        mask_save_dir="cats_dogs/data/mask_paths.txt",
                        pre_cleaned=False):
        """ Reads images that end with '.jpg' or masks that end with '.png'
        in a directory and checks the image for errors

        Args:
            image_dir(str):
                The path of directory of paths or images
            mask_dir(bool):
                The path of directory of paths or masks

        Returns:
            (list):
                The list of correct image_paths and corresponding mask paths
        """
        correct_image_paths, correct_mask_paths = [], []
        if (pre_cleaned and os.path.exists(imgs_save_dir) and 
            os.path.exists(mask_save_dir)):
            with open(imgs_save_dir, "r") as f:
                correct_image_paths = f.read().splitlines()
            with open(mask_save_dir, "r") as f:
                correct_mask_paths = f.read().splitlines()
        else:   
            # check 1: if mask does not exist
            # check 2: both mask and images are in correct format
            # print(images, masks)

            for img in os.listdir(image_dir):
                not_corrupt = False
                if os.path.exists(mask_dir + img[:-4] + ".png"):
                    if ((cv2.imread(mask_dir + img[:-4] + ".png") is not None)
                        and
                        (cv2.imread(image_dir + img) is not None)):
                        not_corrupt = True
                if not_corrupt:
                    correct_image_paths.append(image_dir + img)
                    correct_mask_paths.append(mask_dir + img[:-4] + ".png")
        
        # save the correct paths in file to save computation time
        with open(imgs_save_dir, "w") as file:
            file.write("\n".join(correct_image_paths))
        with open(mask_save_dir, "w") as file:
            file.write("\n".join(correct_mask_paths))

        return correct_image_paths, correct_mask_paths
