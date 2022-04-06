import os
import cv2
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class DataLoader:
    """ Loads the dataset from the directory and converts into the required
    shape and format

    Args:
        width(int):
            Global width of all the images
        height(int):
            Global height of all the images
        image_path(str):
            The directory path of images
        mask_path(str):
            The directory path of masks
    """
    def __init__(self, width=256, height=256,
                 image_path="cats_dogs/data/images/",
                 mask_path="cats_dogs/data/annotations/trimaps/"):
        self.height = height
        self.width = width
        self.image_path = image_path
        self.mask_path = mask_path

    def read_image(self, image_path):
        """ Reads the image from the image_path and returns it
        in required format

        Args:
            image_path(str):
                The directory path of an image
        Returns:
            image(arr):
                The normalised array of the image with 3 dimensions
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Resize the image
        img = cv2.resize(img, (self.width, self.height))
        # Normalize the image
        img = img/255.0
        img = img.astype(np.float32)
        return img

    def train_test_cv_files(self, train_path="cats_dogs/data/annotations/trainval.txt",
                            test_path="cats_dogs/data/annotations/test.txt", cval=True):
        """ Reads the train, test file names and returns train, test and cv
        file names

        Args:
            train_path(str):
                The directory path of the train files without images
            test_path(str):
                The directory path of the test files without images
            cval(bool):
                Creates the cross validation files if true
        Returns:
            (list):
                list of tuples train, cv(optional), test filenames
        """
        pass

    def display_image_mask(self, img=None, mask=None,
                           fname="Abyssinian_1"):
        """ A display of image and its correspoinding mask

        Args:
            img(arr):
                An array of image in pixels to be displayed
            mask(arr):
                An array of mask in pixels to be displayed
            fname(str):
                If the image or mask are not given, then use fname to display a
                mask and image
        """
        if img is None or mask is None:
            img = self.read_image(self.image_path+fname+".jpg")
            mask = self.read_mask(self.mask_path+fname+".png")
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img)
        axarr[1].imshow(mask)
        plt.show()


class SegmentationDataLoader(DataLoader):
    """ Loads the dataset from the directory and converts into the required
    shape and format
    """
    def read_mask(self, mask_path):
        """ Reads the mask from the mask_path and returns it
        in required format

        Args:
            mask_path(str):
                The directory path of an mask
        Returns:
            mask(arr):
                The normalised array of the mask
        """
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Resize the image
        msk = cv2.resize(msk, (self.width, self.height))

        msk -= 1
        msk = msk.astype(np.int32)
        return msk

    def train_test_cv_files(self, train_path="cats_dogs/data/annotations/trainval.txt",
                            test_path="cats_dogs/data/annotations/test.txt", cval=True):
        """ Reads the train, test file names and returns train, test and cv
        file names

        Args:
            train_path(str):
                The directory path of the train files without images
            test_path(str):
                The directory path of the test files without images
            cval(bool):
                Creates the cross validation files if true
        Returns:
            (list):
                list of tuples train, cv(optional), test filenames
        """
        def load_files(path):
            """ Helper fiunction to read the image names and mask names
            from path

            Args:
                path(str):
                    Directory path of file
            Returns:
                (list):
                    List of images and masks
            """
            df = pd.read_csv(path, sep=" ", header=None)
            files_names = df[0].values
            images = [self.image_path+fname+".jpg" for fname in files_names]
            masks = [self.mask_path+fname+".png" for fname in files_names]
            return images, masks
        train_imgs, train_msks = load_files(train_path)
        test_imgs, test_msks = load_files(test_path)
        if cval:
            train_imgs, cv_imgs, train_msks, cv_msks = train_test_split(
                                                            train_imgs,
                                                            train_msks,
                                                            test_size=0.2,
                                                            random_state=25)
            return [(train_imgs, train_msks), (test_imgs, test_msks),
                    (cv_imgs, cv_msks)]
        return [(train_imgs, train_msks), (test_imgs, test_msks)]

    def segmentation_data(self, img_names, msk_names,
                          batch_size=8, num_map_threads=1, n_repeat=10, shuffle=True):
        """ Creates a tensor dataset based on the batchsize

        Args:
            img_name(list):
                List of image file names
            msk_names(list):
                List of mask file names
            batch_size(int):
                The batch size of each dataset
            num_map_threads(int):
                The number of map threads for the map function to be
                called in parallel
        Returns:
            data(tensor)
        """
        data = tf.data.Dataset.from_tensor_slices((img_names, msk_names))
        if shuffle:
            data = data.shuffle(buffer_size=5000)
        data = data.map(self.read_wrapper,
                        num_parallel_calls=num_map_threads)
        data = data.batch(batch_size)
        data = data.repeat(n_repeat)
        data = data.prefetch(batch_size)
        return data

    def read_wrapper(self, image_fname, mask_fname):
        """ A wrapper function to read the image and mask

        Args:
            image_fname(str):
                An image file name
            mask_fname(str):
                A mask file name
        Retuns:
            (list)
                An image and mask in required format
        """
        def helper(img, msk):
            img = img.decode()
            msk = msk.decode()

            image = self.read_image(img)
            mask = self.read_mask(msk)

            return image, mask

        image, mask = tf.numpy_function(helper,
                                        [image_fname, mask_fname],
                                        [tf.float32, tf.int32])
        mask = tf.one_hot(mask, 3, dtype=tf.int32)
        image.set_shape([self.width, self.height, 3])
        mask.set_shape([self.width, self.height, 3])

        return image, mask


class ClassificationDataLoader(DataLoader):
    def train_test_cv_files(self, train_path="cats_dogs/data/annotations/trainval.txt",
                            test_path="cats_dogs/data/annotations/test.txt", cval=True):
        """ Reads the train, test file names and returns train, test and cv
        file names

        Args:
            train_path(str):
                The directory path of the train files without images
            test_path(str):
                The directory path of the test files without images
            cval(bool):
                Creates the cross validation files if true

        Returns:
            (list):
                list of tuples train, cv(optional), test filenames
        """
        def load_files(path, train=True):
            """ Reads the file names from the given directory and
            returns a dataframe
            """
            column_names = ["Image", "ClassId", "Species", "BreedId"]
            df = pd.read_csv(path, sep=" ", names=column_names)
            df["Species"] -= 1
            if train:
                # Create and save the class_map
                self.class_map = self.create_breed_map(df.Image,
                                                       df.BreedId,
                                                       df.Species,
                                                       df.ClassId)
                with open("class_map.pkl", "wb") as f:
                    pickle.dump(self.class_map, f)

            images = [self.read_image(self.image_path + x + ".jpg")
                      for x in df.Image]
            df["Images"] = images
            df.Species = df.Species.astype(np.int32)
            df.ClassId = df.ClassId.astype(np.int32)
            df = df.drop(columns=["Image", "BreedId"])
            return df

        train_df = load_files(train_path)
        train_df, cv_df = train_test_split(train_df, test_size=0.2)
        test_df = load_files(test_path, False)
        return train_df, test_df, cv_df

    def classification_data(self, train_path="cats_dogs/data/annotations/trainval.txt",
                            test_path="cats_dogs/data/annotations/test.txt",
                            cval=True, classid=False):
        """ Reads the data from the saved files if present or creates the data
        from the directory

        Args:
            train_path(str):
                The directory path of the train files without images
            test_path(str):
                The directory path of the test files without images
            cval(bool):
                Creates the cross validation files if true

        Returns:
            (list):
                list of tuples train, cv(optional), test filenames
        """
        if not os.path.exists("train.pkl"):
            train_df, test_df, cv_df = self.train_test_cv_files(train_path,
                                                                test_path,
                                                                cval)
            train_df.to_pickle("train.pkl")
            test_df.to_pickle("test.pkl")
            cv_df.to_pickle("cv.pkl")
        else:
            train_df = pd.read_pickle("train.pkl")
            test_df = pd.read_pickle("test.pkl")
            cv_df = pd.read_pickle("cv.pkl")
            with open("class_map.pkl", "rb") as f:
                self.class_map = pickle.load(f)
        return train_df, test_df, cv_df

    def create_breed_map(self, image_name_col,
                         breed_col, species_col,
                         class_id):
        """ Creates a breed map for the respective cat and dog breed names
        with their ids.

        Args:
            image_name_col(arr):
                Breed names of the pets
            breed_col(arr):
                The id's of respective breeds
            species_col(arr):
                The id's of cat or dog species (0:cat, 1:dog)
            class_id(arr):
                The id's of respective classses

        Returns:
            class_dict(dict):
                A class dictionary with each id mapped to corresponding
                dog or cat breeds
        """
        cat_breed, dog_breed = dict(), dict()
        class_dict = dict()
        for i, name in enumerate(image_name_col):
            name = "".join(re.findall(r"[A-Za-z_]+", name)).rstrip("_").lower()
            class_dict[str(class_id[i])] = (species_col[i], breed_col[i], name)
        return class_dict


# dl = SegmentationDataLoader()
# (X_t, y_t), (X_c, y_c), (X_e, y_e) = dl.train_test_cv_files()
# data = dl.segmentation_data(X_t, y_t)
# print(len(data))
# for x, y in data:
#     print(x.shape, y.shape) ## (8, 256, 256, 3), (8, 256, 256, 3)
#     print(x)
#     ans = input()

# cdl = ClassificationDataLoader()
# train_df, test_df,cv_df = cdl.classification_data()
# data = cdl.classification_data(train_df)
