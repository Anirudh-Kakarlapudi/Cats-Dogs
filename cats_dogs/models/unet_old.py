import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose,
                                     BatchNormalization,
                                     Activation, MaxPool2D, Concatenate,
                                     Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


class Unet:
    def conv_block(self, inp, num_filters, kernel_size=3):
        """ Implements a Convolution layer followed by Relu activation twice

        Args:
            inp(arr):
                Input image or previous layer output
            num_filters(int):
                Number of filters for the convolution operation
            kernel_size(int):
                Size of filter

        Returns:
            X(arr):
                The output of the convolution block
        """
        X = inp
        for i in range(2):
            X = Conv2D(filters=num_filters,
                    kernel_size=kernel_size,
                    padding="same", activation="relu")(X)
        return X

    def downsample_block(self, inp, num_filters, kernel_size=3, dropout=0.1):
        """ Implements a single downsample block

        Args:
            inp(arr):
                Input image or previous layer output
            num_filters(int):
                Number of filters for the convolution operation
            kernel_size(int):
                Size of filter

        Returns:
            (list):
                Outputs of convolution and pooling layers
        """
        c = self.conv_block(inp, num_filters, kernel_size)
        p = MaxPool2D((2, 2), strides=2)(c)
        p = Dropout(dropout)(p)
        return c, p

    def upsample_block(self, inp, concat_feature, num_filters,
                       kernel_size=3,  dropout=0.1, strides=2,):
        """ Implements a single upsample block

        Args:
            inp(arr):
                Input image or previous layer output
            concat_feature(arr):
                The downsample convolution feature that is to be concatenated
            num_filters(int):
                Number of filters for the convolution operation
            kernel_size(int):
                Size of filter
            dropout(float):
                The percentage of neurons to be dropped during training
            strides(int):
                Number of pixels for the filter to move before next
                convolution operation

        Returns:
            (list):
                Outputs of convolution and pooling layers
        """
        X = Conv2DTranspose(num_filters, kernel_size,
                            strides, padding="same")(inp)
        X = Concatenate()([X, concat_feature])
        X = Dropout(dropout)(X)
        X = self.conv_block(X, num_filters, kernel_size=3)
        return X

    def construct(self, shape=(256, 256, 3), num_classes=3, num_filters=16,
                  dropout=0.1, kernel_size=3, strides=2):
        """Constructs a Unet model by Downsampling and Upsampling

        Args:
            num_classes(int):
                Number of desired classes in the output
            num_filters(int):
                Number of filters to be used at each stage
            dropout(float):
                The percentage of neurons to be dropped during training
            strides(int)
                Number of pixels for the filter to move before next
                convolution operation

        Returns:
            (keras.models.Model)
                A unet model
        """
        # Input Layer
        input = Input(shape=shape)

        # Encoder or Downsample
        c1, p1 = self.downsample_block(input, num_filters,
                                       kernel_size, dropout)

        c2, p2 = self.downsample_block(p1, num_filters*2,
                                       kernel_size, dropout)

        c3, p3 = self.downsample_block(p2, num_filters*4,
                                       kernel_size, dropout)

        c4, p4 = self.downsample_block(p3, num_filters*8,
                                       kernel_size, dropout)

        c5 = self.conv_block(p4, num_filters*16, kernel_size=kernel_size)

        # Upsample or Decoder

        u6 = self.upsample_block(c5, c4, num_filters*8, kernel_size,
                                 dropout, strides)

        u7 = self.upsample_block(u6, c3, num_filters*4, kernel_size,
                                 dropout, strides)

        u8 = self.upsample_block(u7, c2, num_filters*2, kernel_size,
                                 dropout, strides)

        u9 = self.upsample_block(u8, c1, num_filters*1, kernel_size,
                                 dropout, strides)

        # Output Layer
        output = Conv2D(num_classes, kernel_size=1, padding="same",
                        activation="softmax")(u9)

        self.model = Model(input, output, name="U-net")
        return self.model

    def view_model_summary(self, model, fname="unet-model.png", plot=False):
        """ Prints model summary and Plots the given model or constructed unet model
        """
        if plot:
            plot_model(model, to_file=fname)
        print(model.summary())


# un = Unet()
# model = un.construct()
# un.view_model_summary(model)
