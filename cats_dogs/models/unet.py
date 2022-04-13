""" This script allows user to build u-nets by constructing encoder, decoder
blocks separately and to view the constructed model

Author:
Anirudh Kakarlapudi
"""
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def conv_block(inp, n_filters=16, k_size=3):
    """ Implements a Convolution layer followed by Relu activation twice

    Args:
        inp(arr):
            Input image or previous layer output
        n_filters(int):
            Number of filters for the convolution operation
        k_size(int):
            Size of kernel

    Returns:
        conv(arr):
            The output of the convolution block
    """
    conv = layers.SeparableConv2D(n_filters, k_size, padding="same")(inp)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.SeparableConv2D(n_filters, k_size, padding="same")(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    return conv


def encoder_block(inp, n_filters, k_size=3, dropout=0.3):
    """ Implements a single down-sample block

    Args:
        inp(arr):
            Input image or previous layer output
        n_filters(int):
            Number of filters for the convolution operation
        k_size(int):
            Size of kernel
        dropout(float):
            Percentage of neurons tobe dropped during training

    Returns:
        (list):
            Outputs of convolution and pooling layers
    """
    conv = conv_block(inp, n_filters, k_size)
    pool = layers.MaxPooling2D((2, 2))(conv)

    if dropout > 0:
        pool = layers.Dropout(dropout)(pool)

    return conv, pool


def decoder_block(inp, concat_feature, n_filters,
                  k_size=3,  dropout=0.1, strides=2):
    """ Implements a single up-sample block

    Args:
        inp(arr):
            Input image or previous layer output
        concat_feature(arr):
            The down sample convolution feature that is to be concatenated
        n_filters(int):
            Number of filters for the convolution operation
        k_size(int):
            Size of kernel
        dropout(float):
            The percentage of neurons to be dropped during training
        strides(int):
            Number of pixels for the filter to move before next
            convolution operation

    Returns:
        (list):
            Outputs of convolution and pooling layers
    """
    conv = layers.Conv2DTranspose(n_filters, k_size, strides,
                                  padding="same")(inp)
    concat = layers.Concatenate()([conv, concat_feature])
    if dropout > 0:
        concat = layers.Dropout(dropout)(concat)

    return concat


def construct_model(img_shape, n_classes=3, n_filters=16,
                    dropout=0.3, k_size=3, strides=2):
    """Constructs a U-net model

        Args:
            img_shape(tuple):
                Shape of input image
            n_classes(int):
                Number of desired classes in the output
            n_filters(int):
                Number of filters to be used at each stage
            k_size(int):
                Size of kernel
            dropout(float):
                The percentage of neurons to be dropped during training
            strides(int)
                Number of pixels for the filter to move before next
                convolution operation

        Returns:
            (keras.models.Model)
                An u-net model
    """
    # Input Layer
    inp = layers.Input(shape=img_shape)

    # Encoder or Downs-ample
    conv1, pool1 = encoder_block(inp, n_filters, k_size, dropout)

    conv2, pool2 = encoder_block(pool1, n_filters * 2, k_size, dropout)

    conv3, pool3 = encoder_block(pool2, n_filters * 4, k_size, dropout)
    conv4, pool4 = encoder_block(pool3, n_filters * 8, k_size, dropout)

    conv5 = conv_block(pool4, n_filters * 16, k_size)

    # Up-sample or Decoder

    up6 = decoder_block(conv5, conv4, n_filters * 8, k_size, dropout, strides)

    up7 = decoder_block(up6, conv3, n_filters * 4, k_size, dropout, strides)

    up8 = decoder_block(up7, conv2, n_filters * 2, k_size, dropout, strides)

    up9 = decoder_block(up8, conv1, n_filters * 1, k_size, dropout, strides)

    # Output Layer - Classification layer
    output = layers.Conv2D(n_classes, kernel_size=k_size, padding="same",
                           activation="softmax")(up9)

    model = Model(inp, output, name="U-net")
    return model


def view_model_summary(model, f_name="model_unet.png", plot=False):
    """Constructs a U-net model

    Args:
        model (keras.models.Model):
            A model that is built using a keras.models.Model
        plot(bool):
            If true, plots the image of the model and saves it into file
        f_name(str):
            name of the output filename
    """
    if plot:
        plot_model(model, to_file=f_name)
    model.summary()
