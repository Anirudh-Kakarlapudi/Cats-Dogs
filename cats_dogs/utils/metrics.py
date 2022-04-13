""" This scripts defines various metrics and loss functions to be used for
training computer vision models especially for the image segmentation tasks

Author:
Anirudh Kakarlapudi
"""
import tensorflow as tf


def f1_loss(y_true, y_pred, smooth=1):
    """ Computes the f1_loss or dice loss value for the predictions.
    Generally works well for the class imbalance datasets

    Args:
        y_true(arr):
            Ground truth array containing one-hot encoded class
            labels with last axis being number of classes
        y_pred(arr):
            Predicted array with same dimensions as y_true
        smooth(float):
            Smoothing parameter to negate division by zero error

    Returns:
        (float):
            Computed F1 score value
    """
    y_true = tf.cast(y_true, dtype=tf.dtypes.float32)
    y_pred = tf.cast(y_pred, dtype=tf.dtypes.float32)
    f1_score_val = f1_score(y_true, y_pred, smooth)
    return smooth * (1 - f1_score_val)


def jaccard_coefficient(y_true, y_pred, smooth=0.001):
    """ The jaccard similarity coefficient is a statistic used for
    gauging the similarity and diversity among the prefictions and ground
    truth sets. Also known as Intersection over Union

    Jaccard Coefficient = IOU = Area of overlap/ Area of Union
                        = Intersection(A,B)/ Union(A,B)
                        = TP / (TP + FP + FN)

    Args:
        y_true(arr):
            Ground truth array containing one-hot encoded class
            labels with last axis being number of classes
        y_pred(arr):
            Predicted array with same dimensions as y_true
        smooth(float):
            Smoothing parameter to negate division by zero error

    Returns
        jaccard_coeff (float):
            Computed jaccard coefficient
    """
    y_true = tf.cast(y_true, dtype=tf.dtypes.float32)
    y_pred = tf.cast(y_pred, dtype=tf.dtypes.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection
    jaccard_coeff = (intersection + smooth) / (union + smooth)
    return tf.math.reduce_mean(jaccard_coeff)


def jaccard_loss(y_true, y_pred, smooth=100):
    """ Computes jaccard loss which inturn measures dissimilarity
    between the two arrays.

    Jaccard Loss =  1 - Jaccard Coefficient

    Args:
        y_true(arr):
            Ground truth array containing one-hot encoded class
            labels with last axis being number of classes
        y_pred(arr):
            Predicted array with same dimensions as y_true
        smooth(float):
            smoothing parameter to negate division by zero error

    Returns:
        (float):
            Computed jaccard loss value
    """
    y_true = tf.cast(y_true, dtype=tf.dtypes.float32)
    y_pred = tf.cast(y_pred, dtype=tf.dtypes.float32)
    jaccard_coeff = jaccard_coefficient(y_true, y_pred, smooth)
    return smooth* (1 - jaccard_coeff)


def f1_score(y_true, y_pred, smooth=1):
    """ Computes the f1_loss value for the predictions. Generally works
    well for the class imbalance datasets

    F1 Score = (2 * TP) / (2*TP + FP + FN)
             = (2 * IOU) / (IOU + 1)

    Args:
        y_true(arr):
            Ground truth array containing one-hot encoded class
            labels with last axis being number of classes
        y_pred(arr):
            Predicted array with same dimensions as y_true
        smooth(float):
            Smoothing parameter to negate division by zero error

    Returns(float):
        Computed F1 score value
    """
    y_true = tf.cast(y_true, dtype=tf.dtypes.float32)
    y_pred = tf.cast(y_pred, dtype=tf.dtypes.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return (2*intersection + smooth)/(denominator + smooth)
