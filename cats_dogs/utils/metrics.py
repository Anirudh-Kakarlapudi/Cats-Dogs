import tensorflow.keras.backend as K
import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-15):
    """ Computes the dice_coefficient value for the predictions. Generally works
    well for the class imbalance datasets

    Args:
    y_true(arr):
        ground truth
    y_pred(arr):
        prediction
    smooth(float):
        smoothing parameter to negate division by zero error

    Returns(float):
        dice coefficient
    """
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """ Computes the dice loss
    Args:
    y_true(arr):
        ground truth
    y_pred(arr):
        prediction

    Returns(float):
        dice coefficient
    """
    return 1.0 - dice_coefficient(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1e-15):
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(K.flatten(y_pred), tf.float32)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f)+K.sum(y_pred_f)-intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou