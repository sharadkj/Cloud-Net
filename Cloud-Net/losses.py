import tensorflow as tf
from keras import backend as K

smooth = 1e-7  # Better readability

def jacc_coef(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # Ensure float32
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)  # Ensure float32
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection + smooth))
