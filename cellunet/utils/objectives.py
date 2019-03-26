import keras.metrics
import tensorflow as tf


def weighted_crossentropy(y_true, y_pred, weights):

    class_weights = tf.constant([[[[weights[0], weights[1], weights[2]]]]])

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)

    weights = tf.reduce_sum(class_weights * y_true, axis=-1)

    weighted_losses = weights * unweighted_losses

    loss = tf.reduce_mean(weighted_losses)

    return loss

