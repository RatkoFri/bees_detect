



import numpy as np
import plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

seed = 0
np.random.seed(seed)
import tensorflow as tf

tf.random.set_seed(seed)
import os

from tensorflow import keras

@keras.saving.register_keras_serializable()
class PreprocessInput(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(PreprocessInput, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PreprocessInput, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        x = tf.divide(inputs, 255.)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

@keras.saving.register_keras_serializable()
class RouteGroup(tf.keras.layers.Layer):

    def __init__(self, ngroups, group_id, **kwargs):
        super(RouteGroup, self).__init__(**kwargs)
        self.ngroups = ngroups
        self.group_id = group_id

    def call(self, inputs, *args, **kwargs):
        convs = tf.split(inputs, num_or_size_splits=self.ngroups, axis=-1)
        return convs[self.group_id]

    def get_config(self):
        config = super(RouteGroup, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.saving.register_keras_serializable()
def YOLOLoss(anchors, stride, num_classes, ignore_thresh, type):
    @keras.saving.register_keras_serializable()
    def wrapper(y_true, y_pred):
        # 0. default
        dtype = y_pred.dtype
        y_shape = tf.shape(y_pred)
        grid_w, grid_h = y_shape[2], y_shape[1]
        anchors_tensor = tf.cast(anchors, dtype)
        y_true = tf.reshape(y_true, (y_shape[0], y_shape[1], y_shape[2], anchors_tensor.shape[0], num_classes + 5))
        y_pred = tf.reshape(y_pred, (y_shape[0], y_shape[1], y_shape[2], anchors_tensor.shape[0], num_classes + 5))

        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_xy, pred_wh, pred_obj, pred_cls = tf.split(y_pred, (2, 2, 1, num_classes), axis=-1)

        # !!! grid[x][y] == (y, x)
        grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        pred_xy = (tf.sigmoid(pred_xy) + tf.cast(grid, dtype)) * stride
        pred_wh = tf.exp(pred_wh) * anchors_tensor
        pred_obj = tf.sigmoid(pred_obj)
        pred_cls = tf.sigmoid(pred_cls)

        pred_wh_half = pred_wh / 2.
        pred_x1y1 = pred_xy - pred_wh_half
        pred_x2y2 = pred_xy + pred_wh_half
        pred_box = tf.concat([pred_x1y1, pred_x2y2], axis=-1)

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_cls = tf.split(y_true, (4, 1, num_classes), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2.
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1] / (
            tf.cast(tf.reduce_prod([grid_w, grid_h, stride, stride]), dtype))

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(_broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            dtype)
        ignore_mask = tf.cast(best_iou < ignore_thresh, dtype)

        # 5. calculate all losses
        if 'L2' in type:
            xy_loss = 0.5 * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            wh_loss = 0.5 * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
            box_loss = xy_loss + wh_loss
        elif 'GIoU' in type:
            giou = GIoU(pred_box, true_box)
            box_loss = 1. - giou
        elif 'DIoU' in type:
            diou = DIoU(pred_box, true_box)
            box_loss = 1. - diou
        elif 'CIoU' in type:
            ciou = CIoU(pred_box, true_box)
            box_loss = 1. - ciou
        else:
            raise NotImplementedError('Loss Type', type, 'is Not Implemented!')

        box_loss = obj_mask * box_loss_scale * box_loss
        obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        cls_loss = obj_mask * tf.keras.losses.binary_crossentropy(true_cls, pred_cls)

        def _focal_loss(y_true, y_pred, alpha=1, gamma=2):
            focal_loss = tf.squeeze(alpha * tf.pow(tf.abs(y_true - y_pred), gamma), axis=-1)
            return focal_loss

        if 'FL' in type:
            focal_loss = _focal_loss(true_obj, pred_obj)
            obj_loss = focal_loss * obj_loss

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        box_loss = tf.reduce_mean(tf.reduce_sum(box_loss, axis=(1, 2, 3)))
        obj_loss = tf.reduce_mean(tf.reduce_sum(obj_loss, axis=(1, 2, 3)))
        cls_loss = tf.reduce_mean(tf.reduce_sum(cls_loss, axis=(1, 2, 3)))

        return box_loss + obj_loss + cls_loss

    return wrapper



from tensorflow.keras.models import load_model
model = load_model('final_modelV5.h5', custom_objects={'YOLOLoss': YOLOLoss })

import hls4ml


#custom_objects={'PreprocessInput': PreprocessInput, 'RouteGroup': RouteGroup(ngroups=2, group_id=1)}

config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')
print("-----------------------------------")
print("Configuration")
plotting.print_dict(config)
# plot to file
plotting.plot_dict(config, 'config.png')