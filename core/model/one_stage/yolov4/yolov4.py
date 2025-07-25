# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

from core.losses.iou import GIoU, DIoU, CIoU

WEIGHT_DECAY = 0.  # 5e-4
LEAKY_ALPHA = 0.1


def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {"kernel_regularizer": tf.keras.regularizers.l2(WEIGHT_DECAY),
                           "kernel_initializer": tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                           "padding": "valid" if kwargs.get(
                               "strides") == (2, 2) else "same"}
    darknet_conv_kwargs.update(kwargs)

    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)
    #return FakeApproxConv2Dv2(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    without_bias_kwargs = {"use_bias": False}
    without_bias_kwargs.update(kwargs)

    def wrapper(x):
        x = DarknetConv2D(*args, **without_bias_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)(x)
        #x = tf.keras.layers.ReLU()(x)
        return x

    return wrapper


def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    def wrapper(x):
        x = DarknetConv2D(*args, **no_bias_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Mish()(x)
        return x

    return wrapper


def DarknetBlock(num_filters, niter, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''

    # Darknet uses left and top padding instead of 'same' mode
    def wrapper(x):
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(x)
        shortcut = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)
        x = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)
        for _ in range(niter):
            y = DarknetConv2D_BN_Mish(num_filters // 2, (1, 1))(x)
            y = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3))(y)

            x = tf.keras.layers.Add()([x, y])
        x = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)
        x = tf.keras.layers.Concatenate()([x, shortcut])
        x = DarknetConv2D_BN_Mish(num_filters, (1, 1))(x)
        return x

    return wrapper


def YOLOv4_Tiny(cfg,
                input_size=None,
                name=None):
    iou_threshold = cfg["yolo"]["iou_threshold"]
    score_threshold = cfg["yolo"]["score_threshold"]
    max_outputs = cfg["yolo"]["max_boxes"]
    num_classes = cfg["yolo"]["num_classes"]
    strides = cfg["yolo"]["strides"]
    mask = cfg["yolo"]["mask"]
    anchors = cfg["yolo"]["anchors"]

    if input_size is None:
        x = inputs = tf.keras.Input([416, 416, 3])
    else:
        x = inputs = tf.keras.Input([input_size, input_size, 3])

    x = PreprocessInput()(x)

    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(32, (3, 3), strides=(2, 2))(x)  # 0
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(64, (3, 3), strides=(2, 2))(x)  # 1
    x = DarknetConv2D_BN_Leaky(64, (3, 3))(x)  # 2

    route = x
    x = RouteGroup(2, 1)(x)  # 3
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)  # 4
    route_1 = x
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)  # 5
    x = tf.keras.layers.Concatenate()([x, route_1])  # 6
    x = DarknetConv2D_BN_Leaky(64, (1, 1))(x)  # 7
    x = tf.keras.layers.Concatenate()([route, x])  # 8
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 9

    x = DarknetConv2D_BN_Leaky(128, (3, 3))(x)  # 10
    route = x
    x = RouteGroup(2, 1)(x)  # 11
    x = DarknetConv2D_BN_Leaky(64, (3, 3))(x)  # 12
    route_1 = x
    x = DarknetConv2D_BN_Leaky(64, (3, 3))(x)  # 13
    x = tf.keras.layers.Concatenate()([x, route_1])  # 14
    x = DarknetConv2D_BN_Leaky(128, (1, 1))(x)  # 15
    x = tf.keras.layers.Concatenate()([route, x])  # 16
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 17

    x = DarknetConv2D_BN_Leaky(256, (3, 3))(x)  # 18
    route = x
    x = RouteGroup(2, 1)(x)  # 19
    x = DarknetConv2D_BN_Leaky(128, (3, 3))(x)  # 20
    route_1 = x
    x = DarknetConv2D_BN_Leaky(128, (3, 3))(x)  # 21
    x = tf.keras.layers.Concatenate()([x, route_1])  # 22
    x = x_23 = DarknetConv2D_BN_Leaky(256, (1, 1))(x)  # 23
    x = tf.keras.layers.Concatenate()([route, x])  # 24
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 25
    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)

    x = DarknetConv2D_BN_Leaky(256, (1, 1))(x)

    _x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)
    output_0 = DarknetConv2D(len(mask[0]) * (num_classes + 5), (1, 1))(_x)

    x = DarknetConv2D_BN_Leaky(128, (1, 1))(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, x_23])

    x = DarknetConv2D_BN_Leaky(256, (3, 3))(x)
    output_1 = DarknetConv2D(len(mask[1]) * (num_classes + 5), (1, 1))(x)

    model = tf.keras.Model(inputs, [output_0, output_1], name=name)

    outputs = Header(num_classes, anchors, mask, strides, max_outputs, iou_threshold, score_threshold)(
        (output_0, output_1))

    eval_model = tf.keras.Model(inputs, outputs, name=name)

    return model, eval_model


def YOLOv4(cfg,
           input_size=None,
           name=None):
    iou_threshold = cfg["yolo"]["iou_threshold"]
    score_threshold = cfg["yolo"]["score_threshold"]
    max_outputs = cfg["yolo"]["max_boxes"]
    num_classes = cfg["yolo"]["num_classes"]
    strides = cfg["yolo"]["strides"]
    mask = cfg["yolo"]["mask"]
    anchors = cfg["yolo"]["anchors"]

    if input_size is None:
        x = inputs = tf.keras.Input([None, None, 3])
    else:
        x = inputs = tf.keras.Input([input_size, input_size, 3])

    x = tf.divide(x, 255.)

    x = DarknetConv2D_BN_Mish(32, (3, 3))(x)
    x = DarknetBlock(64, 1, False)(x)
    x = DarknetBlock(128, 2)(x)
    x = x_131 = DarknetBlock(256, 8)(x)
    x = x_204 = DarknetBlock(512, 8)(x)
    x = DarknetBlock(1024, 4)(x)

    # 19x19 head
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(x)

    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.Concatenate()([maxpool1, maxpool2, maxpool3, x])
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    x = x_19 = DarknetConv2D_BN_Leaky(512, (1, 1))(x)

    x = DarknetConv2D_BN_Leaky(256, (1, 1))(x)
    x19_upsample = tf.keras.layers.UpSampling2D(2)(x)

    # 38x38 head
    x = DarknetConv2D_BN_Leaky(256, (1, 1))(x_204)
    x = tf.keras.layers.Concatenate()([x, x19_upsample])
    x = DarknetConv2D_BN_Leaky(256, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(256, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)
    x = x_38 = DarknetConv2D_BN_Leaky(256, (1, 1))(x)

    x = DarknetConv2D_BN_Leaky(128, (1, 1))(x)
    x38_upsample = tf.keras.layers.UpSampling2D(2)(x)

    # 76x76 head
    x = DarknetConv2D_BN_Leaky(128, (1, 1))(x_131)
    x = tf.keras.layers.Concatenate()([x, x38_upsample])
    x = DarknetConv2D_BN_Leaky(128, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(256, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(128, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(256, (3, 3))(x)
    x = x_76 = DarknetConv2D_BN_Leaky(128, (1, 1))(x)

    # 76x76 output
    x = DarknetConv2D_BN_Leaky(256, (3, 3))(x)
    output_2 = DarknetConv2D(len(mask[2]) * (num_classes + 5), (1, 1))(x)

    # 38x38 output
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x_76)
    x = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, x_38])
    x = DarknetConv2D_BN_Leaky(256, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(256, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)
    x = x_38 = DarknetConv2D_BN_Leaky(256, (1, 1))(x)

    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)
    output_1 = DarknetConv2D(len(mask[2]) * (num_classes + 5), (1, 1))(x)

    # 19x19 output
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x_38)
    x = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, x_19])
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(x)

    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    output_0 = DarknetConv2D(len(mask[2]) * (num_classes + 5), (1, 1))(x)

    model = tf.keras.Model(inputs, [output_0, output_1, output_2], name=name)

    outputs = Header(num_classes, anchors, mask, strides, max_outputs, iou_threshold, score_threshold)(
        (output_0, output_1, output_2))
    eval_model = tf.keras.Model(inputs, outputs, name=name)

    return model, eval_model

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
        #return config
        return {"ngroups": self.ngroups, "group_id": self.group_id}

    def compute_output_shape(self, input_shape):
        return input_shape

@keras.saving.register_keras_serializable()
class Mish(tf.keras.layers.Layer):
    """
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, *args, **kwargs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

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
class Header(tf.keras.layers.Layer):

    def __init__(self, num_classes, anchors, mask, strides,
                 max_outputs, iou_threshold, score_threshold, **kwargs):
        self.num_classes = num_classes
        self.anchors = anchors
        self.mask = mask
        self.strides = strides

        self.max_outputs = max_outputs
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        super(Header, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Header, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        boxes, objects, classes = [], [], []
        dtype = inputs[0].dtype
        for i, logits in enumerate(inputs):
            anchors, stride = self.anchors[self.mask[i]], self.strides[i]
            x_shape = tf.shape(logits)
            logits = tf.reshape(logits, (x_shape[0], x_shape[1], x_shape[2], len(anchors), self.num_classes + 5))

            box_xy, box_wh, obj, cls = tf.split(logits, (2, 2, 1, self.num_classes), axis=-1)
            box_xy = tf.sigmoid(box_xy)
            obj = tf.sigmoid(obj)
            cls = tf.sigmoid(cls)

            grid_shape = x_shape[1:3]
            grid_h, grid_w = grid_shape[0], grid_shape[1]
            anchors = tf.cast(anchors, dtype)
            grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
            grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

            box_xy = (box_xy + tf.cast(grid, dtype)) * stride
            box_wh = tf.exp(box_wh) * tf.cast(anchors, dtype)

            box_x1y1 = box_xy - box_wh / 2.
            box_x2y2 = box_xy + box_wh / 2.
            box = tf.concat([box_x1y1, box_x2y2], axis=-1)

            boxes.append(tf.reshape(box, (x_shape[0], -1, 1, 4)))
            objects.append(tf.reshape(obj, (x_shape[0], -1, 1)))
            classes.append(tf.reshape(cls, (x_shape[0], -1, self.num_classes)))

        boxes = tf.concat(boxes, axis=1)
        objects = tf.concat(objects, axis=1)
        classes = tf.concat(classes, axis=1)

        scores = objects * classes
        boxes, scores, classes, valid = tf.image.combined_non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size_per_class=self.max_outputs,
            max_total_size=self.max_outputs,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            clip_boxes=False
        )

        return boxes, scores, classes, valid

    def compute_output_shape(self, input_shape):
        return ([input_shape[0][0], self.max_outputs, 4],
                [input_shape[0][0], self.max_outputs],
                [input_shape[0][0], self.max_outputs],
                [input_shape[0][0]])


def _broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / tf.maximum(box_1_area + box_2_area - int_area, 1e-8)


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


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    x = inputs = tf.keras.Input([None, None, 3])
    x = PreprocessInput()(x)

    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(32, 3, strides=(2, 2))(x)  # 0
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(64, (3, 3), strides=(2, 2))(x)  # 1
    x = DarknetConv2D_BN_Leaky(64, (3, 3))(x)  # 2

    route = x
    x = RouteGroup(2, 1)(x)  # 3
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)  # 4
    route_1 = x
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)  # 5
    x = tf.keras.layers.Concatenate()([x, route_1])  # 6
    x = DarknetConv2D_BN_Leaky(64, (1, 1))(x)  # 7
    x = tf.keras.layers.Concatenate()([route, x])  # 8
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 9

    x = DarknetConv2D_BN_Leaky(128, (3, 3))(x)  # 10
    route = x
    x = RouteGroup(2, 1)(x)  # 11
    x = DarknetConv2D_BN_Leaky(64, (3, 3))(x)  # 12
    route_1 = x
    x = DarknetConv2D_BN_Leaky(64, (3, 3))(x)  # 13
    x = tf.keras.layers.Concatenate()([x, route_1])  # 14
    x = DarknetConv2D_BN_Leaky(128, (1, 1))(x)  # 15
    x = tf.keras.layers.Concatenate()([route, x])  # 16
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 17

    x = DarknetConv2D_BN_Leaky(256, (3, 3))(x)  # 18
    route = x
    x = RouteGroup(2, 1)(x)  # 19
    x = DarknetConv2D_BN_Leaky(128, (3, 3))(x)  # 20
    route_1 = x
    x = DarknetConv2D_BN_Leaky(128, (3, 3))(x)  # 21
    x = tf.keras.layers.Concatenate()([x, route_1])  # 22
    x = x_23 = DarknetConv2D_BN_Leaky(256, (1, 1))(x)  # 23
    x = tf.keras.layers.Concatenate()([route, x])  # 24
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 25

    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)

    model = tf.keras.Model(inputs, x)
    model.summary()
    print(len(model.layers))
