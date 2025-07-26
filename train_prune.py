# -*- coding: utf-8 -*-
from __future__ import print_function


import os
import time

from absl import app, flags
from tensorflow.keras import optimizers

from core.utils import decode_cfg, load_weights
from core.dataset import Dataset
from core.callbacks import COCOEvalCheckpoint, CosineAnnealingScheduler, WarmUpScheduler
from core.utils.optimizers import Accumulative
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
import tensorflow as tf

def pruneFunction(layer):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.50, begin_step=NSTEPS * 2, end_step=NSTEPS * 10, frequency=NSTEPS
        )
    }
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer

flags.DEFINE_string('config', '', 'path to config file')
FLAGS = flags.FLAGS


def main(_argv):
    print('Config File From:', FLAGS.config)
    cfg = decode_cfg(FLAGS.config)

    model_type = cfg['yolo']['type']
    if model_type == 'yolov3':
        from core.model.one_stage.yolov3 import YOLOv3 as Model
        from core.model.one_stage.yolov3 import YOLOLoss as Loss
        num = 186
        epochs = 200
    elif model_type == 'yolov3_tiny':
        from core.model.one_stage.yolov3 import YOLOv3_Tiny as Model
        from core.model.one_stage.yolov3 import YOLOLoss as Loss
        num = 29
        epochs = 200
    elif model_type == 'yolov4':
        from core.model.one_stage.yolov4 import YOLOv4 as Model
        from core.model.one_stage.yolov4 import YOLOLoss as Loss
        num = 251
        epochs = 200
    elif model_type == 'yolov4_tiny':
        from core.model.one_stage.yolov4 import YOLOv4_Tiny as Model
        from core.model.one_stage.yolov4 import YOLOLoss as Loss
        num = 61
        epochs = 200
    elif model_type == 'yolox':
        from core.model.one_stage.custom import YOLOX as Model
        from core.model.one_stage.custom import YOLOLoss as Loss
        num = 61
        epochs = 80
    elif model_type == 'unofficial_yolov4_tiny':
        from core.model.one_stage.custom import Unofficial_YOLOv4_Tiny as Model
        from core.model.one_stage.custom import YOLOLoss as Loss
        num = 29
        epochs = 80
    else:
        raise NotImplementedError()


    model, eval_model = Model(cfg)
    model.summary()
    eval_model.summary()
    train_dataset = Dataset(cfg)

    init_weight = cfg["train"]["init_weight_path"]
    anchors = cfg['yolo']['anchors']
    mask = cfg['yolo']['mask']
    strides = cfg['yolo']['strides']
    ignore_threshold = cfg['train']['ignore_threshold']
    loss_type = cfg['train']['loss_type']
    
    epoch_steps =  len(train_dataset)
    print(len(train_dataset))
    if init_weight:
        load_weights(model, init_weight)
    else:
        print("Training from scratch")
        num = 0

    loss = [Loss(anchors[mask[i]],
                 strides[i],
                 train_dataset.num_classes,
                 ignore_threshold,
                 loss_type) for i in range(len(mask))]

    ckpt_path = os.path.join(cfg["train"]["save_weight_path"], 'tmp', cfg["train"]["label"],
                             time.strftime("%Y%m%d%H%M", time.localtime()))

    warmup_callback = [WarmUpScheduler(learning_rate=1e-3, warmup_step=1 * epoch_steps, verbose=1)]

    eval_callback = [COCOEvalCheckpoint(save_path=os.path.join(ckpt_path, "mAP-{mAP:.4f}.h5"),
                                        eval_model=eval_model,
                                        model_cfg=cfg,
                                        verbose=1)
                     ]
    lr_callback = [CosineAnnealingScheduler(learning_rate=1e-2,
                                            eta_min=1e-6,
                                            T_max=epochs * epoch_steps,
                                            verbose=1)]

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
        os.makedirs(os.path.join(ckpt_path, 'train', 'plugins', 'profile'))

    #opt = Accumulative(optimizers.RMSprop(lr=0.01), 16)
    opt = optimizers.RMSprop(lr=0.01)
    # warm-up
    for i in range(num):
        model.layers[i].trainable = True
        print(model.layers[i].name)
    print('Freeze the first {} layers of total {} layers.'.format(num, len(model.layers)))

    for i in range(len(model.layers)): model.layers[i].trainable = True

    model.compile(loss=loss, optimizer=opt, run_eagerly=False)
  
    epoch_steps =  len(train_dataset)

    print('Number of training steps per epoch is {}'.format(epoch_steps))



    # Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs,
    # ending by the 10th epoch
    

    train = True  # True if you want to retrain, false if you want to load a previsously trained model

    n_epochs = 30
    NSTEPS = epoch_steps

    def pruneFunction(layer):
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(
                initial_sparsity=0.0, final_sparsity=0.50, begin_step=NSTEPS * 2, end_step=NSTEPS * 10, frequency=NSTEPS
            )
        }
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer


    model_pruned = tf.keras.models.clone_model(model, clone_function=pruneFunction)

    if train:
        OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=3e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)

        model_pruned.compile(loss=loss, optimizer=OPTIMIZER)

        callbacks = eval_callback + eval_callback + [pruning_callbacks.UpdatePruningStep()]

        start = time.time()
        model_pruned.fit(train_dataset, epochs=n_epochs, callbacks=callbacks)
        end = time.time()

        print('It took {} minutes to train Keras model'.format((end - start) / 60.0))

        model_pruned.save('pruned_cnn_model.h5')

    else:
        from qkeras.utils import _add_supported_quantized_objects
        from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

        co = {}
        _add_supported_quantized_objects(co)
        co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
        model_pruned = tf.keras.models.load_model('pruned_cnn_model.h5', custom_objects=co)
if __name__ == "__main__":
    app.run(main)
