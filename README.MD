# Repository for HLS4ML for the bee pollen detection

## Setup

1. Install conda 
2. Create a new conda environment:
   ```bash
    conda env create -f environment.yml
   ```
   > I hope it works on your machine, if not, please let me know.
3. Activate the environment:
   ```bash
    conda activate hls4ml-bee
   ```
4. How to run the code:
   - To start the training, run:
     ```bash
     python train.py --config=./cfgs/bee_yolov3_tiny_only.yaml
     ```
   - The config file `bee_yolov3_tiny_only.yaml` contains the training parameters. (This file is used in the `train_slurm.sh` script as well.)
   - YAML files explanation:
     ```yaml
         yolo:
            type: "yolov3_tiny"
            # parameters for the YOLO model
            iou_threshold: 0.5
            score_threshold: 0.10
            max_boxes: 100
            strides: "32,16"
            anchors: "23,27 37,58 81,82 81,82 135,169 344,319"
            mask: "3,4,5 0,1,2"
            # file containing the class names of dataset. Now we ony use one class: bee
            name_path: "./data/bee_dataset/beedataset_only.name" 
         train:
            label: "bee_yolov3_tiny"
            # anotation file for training in the format: "image_name x1 y1 x2 y2 class_id, ..."
            # each line corresponds to one image
            anno_path: "./data/bee_dataset/labels/labels_scaled/bee_train_only_V2.txt"
            image_size: "416"

            batch_size: 4
            init_weight_path: "./ckpts/yolov3-tiny.h5"
            save_weight_path: "./ckpts"

            # do not change these parameters, they are specific for the YOLO model
            loss_type: "CIoU+FL"

            mosaic: true
            label_smoothing: true
            normal_method: true

            ignore_threshold: 0.5
         test:
            anno_path: "./data/bee_dataset/labels/labels_scaled/bee_test_only_V2.txt"
            image_size: "416"
            batch_size: 1
            init_weight_path: "./ckpts/yolov3_tiny.h5"
     ```
5. If you want to run the training on a SLURM cluster, use the provided `train_slurm.sh` script:
   ```bash
   sbatch train_slurm.sh
   ```
   else you can run the trainining directly with:
   ```bash
   ./train_local.sh
   ```
6. The results have following form:
   ```bash
   Running per image evaluation...
   DONE (t=0.14s).
   Accumulating evaluation results...
   DONE (t=0.01s).
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.638
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.194
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.296
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.081
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.387
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.475
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.379
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
   ```
   - To measure the performance of the model, we will use
   ```bash
      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] 

7. I also provide a script for prunning the model `train_prune.py`, which can be used to reduce the size of the model. You can run it with:
   ```bash
   python train_prune.py --config=./cfgs/bee_yolov3_tiny_only.yaml
   ```
   - The script will create a pruned model and save it in the `ckpts` directory.