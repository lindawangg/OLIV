# Training Steps

## Directory Structure
```
+data
  -label_map file
  -train TFRecord file
  -eval TFRecord file
  -test TFRecord file
  +images
+models
  +ssd_mobilenetv2_1_224
    -pipeline config file
    +export
      -frozen .pb graph
      -checkpoint
      -ckpt
      +saved_model
  +ssd_mobilenetv2_1_192
  +ssd_mobilenetv2_1_160
  +ssd_mobilenetv2_1_128
  +ssd_mobilenetv2_14_224
  +ssd_mobilenetv2_075_224
  +ssd_mobilenetv2_05_224
```

## Datasets
1. Download COCO 2017 train/val images and annotations
2. Move downloaded images and annotations to data folder
3. clone cocoapi in home directory and add to pythonpath
4. clone tensorflow models in home directory and add to pythonpath
5. create train and test csv files
6. create train/val and test TFRecord file
7. data is now ready for training

## Models
1. mobilenet v1 pretrained on ImageNet
https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html
2. mobilenet v2 pretrained on ImageNet https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md

## Training
1. Set CUDA_VISIBLE_DEVICES=X
2. Add models/research and models/research/slim to PYTHONPATH
3. run model_main.py
```sh
python object_detection/model_main.py \
  --pipeline_config_path=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/pipeline.config \
  --model_dir=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/ \
  --alsologtostderr
```
4. view progress on tensorboard
```sh
tensorboard --logdir=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/
```

## Evaluation
1. Set CUDA_VISIBLE_DEVICES=X
2. Set PYTHONPATH=${PATH_TO_models/research}
3. Change pipeline config eval path to test.record and set num_examples to exact number of test images
4. run model_main.py
```sh
python object_detection/model_main.py \
  --pipeline_config_path=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/pipeline.config \
  --model_dir=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/ \
  --checkpoint_dir=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/ \
  --alsologtostderr
```

## Exporting Model
1. Set CUDA_VISIBLE_DEVICES=X
2. Set PYTHONPATH=${PATH_TO_models/research}
3. Declare variables
```sh
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/pipeline.config
TRAINED_CKPT_PREFIX=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/model.ckpt-300000
EXPORT_DIR=/home/linda/OLIV/training/models/ssd_mobilenetv2_1_224/export/ssd_mobilenetv2_1_224_300000
```
4. From models/research, run object_detection/export_inference_graph.py
```sh
python object_detection/export_inference_graph.py \
  --input_type=$INPUT_TYPE \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --trained_checkpoint_prefix=$TRAINED_CKPT_PREFIX \
  --output_directory=$EXPORT_DIR
```

## Issues
1. https://github.com/tensorflow/models/issues/1795
