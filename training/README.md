# Training Steps

## Datasets
1. Download COCO 2017 train/val images and annotations
2. Move downloaded images and annotations to data folder
3. clone cocoapi in home directory and add to pythonpath
4. clone tensorflow models in home directory and add to pythonpath
5. create train and test csv files
6. create train/val and test TFRecord file
7. data is now ready for training

## Models
1. ssd mobilenet v2  

## Training
1. Set CUDA_VISIBLE_DEVICES=X
2. Add models/research and models/research/slim to PYTHONPATH
3. run model_main.py
python object_detection/model_main.py --pipeline_config_path=/home/linda/OLIV/training/models/ssd_mobilenet_1_224/pipeline.config --model_dir=/home/linda/OLIV/training/models/ssd_mobilenet_1_224/ --alsologtostderr
