# SRFENet
Pytorch implementation of paper "Shrinkage and Redundant Feature Elimination Network Based Robust Image Zero-watermarking"

## Note: 
Our model is placed in the model.decoder, and the encoder and discriminator do not work in our paper
After the model is trained, a run folder appears, which is needed to test our model.

## Dataset
We use 10000 COCO images as the training data set and 1000 COCO verification set.In the test, we not only test the COCO data set, but also test the DIV2k data and VOC data set.
The data directory has the following structure:
<data_root>/
  train/
    train_class/
      train_image1.jpg
      train_image2.jpg
      ...
  val/
    val_class/
      val_image1.jpg
      val_image2.jpg
      ...
train_class and val_class folders are so that we can use the standard torchvision data loaders without change.

## Noise_add:
If you want to set some random noises,you can place these noise into noise_layers.Noiser file list rather than Command-line input 

## Running
If you start training for the first time, you can use
```
python main.py new --name <experiment_name> --data-dir <data_root> --batch-size <b> 
```
If you want to continue from a training run, use 
```
python main.py continue --folder <incomplete_run_folder>
```
if you want to test model ,you can use
```
python test_model_new.py -o <options_file_path> -c <checkpoint_file_path> -b <b> -s <test_data_path> -n <noise type>
```

