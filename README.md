# TO-DO 
- [x] Add codecarbon measures
- [ ] Add auxiliary losses | manage nb aux losses with SEN keys
- [ ] Add normalization for all modalities : config + dataloader
- [ ] Check labels starting at 0/1
- [ ] MNT/MNS : allow pass the two bands and not calc elev
- [ ] MNT/MNS : check if the 0/1 normalisation works
- [ ] Add auxiliary head (stage 3 SWIN) to monotemp ?
- [ ] Modify reshape_to_labels and MetadataMLP to extract self.list_x.shape[-2:]
- [ ] Check alignement spatial dimensions aerial RLT
- [ ] If multitemp only, verify base_shape
- [ ] Test LPIS labels
- [ ] Add to config : filtering values S2



<div align="center">
  
# FLAIR-INC
# Semantic segmentation.... 


![Static Badge](https://img.shields.io/badge/Code%3A-lightgrey?color=lightgrey) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/IGNF/FLAIR-1-AI-Challenge/blob/master/LICENSE) <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> &emsp; ![Static Badge](https://img.shields.io/badge/Dataset%3A-lightgrey?color=lightgrey) [![license](https://img.shields.io/badge/License-IO%202.0-green.svg)](https://github.com/etalab/licence-ouverte/blob/master/open-licence.md)




Participate in obtaining more accurate maps for a more comprehensive description and a better understanding of our environment! Come push the limits of state-of-the-art semantic segmentation approaches on a large and challenging dataset. Get in touch at :email: flair@ign.fr



![Alt bandeau FLAIR-IGN](images/flair_bandeau.jpg?raw=true)

</div>

<div style="border-width:1px; border-style:solid; border-color:#d2db8c; padding-left: 1em; padding-right: 1em; ">
  
<h2 style="margin-top:5px;">:mag_right:Quicklinks</h2>


- **Datapaper :** 

- **Dataset links :** 

- **Pre-trained models :** 

</div>


## Context & Data

<br>

<!--
<figure>
  <img
  src="images/flair-1_spatiotemporal.png"
  alt="ortho image and train/test geographical repartition">
  <figcaption>ORTHO HR® aerial image cover of France (left), train and test spatial domains of the dataset (middle) and acquisition months defining temporal domains (right).</figcaption>
</figure>


<br>
<br>

<p align="center">
  <img width="70%" src="images/flair-1_patches.png">
  <br>
  <em>Example of input data (first three columns) and corresponding supervision masks (last column).</em>
</p>
-->

```
flair_data = {
1   : ['building','#db0e9a'] ,
2   : ['pervious surface','#938e7b'],
3   : ['impervious surface','#f80c00'],
4   : ['bare soil','#a97101'],
5   : ['water','#1553ae'],
6   : ['coniferous','#194a26'],
7   : ['deciduous','#46e483'],
8   : ['brushwood','#f3a60d'],
9   : ['vineyard','#660082'],
10  : ['herbaceous vegetation','#55ff00'],
11  : ['agricultural land','#fff30d'],
12  : ['plowed land','#e4df7c'],
13  : ['swimming_pool','#3de6eb'],
14  : ['snow','#ffffff'],
15  : ['clear cut','#8ab3a0'],
16  : ['mixed','#6b714f'],
17  : ['ligneous','#c5dc42'],
18  : ['greenhouse','#9999ff'],
19  : ['other','#000000'],
}
```


<br>


## Baseline models


<br>

## Pre-trained models

<b>Pre-trained models &#9889;&#9889;&#9889;</b>  

<br>

## Lib usage 

<br><br>

### Installation :pushpin:

```bash
# it's recommended to install on a conda virtual env
conda create -n FLAIR_INC -c conda-forge python=3.12.5
conda activate FLAIR_INC
git clone git@github.com:IGNF/FLAIR-INC.git
cd FLAIR_INC*
pip install -e .
# if torch.cuda.is_available() returns False, do the following :
# pip install torch>=2.0.0 --extra-index-url=https://download.pytorch.org/whl/cu117

```

<br><br>

### Tasks :mag_right:

This library comprises two main entry points:<br>

#### :file_folder: flair

The flair module is used for training, inference and metrics calculation at the patch level. To use this pipeline :

```bash
flair --conf=/my/conf/file.yaml
```
This will perform the tasks specified in the configuration file. If ‘train’ is enabled, it will train the model and save the trained model to the output folder. If ‘predict’ is enabled, it will load the trained model (or a specified checkpoint if ‘train’ is not enabled) and perform prediction on the test data. If ‘metrics’ is enabled, it will calculate the mean Intersection over Union (mIoU) and other IoU metrics for the predicted and ground truth masks.
A toy dataset (reduced size) is available to check that your installation and the information in the configuration file are correct.
Note: A notebook is available in the legacy-torch branch (which uses different libraries versions and structure) that was used during the challenge.

#### :file_folder: zone_detect
This module aims to infer a pre-trained model at a larger scale than individual patches. It allows overlapping inferences using a margin argument. Specifically, this module expects a single georeferenced TIFF file as input.

```bash
flair-detect --conf=/my/conf/file-detect.yaml
```

<br><br>

### Configuration for flair :page_facing_up:

The pipeline is configured using a YAML file (`flair-1-config.yaml`). The configuration file includes sections for data paths, tasks, model configuration, hyperparameters and computational resources.

`out_folder`: The path to the output folder where the results will be saved.<br>
`out_model_name`: The name of the output model.<br>
`train_csv`: Path to the CSV file containing paths to image-mask pairs for training.<br>
`val_csv`: Path to the CSV file containing paths to image-mask pairs for validation.<br>
`test_csv`: Path to the CSV file containing paths to image-mask pairs for testing.<br>
`ckpt_model_path`: The path to the checkpoint file of the model. Used if train_tasks/init_weights_only_from_ckpt or resume_training_from_ckpt is True and for prediction if train is disabled.<br>
`path_metadata_aerial`: The path to the aerial metadata JSON file if used with FLAIR data and `model_provider` is SegmentationModelsPytorch.<br><br>


`train`: If set to True, the model will be trained.<br>
`init_weights_only_from_ckpt`: Use if fine-tuning to load weights from the ckpt file and perform training<br>
`resume_training_from_ckpt`: Use if you want to resume an aborted training or complete a training. This will load the weights, optimizer, scheduler and all relevant hyperparameters from the provided ckpt.<br><br>
`predict`: If set to True, predictions will be made using the model.<br>
`metrics`: If set to True, metrics will be calculated.<br>
`delete_preds`: Remove prediction files after metrics calculation.<br><br>

`model_provider`: the library providing models, either HuggingFace or SegmentationModelsPytorch.<br>
`org_model`: to be used if `model_provider` is HuggingFace in the form HFOrganization_Modelname, e.g., "openmmlab/upernet-swin-small".<br>
`encoder_decoder`: to be used if `model_provider` is SegmentationModelsPytorch in the form encodername_decoder_name, e.g., "resnet34_unet".<br><br>

`use_augmentation`: If set to True, data augmentation will be applied during training.<br>
`use_metadata`: If set to True, metadata will be used. If other than the FLAIR dataset, see structure to be provided.<br><br>

`channels`: The channels opened in your input images. Images are opened with rasterio which starts at 1 for the first channel.<br>
`norm_type`: Normalization to be applied: scaling (linear interpolation in the range [0,1]), custom (center-reduced with provided means and standard deviantions), without.<br>
`norm_means`: If custom, means for each input band.<br>
`norm_stds`: If custom standard deviation for each input band.<br><br>

`seed`: The seed for random number generation to ensure reproducibility.<br>
`batch_size`: The batch size for training.<br>
`learning_rate`: The learning rate for training.<br>
`num_epochs`: The number of epochs for training.<br><br>

`use_weights`: If set to True, class weights will be used during training.<br>
`classes`: Dict of semantic classes with value in images as key and list [weight, classname] as value. See config file for an example.<br>

`georeferencing_output`: If set to True, the output will be georeferenced.<br><br>

`accelerator`: The type of accelerator to use (‘gpu’ or ‘cpu’).<br>
`num_nodes`: The number of nodes to use for training.<br>
`gpus_per_node`: The number of GPUs to use per node for training.<br>
`strategy`: The strategy to use for distributed training (‘auto’,‘ddp’,...).<br>
`num_workers`: The number of workers to use for data loading.<br><br>


`ckpt_save_also_last`: on top of best epoch will also save last epoch ckpt file in the same folder.<br>
`ckpt_verbose`: print whenever a ckpt file is saved.<br>
`ckpt_weights_only`: save only weights of model in ckpt for storage optimization. This prevents `resume_training_from_ckpt`.<br>
`ckpt_monitor`: metric to be monitored for saving ckpt files. By default val_loss.<br>
`ckpt_monitor_mode`: wether min or max of `ckpt_monitor` for saving a ckpt file.<br>
`ckpt_earlystopping_patience`: ending training if no improvement after defined number of epochs. Default is 30.<br><br>

`cp_csv_and_conf_to_output`: Makes a copy of paths csv and config file to the output directory.<br>
`enable_progress_bar`: If set to True, a progress bar will be displayed during training and inference.<br>
`progress_rate`: The rate at which progress will be displayed.<br>

<br><br>

### Configuration for zone_detect :page_facing_up:

The pipeline is configured using a YAML file (`flair-1-config-detect.yaml`).

`output_path`: path to output result.<br>
`output_name`: name of resulting raster.<br><br>

`input_img_path` : path to georeferenced raster.<br>
`bands` : bands to be used in your raster file.<br><br>

`img_pixels_detection` : size in pixels of infered patches, default is 512.<br>
`margin` : margin between patchs for overlapping detection. 128 by exemple means that every 128*resolution step, a patch center will be computed.<br>
`output_type` : type of output, can be "class_prob" for integer between 0 and 255 representing the output of the model or "argmax" which will output only one band with the index of the class.<br>
`n_classes` : number of classes.<br><br>

`model_weights` : path to your model weights or checkpoint.<br>
`batch_size` : size of batch in dataloader, default is 2.<br> 
`use_gpu` : boolean, rather use gpu or cpu for inference, default is true.<br>
`model_name` : name of the model in pytorch segmentation models, default is 'unet'.<br>
`encoder_name` :  Name of the encoder from pytorch segmentation model, default is 'resnet34'.<br>
`num_worker` : number of worker used by dataloader, value should not be set at a higher value than 2 for linux because paved detection can have concurrency issues compared with traditional detection and set to 0 for mac and windows (gdal implementation's problem).<br><br>

`write_dataframe` : wether to write the dataframe of raster slicing to a file.<br><br>

`norm_type`: Normalization to be applied: scaling (linear interpolation in the range [0,1]) or custom (center-reduced with provided means and standard deviantions).<br>
`norm_means`: If custom, means for each input band.<br>
`norm_stds`: If custom standard deviation for each input band.<br><br>

<br><br>

## Baseline results

| AERIAL | AERIAL-RLT | SPOT | ELEV | SENTINEL-2 | SENTINEL-1 ASC | SENTINEL-1 DESC | MODEL | mIoU 
------------ | ------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| 




## Reference
Please include a citation to the following article if you use the FLAIR #1 dataset:

```bibtex
@article{garioud2022flair1,
  doi = {10.13140/RG.2.2.30183.73128/1},
  url = {https://arxiv.org/pdf/2211.12979.pdf},
  author = {Garioud, Anatol and Peillet, Stéphane and Bookjans, Eva and Giordano, Sébastien and Wattrelos, Boris},
  title = {FLAIR #1: semantic segmentation and domain adaptation dataset},
  publisher = {arXiv},
  year = {2022}
}
```

## Acknowledgment
This work was performed using HPC/AI resources from
GENCI-IDRIS (Grant 2022-A0131013803).

## Dataset license

The "OPEN LICENCE 2.0/LICENCE OUVERTE" is a license created by the French government specifically for the purpose of facilitating the dissemination of open data by public administration. 
If you are looking for an English version of this license, you can find it on the official GitHub page at the [official github page](https://github.com/etalab/licence-ouverte).<br>
Applicable legislation : This licence is governed by French law.<br>
Compatibility of this licence : This licence has been designed to be compatible with any free licence that at least requires an acknowledgement of authorship, and specifically with the previous version of this licence as well as with the following licences: United Kingdom’s “Open Government Licence” (OGL), Creative Commons’ “Creative Commons Attribution” (CC-BY) and Open Knowledge Foundation’s “Open Data Commons Attribution” (ODC-BY).
