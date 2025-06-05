<div align="center">
  
# FLAIR-HUB
# Multimodal & multitask semantic segmentation of Earth Observation imagery


![Static Badge](https://img.shields.io/badge/Code%3A-lightgrey?color=lightgrey) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/IGNF/FLAIR-1-AI-Challenge/blob/master/LICENSE) <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> &emsp; ![Static Badge](https://img.shields.io/badge/Dataset%3A-lightgrey?color=lightgrey) [![license](https://img.shields.io/badge/License-IO%202.0-green.svg)](https://github.com/etalab/licence-ouverte/blob/master/open-licence.md)




Participate in obtaining more accurate maps for a more comprehensive description and a better understanding of our environment! Come push the limits of state-of-the-art semantic segmentation approaches on a large and challenging dataset. <br> Get in touch at :email: flair@ign.fr



![Alt bandeau FLAIR-HUB](images/flair_bandeau.jpg?raw=true)

</div>

<div style="border-width:1px; border-style:solid; border-color:#d2db8c; padding-left: 1em; padding-right: 1em; ">
  
<h2 style="margin-top:5px;">:mag_right:Quicklinks</h2>


- **Entry page :**  [FLAIR-HUB dataset page](https://ignf.github.io/FLAIR/flairhub) <br>
  
- **Datapaper :** 

- **Dataset links :**  [FLAIR-HUB dataset](https://huggingface.co/datasets/IGNF/FLAIR-HUB) 

- **Pre-trained models :** [FLAIR-HUB pretrained models](https://huggingface.co/collections/IGNF/flair-models-684035e78bd5bff99199ff87) 

</div>
<hr>



## Context & Data

```yaml
flair_LC = 
0    : 'building'                'bâtiment'                   '#db0e9a'
1    : 'greenhouse'              'serre'                      '#9999ff'
2    : 'swimming_pool'           'piscine'                    '#3de6eb' 
3    : 'impervious surface'      'zone imperméable'           '#f80c00' 
4    : 'pervious surface'        'zone perméable'             '#938e7b' 
5    : 'bare soil'               'sol nu'                     '#a97101' 
6    : 'water'                   'eau'                        '#1553ae' 
7    : 'snow'                    'neige'                      '#ffffff' 
8    : 'herbaceous vegetation'   'surface herbacée'           '#55ff00' 
9    : 'agricultural land'       'culture'                    '#fff30d' 
10   : 'plowed land'             'terre labourée'             '#e4df7c' 
11   : 'vineyard'                'vigne'                      '#660082' 
12   : 'deciduous'               'feuillu'                    '#46e483' 
13   : 'coniferous'              'conifère'                   '#194a26' 
14   : 'brushwood'               'broussaille'                '#f3a60d' 
15   : 'clear cut'               'coupe'                      '#8ab3a0' 
16   : 'ligneous'                'ligneux'                    '#c5dc42' 
17   : 'mixed'                   'mixte'                      '#6b714f'
18   : 'undefined'               'indéterminé'                '#000000'

flair_LPIS = 
0    : 'Grasses'                 'Herbe prédominante'         '#92d050' 
1    : 'Wheat'                   'Blé'                        '#d7e600' 
2    : 'Barley'                  'Orge'                       '#e0e000' 
3    : 'Maize'                   'Maïs'                       '#fff100'  
4    : 'Other cereals'           'Autres céréales'            '#ffff00' 
5    : 'Rice'                    'Riz'                        '#e8e8e8'
6    : 'Hemp/Flax/Tobacco'       'Lin/Chanvre/Tabac'          '#dceaf7' 
7    : 'Sunflower'               'Tournesol'                  '#d29ead' 
8    : 'Rapeseed'                'Colza'                      '#d29ed0' 
9    : 'Other oilseed crops'     'Autres oléagineux'          '#ffbe99' 
10   : 'Soy'                     'Soja'                       '#ffc000' 
11   : 'Other protein crops'     'Autres protéagineux'        '#ff9000' 
12   : 'Fodder legumes'          'Légumineuse fourragères'    '#009999' 
13   : 'Beetroots'               'Betterave'                  '#808000' 
14   : 'Potatoes'                'Pomme de terre'             '#a7a700' 
15   : 'Other arable crops'      'Autres arables'             '#89896d' 
16   : 'Vineyard'                'Vigne'                      '#f2cfee' 
17   : 'Olive groves'            'Oliveraie'                  '#6f6633' 
18   : 'Fruit orchards'          'Verger fruits'              '#ac8141' 
19   : 'Nut orchards'            'Verger noix'                '#996633' 
20   : 'Other permanent crops'   'Autres pérenne'             '#80c1d7' 
21   : 'Mixed crops'             'Mélange'                    '#000000' 
22   : 'Background'              'Autres'                     '#000000'

```





<br><br>

## Usage 



### Tasks :mag_right:
This library comprises two main entry points:

#### :file_folder: flair_hub | used to train, infer models, calculate metrics, at the patch level. <br>
#### :file_folder: flair_zonal_detection | used to infer a pretrained model over larger areas. <br>

### Configuration for flair_hub :page_facing_up:

The pipeline takes as input a folder with 4 configuration YAML files. The configuration file includes sections for data paths, tasks, supervision, model configuration, hyperparameters and computational resources.

#### config_task.yaml : <br>
```yaml
paths:
    out_folder: Directory to store all output results (models, logs, predictions).
    out_model_name: Name identifier for the saved model.
    train_csv: CSV files containing paths and labels for training datasets.
    val_csv: CSV files containing paths and labels for validation datasets.
    test_csv: CSV files containing paths and labels for test datasets.
    global_mtd_folder: Path to folder with global metadata required for processing.
    ckpt_model_path: Path to a pretrained model checkpoint to initialize weights.

tasks:
    train: Enable or disable training phase.
    train_tasks:
        init_weights_only_from_ckpt: Use checkpoint weights for initialization only (no training resume).
        resume_training_from_ckpt: Resume training from the provided checkpoint.
    predict: Run model inference on test set.
    write_files: Save predictions and outputs to disk.
    georeferencing_output: Apply georeferencing to output files.
    metrics_only: Compute and report metrics without running training or prediction.

hyperparams:
    num_epochs: Total number of training epochs.
    batch_size: Number of samples per training batch.
    seed: Random seed for reproducibility.
    learning_rate: Initial learning rate for optimizer.
    optimizer: Choice of optimizer (adamw, adam, or sgd).
    optim_weight_decay: Weight decay regularization.
    optim_betas: Beta coefficients for Adam-based optimizers.
    scheduler: Learning rate scheduler strategy.
    warmup_fraction: Warm-up fraction for one_cycle_lr.
    plateau_patience: Patience for reduce_on_plateau scheduler.

hardware:
    accelerator: Type of hardware to use (gpu, cpu).
    num_nodes: Number of nodes for distributed training.
    gpus_per_node: GPUs used per node.
    strategy: Distributed training strategy (e.g., auto, ddp).
    num_workers: Number of data loading workers.

saving:
    ckpt_save_also_last: Save the final model checkpoint at the end of training.
    ckpt_weights_only: Save only the model weights (not optimizer or scheduler).
    ckpt_monitor: Metric used to determine the best checkpoint.
    ckpt_monitor_mode: Direction to optimize the monitored metric (max or min).
    ckpt_earlystopping_patience: Epochs to wait for improvement before stopping.
    cp_csv_and_conf_to_output: Copy configuration and CSV files to output directory.
    enable_progress_bar: Show training progress bar.
    progress_rate: Update frequency of progress bar (in steps).
    ckpt_verbose: Print detailed checkpoint saving information.
    verbose_config: Print full config details to console.
```

#### config_modalities.yaml : <br>
```yaml
modalities:

    inputs:
        AERIAL_RGBI       : Enable/disable AERIAL_RGBI modality. [True/False]
        AERIAL-RLT_PAN    : [True/False]
        DEM_ELEV          : [True/False]
        SPOT_RGBI         : [True/False]
        SENTINEL2_TS      : [True/False].
        SENTINEL1-ASC_TS  : [True/False]
        SENTINEL1-DESC_TS : [True/False]
        
    inputs_channels:
        AERIAL_RGBI       : Selected channels for AERIAL_RGBI input. Starts at 1.
        SPOT_RGBI         : Selected channels for SPOT_RGBI input. Starts at 1.
        SENTINEL2_TS      : Selected channels for Sentinel-2 time series. Starts at 1.
        SENTINEL1-ASC_TS  : Selected channels for Sentinel-1 ascending time series. Starts at 1.
        SENTINEL1-DESC_TS : Selected channels for Sentinel-1 descending time series. Starts at 1.

    aux_loss: 
        AERIAL_RGBI       : Apply auxiliary loss to AERIAL_RGBI input if mutliple modalities. [True/False]
        AERIAL-RLT_PAN    : [True/False]
        DEM_ELEV          : [True/False]
        SPOT_RGBI         : [True/False]
        SENTINEL2_TS      : [True/False]
        SENTINEL1-ASC_TS  : [True/False]
        SENTINEL1-DESC_TS : [True/False]

    aux_loss_weight: Multiplier for auxiliary loss before combining with main loss. Default 1.
    modality_dropout: Dropout probability per modality (0 = keep all, 1 = drop all systematically).
        AERIAL_RGBI       : 0
        AERIAL-RLT_PAN    : 0


    pre_processings: 
        filter_sentinel2: Enable filtering of Sentinel-2 based on masks.
        filter_sentinel2_max_cloud : Max acceptable cloud cover in Sentinel-2 [%]. Default 1.
        filter_sentinel2_max_snow : Max acceptable snow cover in Sentinel-2 [%]. Default 1.
        filter_sentinel2_max_frac_cover : Max acceptable fractional invalid coverage in Sentinel-2. Default 0.05.
        temporal_average_sentinel2 : Temporal averaging method for Sentinel-2 (False/monthly/semi-monthly).
        temporal_average_sentinel1 : Temporal averaging method for Sentinel-1 (False/monthly/semi-monthly).
        calc_elevation : Enable calculation oF DTM-DSM.
        calc_elevation_stack_dsm : Stack DSM if elevation calculated.
        use_augmentation: Enable data augmentation for training. Default False.

    normalization: 
        norm_type : Normalization strategy (custom, scaling, without).
        AERIAL_RGBI_means : Mean values used to normalize AERIAL_RGBI input if custom.
        AERIAL_RGBI_stds  : Std. dev. values used to normalize AERIAL_RGBI input if custom.
        AERIAL-RLT_PAN_means : Mean values for AERIAL-RLT_PAN input if custom.
        AERIAL-RLT_PAN_stds  : Std. dev. values for AERIAL-RLT_PAN input if custom.
        SPOT_RGBI_means : Mean values for SPOT_RGBI input if custom.
        SPOT_RGBI_stds  : Std. dev. values for SPOT_RGBI input if custom.
        DEM_ELEV_means : Mean elevation values (DSM, DTM) if custom.
        DEM_ELEV_stds  : Std. dev. of elevation values (DSM, DTM) if custom.
```

#### config_models.yaml : <br>
```yaml
models:

    monotemp_model:  # Encoder-decoder architecture for single-date (monotemporal) inputs

        arch: Encoder-decoder architecture from SMP. Eg, swin_tiny_patch4_window7_224-upernet
        new_channels_init_mode: Strategy to initialize new input channels if above 3 channels (options: copy_first, copy_second, copy_third, random).

    multitemp_model:  # Architecture for handling multi-temporal (time series) inputs

        ref_date: Reference date (MM-DD) for temporal position encoding.
        encoder_widths: Channel sizes at each stage in the encoder; must match decoder.
        decoder_widths: Channel sizes at each stage in the decoder; must match encoder.
        out_conv: Output convolutional layer configuration [intermediate_channels, num_classes].
        str_conv_k: Kernel size for structured convolutions.
        str_conv_s: Stride for structured convolutions.
        str_conv_p: Padding for structured convolutions.
        agg_mode: Aggregation method for temporal features (e.g., att_group).
        encoder_norm: Normalization type used in the encoder (e.g., group).
        n_head: Number of attention heads in the model.
        d_model: Dimensionality of the model features.
        d_k: Key dimensionality for attention mechanism.
        pad_value: Value used for padding missing temporal data.
        padding_mode: Type of padding used in convolutions (e.g., reflect).
```

#### config_supervision.yaml : <br>
```yaml
labels: # List of tasks
  - AERIAL_LABEL-COSIA
  - ALL_LABEL-LPIS 

labels_configs:

    AERIAL_LABEL-COSIA:  
        task_weight: Weight assigned to this task during multi-task training. Default 1.
        value_name: Mapping of class indices to semantic category names.
            0  : 'building'
            1  : 'greenhouse'
            2  : 'swimming_pool'
            3  : 'impervious surface'
            4  : 'pervious surface'
            5  : 'bare soil'
            6  : 'water'
            7  : 'snow'
            8  : 'herbaceous vegetation'
            9  : 'agricultural land'
            10 : 'plowed land'
            11 : 'vineyard'
            12 : 'deciduous'
            13 : 'coniferous'
            14 : 'brushwood'
            15 : 'clear cut'
            16 : 'ligneous'
            17 : 'mixed'
            18 : 'undefined'
        value_weights:
            default: Default weight for all classes (used in loss calculation). Default 1.
            default_exceptions: Specific class indices assigned 0 weight (ignored).
                15: 0
                16: 0
                17: 0
                18: 0
            per_modality_exceptions: Placeholder for setting modality-specific class weights.
                AERIAL_RGBI:
                  18: 0

    ALL_LABEL-LPIS:  
        task_weight: Weight assigned to this task during multi-task training.
        label_channel_nomenclature: Index of the label channel used.
        value_name: Mapping of class indices to crop type names.
        value_weights:
            default: Default class weight for the entire label set.
            default_exceptions: No exception weights defined (can be added).
            per_modality_exceptions: Placeholder for modality-specific class weight overrides.
```
<br><br>


### Configuration for flair_zonal_detection :page_facing_up:

#### config_zonal_detection.yaml : <br>
```yaml
# ======================
# I/O
# ======================
output_path: Path to store output results.
output_name: Identifier name for the output files.

write_dataframe: Whether to write slicing geometries of input area. [True/False]
output_type: Format of the output predictions (argmax or class_prob).
cog_conversion: Convert outputs to Cloud-Optimized GeoTIFF format. [True/False]

# ======================
# Model & Inference
# ======================
model_weights: Path to pretrained model weights for inference. [.safetensors / .ckpt / .pth]
use_gpu: Whether to use GPU for inference.
batch_size: Number of samples per inference batch.
num_worker: Number of data loading workers during inference.

img_pixels_detection: Size of input image chunks in pixels.
margin: Overlap margin (in pixels) between tiles.
output_px_meters: Output resolution in meters per pixel.

# ======================
# Model Framework
# ======================
monotemp_arch: SMP Architecture used for monotemporal inference. Eg., swin_base_patch4_window12_384-upernet
multitemp_model_ref_date: Reference date (MM-DD) for aligning multi-temporal inputs.

# ======================
# Modalities (inputs used for inference)
# ======================
modalities:

  inputs:
      AERIAL_RGBI       : Enable AERIAL_RGBI input. [True / False]
      AERIAL-RLT_PAN    : [True / False]
      DEM_ELEV          : [True / False]
      SPOT_RGBI         : [True / False]
      SENTINEL2_TS      : [True / False]
      SENTINEL1-ASC_TS  : [True / False]
      SENTINEL1-DESC_TS : [True / False]

  AERIAL_RGBI:
      input_img_path: Path to AERIAL_RGBI input image.
      channels: Channels used from the input image. Starts at 1.
      normalization:
        type: Normalization method (e.g., custom).
        means: Per-channel mean for normalization.
        stds: Per-channel standard deviation for normalization.

  AERIAL-RLT_PAN:
      input_img_path: Path to AERIAL-RLT_PAN input.
      channels: Channels used.
      normalization:
        type: Normalization method.
        means: Mean value.
        stds: Standard deviation.

  DEM_ELEV:
      input_img_path: Path to DEM input.
      channels: Channels used.
      normalization:
        type: Normalization method.
        means: Per-channel mean.
        stds: Per-channel standard deviation.
      calc_elevation: Whether to compute elevation from DSM/DTM.
      calc_elevation_stack_dsm: Stack derived DSM with the input.

  SPOT_RGBI:
      input_img_path: Path to SPOT imagery input.
      channels: Channels used.
      normalization:
        type: Normalization method.
        means: Per-channel mean.
        stds: Per-channel standard deviation.

  SENTINEL2_TS:
      input_img_path: Path to Sentinel-2 time series input.
      channels: Channels used.
      dates_txt: Path to text file listing image dates.
      filter_clouds: Enable cloud filtering.
      filter_clouds_img_path: Path to Sentinel-2 cloud mask.
      temporal_average: Whether to average the time series over time.

  SENTINEL1-ASC_TS:
      input_img_path: Path to Sentinel-1 ascending orbit input.
      channels: Channels used.
      dates_txt: Path to date list (optional).
      temporal_average: Whether to average over time.

  SENTINEL1-DESC_TS:
      input_img_path: Path to Sentinel-1 descending orbit input.
      channels: Channels used.
      dates_txt: Path to date list (optional).
      temporal_average: Whether to average over time.

# ======================
# Tasks
# ======================
tasks:

  - name: AERIAL_LABEL-COSIA  # Semantic segmentation label task (land cover classes)
    active: Whether to activate this label set during inference. [True / False]
    class_names: Mapping of class indices to category names.
      0: building
      1: greenhouse
      2: swimming_pool
      3: impervious surface
      4: pervious surface
      5: bare soil
      6: water
      7: snow
      8: herbaceous vegetation
      9: agricultural land
      10: plowed land
      11: vineyard
      12: deciduous
      13: coniferous
      14: brushwood
      15: clear cut
      16: ligneous
      17: mixed
      18: undefined

  - name: ALL_LABEL-LPIS  # Crop type classification label task
    active: Whether to activate this label set during inference. [True / False]
    class_names: Mapping of class indices to crop categories.
```





<br><br>

## Baseline results

| Model ID | Aerial VHR | Elevation | SPOT | S2 t.s. | S1 t.s. | Historical | PARA. | O.A. | mIoU |
|----------|------------|-----------|------|---------|---------|------------|--------|------|------|
|----------|------------|-----------|------|---------|---------|------------|--------|------|------|
| LC-A     | ✓          |           |      |         |         |            |  89.4  | 77.5 | 64.1 |
| LC-B     | ✓          | ✓         |      |         |         |            | 181.4  | 78.1 | 65.1 |
| LC-C     | ✓          | ✓         | ✓    |         |         |            | 270.6  | 78.2 | 65.2 |
| LC-D     | ✓          |           |      | ✓       |         |            |  93.9  | 77.6 | 64.7 |
| LC-E     | ✓          |           |      |         | ✓       |            |  95.8  | 77.7 | 64.5 |
| LC-F     | ✓          |           |      | ✓       | ✓       |            |  97.7  | 77.7 | 64.9 |
| LC-G     |            |           |      | ✓       |         |            |   0.9  | 57.8 | 34.2 |
| LC-H     |            |           |      |         | ✓       |            |   1.8  | 54.5 | 28.2 |
| LC-I     |            |           | ✓    |         |         |            |  89.2  | 64.1 | 43.5 |
| LC-J     |            | ✓         |      |         |         |            |  89.4  | 67.4 | 51.2 |
| LC-K     | ✓          |           |      |         |         | ✓          | 181.4  | 77.6 | 64.3 |
| LC-L     | ✓          | ✓         | ✓    | ✓       | ✓       |            | 276.4  | 78.2 | 65.8 |
| LC-ALL   | ✓          | ✓         | ✓    | ✓       | ✓       | ✓          | 365.8  | 78.2 | 65.6 |
|----------|------------|-----------|------|---------|---------|------------|--------|------|------|
| LPIS-A   | ✓          |           |      |         |         |            |  89.4  | 86.6 | 24.4 |
| LPIS-B   | ✓          |           | ✓    |         |         |            | 181.2  | 87.1 | 26.1 |
| LPIS-C   | ✓          |           |      | ✓       |         |            |  93.9  | 87.5 | 29.8 |
| LPIS-D   | ✓          |           |      | ✓       | ✓       |            |  97.7  | 88.0 | 36.1 |
| LPIS-E   | ✓          |           | ✓    | ✓       |         |            | 183.1  | 87.6 | 30.3 |
| LPIS-F   |            |           |      | ✓       |         |            |   0.9  | 85.3 | 23.8 |
| LPIS-G   |            |           |      |         | ✓       |            |   1.8  | 84.5 | 18.1 |
| LPIS-H   |            |           |      | ✓       | ✓       |            |   2.8  | 84.9 | 23.8 |
| LPIS-I   |            |           | ✓    | ✓       | ✓       |            |  97.5  | 87.2 | 39.2 |
| LPIS-J   | ✓          |           | ✓    | ✓       | ✓       |            | 186.9  | 88.0 | 35.4 |
| LPIS-K   |            |           | ✓    |         |         |            |  89.2  | 84.5 | 15.1 |





## Reference
Cite the following article if you use the FLAIR #1 dataset:

```bibtex
@article{ign2025flairhub,
  doi = {10.13140/RG.2.2.30183.73128/1},
  url = {https://arxiv.org/pdf/2211.12979.pdf},
  author = {Garioud, Anatol and Giordano, Sébastien and David, Nicolas and Gonthier, Nicolas},
  title = {FLAIR-HUB: Large-scale Multimodal Dataset for Land Cover and Crop Mapping},
  publisher = {arXiv},
  year = {2025}
}
```

```
Anatol Garioud, Sébastien Giordano, Nicolas David, Nicolas Gonthier. 
FLAIR-HUB: Large-scale Multimodal Dataset for Land Cover and Crop Mapping. (2025). 
DOI: https://doi.org/10.13140/RG.2.2.30183.73128/1
```




## Acknowledgment
The experiments conducted in this study were performed using HPC/AI resources provided by GENCI-IDRIS (Grant 2024-A0161013803, 2024-AD011014286R2 and 2025-A0181013803).

## Dataset license

The "OPEN LICENCE 2.0/LICENCE OUVERTE" is a license created by the French government specifically for the purpose of facilitating the dissemination of open data by public administration. 
If you are looking for an English version of this license, you can find it on the official GitHub page at the [official github page](https://github.com/etalab/licence-ouverte).<br>
Applicable legislation : This licence is governed by French law.<br>
Compatibility of this licence : This licence has been designed to be compatible with any free licence that at least requires an acknowledgement of authorship, and specifically with the previous version of this licence as well as with the following licences: United Kingdom’s “Open Government Licence” (OGL), Creative Commons’ “Creative Commons Attribution” (CC-BY) and Open Knowledge Foundation’s “Open Data Commons Attribution” (ODC-BY).
