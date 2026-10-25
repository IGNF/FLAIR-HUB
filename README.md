<div align="center">

# FLAIR-HUB
# Multimodal & multitask semantic segmentation of Earth Observation imagery


![Static Badge](https://img.shields.io/badge/Code%3A-lightgrey?color=lightgrey) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> &emsp; ![Static Badge](https://img.shields.io/badge/Dataset%3A-lightgrey?color=lightgrey) [![license](https://img.shields.io/badge/License-IO%202.0-green.svg)](https://github.com/etalab/licence-ouverte/blob/master/open-licence.md)


Participate in obtaining more accurate maps for a more comprehensive description and a better understanding of our environment! Come push the limits of state-of-the-art semantic segmentation approaches on a large and challenging dataset. <br> Get in touch at :email: flair@ign.fr


![Alt bandeau FLAIR-HUB](imgs/flair_bandeau.jpg?raw=true)

</div>

<div style="border-width:1px; border-style:solid; border-color:#d2db8c; padding-left: 1em; padding-right: 1em; ">

<h2 style="margin-top:5px;">:mag_right:Quicklinks</h2>


- **Entry page :**  [Overall FLAIR-HUB page](https://ignf.github.io/FLAIR/FLAIR-HUB/flairhub.html) <br>

- **Datapaper :** [ISPRS Journal](https://www.sciencedirect.com/science/article/pii/S0924271626001899) / [arXiv Preprint](https://arxiv.org/abs/2506.07080)

- **Dataset links :**  [Dataset link](https://huggingface.co/datasets/IGNF/FLAIR-HUB)

- **Pre-trained models :** [Pretrained models](https://huggingface.co/collections/IGNF/flair-models-684035e78bd5bff99199ff87)

- **Toy Datatset direct downlaod** [Dataset link](https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_hub/FLAIR-HUB_TOY_DATASET.zip)

</div>
<hr>

## Context and data

This repository is built on top of the **FLAIR-HUB** dataset, the largest multi-sensor land cover dataset with very-high-resolution (20\:cm) annotations, covering 2528\:km² of France. It combines six aligned modalities: aerial imagery, Sentinel-1/2 time series, SPOT imagery, topographic data, and historical aerial images. Extensive benchmarks evaluate multimodal fusion and deep learning models (CNNs, transformers) for land cover or crop mapping and also explore multi-task learning. Key figures of the dataset are : <br>

<center>
<table style="width:95%;max-width:600px;">

  <tbody>
    <tr><td>🗺️</td><td>ROI / Area Covered</td><td style="text-align: center">2,822 ROIs / 2,528 km²</td></tr>
    <tr><td>🏛️</td><td>Departments (France)</td><td style="text-align: center">74</td></tr>
    <tr><td>🧩</td><td>AI Patches (512×512 px)</td><td style="text-align: center">241,100</td></tr>
    <tr><td>🖼️</td><td>Annotated Pixels</td><td style="text-align: center">63.2 billion</td></tr>
    <tr><td>🛰️</td><td>Sentinel-2 Acquisitions</td><td style="text-align: center">256,221</td></tr>
    <tr><td>📡</td><td>Sentinel-1 Acquisitions</td><td style="text-align: center">532,696</td></tr>
    <tr><td>📁</td><td>Total Files</td><td style="text-align: center">~2.5 million</td></tr>
    <tr><td>💾</td><td>Total Dataset Size</td><td style="text-align: center">~750 GB</td></tr>
  </tbody>
</table>
</center>

<hr>
<br><br>

## Model and training features

The model combines SMP/timm encoders with a configurable UPerNet decoder (768-channel pyramid and fusion layers in the aerial baselines) and a 256-channel auxiliary head. Training combines cross-entropy and Lovasz-Softmax, with optional online hard-example mining and per-modality auxiliary supervision. Synchronized geometric augmentation, native-value radiometry, sparse class-code mapping and full-state training recovery support single/multiple modalities and tasks.

## Installation

1. Conda env
```
$ conda create -n FLAIRHUB python=3.12 pip -y
```

2. Force torch with GPU support
```
$ conda activate FLAIRHUB
$ pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
```
3. Install flair-hub
```
$ pip install -e .
```

<hr>
<br><br>

## Entry points :mag_right:

flair-hub has two entry points:

#### :file_folder: src/flair_hub --> used to train, infer models, calculate metrics, at the patch level. <br>
```
$ flairhub --conf_folder /path/configs/train/
```

#### :file_folder: src/flair_zonal_detection --> used to infer a pretrained model over larger areas. <br>
```
$ flairhub_zonal --config /path/configs/config_zonal_detection.yaml
```


<hr>
<br><br>

## Configuration for flair-hub :page_facing_up:

The pipeline accepts the four YAML files in [`configs/train`](configs/train), or a single resolved YAML file. Paths are resolved from the working directory; adapt the CSV entries to your dataset location.

| File | Sections and purpose |
|---|---|
| `config_task.yaml` | `experiment`: CSVs and output paths; `execution`: training/prediction; `training`: duration, optimizer and scheduler; `hardware`, `checkpoints`, `logging`. |
| `config_modalities.yaml` | `data.modalities`: enabled sensors, one-based band positions, normalization and auxiliary losses; `data.preprocessing`; `data.augmentation`. |
| `config_models.yaml` | `model.monotemporal`: encoder, decoder and auxiliary head; `model.multitemporal`: temporal encoder; `model.fusion`: multimodal fusion. |
| `config_supervision.yaml` | `segmentation_tasks`: enabled tasks, source class codes, class weights and OHEM settings. |


### config_task.yaml

This file controls the experiment, execution stages, optimization and output management.

```yaml
experiment:
  output_folder: ../results/ # Parent directory for experiment outputs.
  name: test-flair-hub # Run name, used for its output directory, checkpoints and logs. Use a new name for an independent run.
  train_csv: ../toy/TRAIN_FLAIR-INC_TOY.csv # Training table containing active input paths and annotation paths.
  validation_csv: ../toy/VALID_FLAIR-INC_TOY.csv # Validation table used for monitoring and checkpoint selection.
  test_csv: ../toy/TEST_FLAIR-INC_TOY.csv # Prediction/evaluation table; annotation columns may be absent for prediction without metrics.
  metadata_folder: /FLAIR-HUB/GLOBAL_ALL_MTD/ # Global metadata directory, including acquisition dates for temporal inputs.

execution:
  train: true # Run training.
  predict: true # Run prediction on the test table after training, or from the supplied checkpoint.
  metrics_only: false # Compute metrics from saved prediction statistics without running the model.
  write_prediction_files: false # Write patch prediction rasters in addition to evaluation statistics.
  georeference_predictions: true # Preserve spatial metadata in prediction rasters.

training:
  duration:
    unit: steps # Interpret the training budget as steps or epochs.
    total: 90000 # Total training budget, including work already completed when resuming.
  validation:
    every: 7000 # Validation interval in the selected duration unit.
  batch_size: 2 # Batch size per GPU/process.
  seed: 2058167048 # Random seed used to initialize the run.
  optimizer:
    name: adamw # Optimizer: adamw, adam or sgd.
    learning_rate: 6.0e-05 # Configured learning rate, before scheduler adjustments.
    weight_decay: 0.01 # Optimizer weight decay.
    betas: [0.9, 0.999] # Adam/AdamW first- and second-moment coefficients.
  scheduler:
    name: cosine # cosine, one_cycle_lr, reduce_on_plateau, cycle_then_plateau or none.
    warmup_steps: 9000 # Warmup length in optimizer steps for cosine, one-cycle and composite schedules.
    warmup_ratio: 1.0e-06 # Initial/configured learning-rate ratio for cosine warmup.
    minimum_lr_ratio: 1.0e-06 # Final/configured learning-rate ratio for cosine decay.
    plateau_patience: 10 # Number of plateau scheduler checks tolerated without improvement.

hardware:
  accelerator: gpu # Execution device selection: gpu, cpu or auto.
  precision: 16-mixed # Numerical precision, e.g. 16-mixed, bf16-mixed or 32-true; select a mode supported by the hardware.
  num_nodes: 1 # Number of distributed compute nodes.
  gpus_per_node: 1 # Number of GPUs per node.
  num_workers: 8 # Data-loading workers per process; zero loads data in the main process.
  pin_memory: auto # Optional override; defaults to automatic selection from the execution device.
  strategy: auto # Optional Lightning strategy override; selected automatically by default.

checkpoints:
  initialize_from: null # Optional checkpoint/weights path for initializing a new run.
  reset_task_heads: false # Optional explicit reset of task output projections during initialization.
  resume_from: null # null, a full checkpoint path, or auto to search the current run's recovery checkpoint.
  best:
    monitor: val_miou # Validation metric used to select the best checkpoint, usually val_miou.
    mode: max # Maximize (max) or minimize (min) the monitored metric.
    weights_only: false # Save weights only; use false for a checkpoint that can restore full training state.
    also_save_last: false # Also save the latest checkpoint through the best-checkpoint callback.
  recovery:
    enabled: true # Enable periodic full-state recovery checkpoints.
    every_steps: 2000 # Recovery checkpoint interval in optimizer steps.
    directory: recovery_checkpoints # Recovery directory relative to the run output directory.
  early_stopping:
    enabled: false # Enable early stopping based on the monitored validation metric.
    patience: 20 # Number of validation checks without improvement before stopping.

logging:
  progress: true # Display training/prediction progress.
  checkpoint_messages: true # Display checkpoint callback messages.
  verbose_config: false # Print the detailed configuration.
  copy_input_csv: true # Copy the input CSV tables into the experiment output directory.
```

### config_modalities.yaml

`data.modalities` contains one entry per sensor: `AERIAL_RGBI`, `AERIAL-RLT_PAN`, `DEM_ELEV`, `SPOT_RGBI`, `SENTINEL2_TS`, `SENTINEL1-ASC_TS` and `SENTINEL1-DESC_TS`. The following fields apply to each entry.

```yaml
data:
  modalities:
    AERIAL_RGBI:
      enabled: true # Include this modality in the experiment.
      input_band_order: [4, 2, 1] # Ordered source band positions, starting at 1. For temporal data, positions refer to bands within each date. DEM defaults to [1, 2].
      normalization:
        type: custom # custom for per-channel standardization, scaling for an affine transformation, or without.
        mean: [123.675, 116.28, 103.53] # Per-channel statistics in the selected input order for custom normalization.
        std: [58.395, 57.12, 57.375] # Per-channel statistics in the selected input order for custom normalization.
        scale: 0.00392156862745098 # Optional scaling parameters: float32(image) * scale + offset. Aerial scale defaults to 1/255; other sensors require an explicit scale when selecting scaling.
        offset: 0.0 # Optional scaling parameters: float32(image) * scale + offset. Aerial scale defaults to 1/255; other sensors require an explicit scale when selecting scaling.
      auxiliary_loss:
        enabled: true # Add supervision on this modality's auxiliary output during training.
        weight: 0.4 # Contribution of the auxiliary loss.
      dropout_probability: 0.0 # Probability of dropping this modality's contribution during training.
```

Preprocessing is configured under `data.preprocessing`:

```yaml
data:
  preprocessing:
    sentinel2:
      filter_clouds: true # Filter acquisitions using the cloud/snow masks.
      maximum_cloud: 1 # Maximum accepted cloud-mask value at a pixel (0–100).
      maximum_snow: 1 # Maximum accepted snow-mask value at a pixel (0–100).
      maximum_covered_fraction: 0.05 # Maximum permitted fraction of pixels failing the cloud/snow thresholds for an acquisition.
      temporal_average: false # false, monthly or semi-monthly aggregation of the retained acquisitions.
    sentinel1:
      temporal_average: false # Same temporal aggregation choices for Sentinel-1.
    elevation:
      derive_elevation: true # Derive relative elevation from the selected DSM and DTM source bands.
      stack_with_dsm: true # When deriving elevation, retain DSM alongside the relative elevation channel.
```

`data.augmentation.enabled` switches training augmentation on or off. Geometry is configured under `data.augmentation.geometry`:

```yaml
data:
  augmentation:
    geometry:
      scale_range: [0.5, 2.0] # Minimum and maximum random spatial scale factors.
      cropping_reference_task: AERIAL_LABEL-COSIA # Annotation used to evaluate the dominant-class constraint during random cropping. Automatically set when only one task is active.
      category_max_ratio: 0.75 # Maximum desired fraction of one class in the sampled crop; the sampler uses a bounded number of attempts.
      horizontal_flip_probability: 0.0 # Probability of horizontal reflection.
      vertical_flip_probability: 0.0 # Probability of vertical reflection.
      rotation_probability: 0.5 # Probability of applying rotation.
      rotation_degrees: [-90.0, 90.0] # Minimum and maximum rotation angle in degrees.
```

If cloud filtering rejects every date, the implementation retries using only the snow criterion. If no date remains, a masked zero frame is supplied.

Radiometry is configured separately under `data.augmentation.radiometry.<modality>`:

```yaml
data:
  augmentation:
    radiometry:
      AERIAL_RGBI:
        enabled: true # Enable radiometric augmentation; automatically set to false if the modality is inactive.
        mode: hsv # hsv for three channels with values in [0, 255], or gain_offset for native sensor values. Defaults to HSV for three-band aerial input and gain/offset otherwise.
        brightness_delta: 32 # HSV-mode brightness perturbation amplitude.
        contrast_range: [0.5, 1.5] # HSV-mode contrast multiplier range.
        saturation_range: [0.5, 1.5] # HSV-mode saturation multiplier range.
        hue_delta: 18 # HSV-mode hue perturbation amplitude.
        gain_range: [0.9, 1.1] # Gain/offset-mode per-band multiplier range.
        offset_range: [0.0, 0.0] # Gain/offset-mode additive range in native sensor units.
```

Geometry is synchronized across inputs and labels. Radiometry never modifies labels; gain/offset draws are shared across dates of a temporal series. Normalization follows augmentation.

### config_models.yaml

```yaml
model:
  monotemporal:
    encoder:
      name: swin_large_patch4_window12_384.ms_in22k # SMP/timm encoder identifier.
      weights: imagenet # imagenet for pretrained initialization, or null for random initialization.
      drop_path_rate: 0.3 # Encoder stochastic-depth rate, applied by the flair_hub decoder construction path; ignored by the current smp path.
    decoder:
      name: upernet # Decoder architecture, e.g. upernet, unet, unetplusplus or deeplabv3plus.
      provider: flair_hub # flair_hub for the configurable UPerNet implementation, or smp for an SMP decoder.
      upernet_flair_hub_params:
        channels: 768 # Width of the UPerNet pooling/fusion layers.
        pool_scales: [1, 2, 3, 6] # Spatial pyramid pooling output scales.
        dropout: 0.1 # Decoder dropout probability.
        norm: sync_batchnorm # sync_batchnorm, batchnorm or groupnorm.
        align_corners: false # Corner alignment used for bilinear feature interpolation.
    auxiliary_head:
      channels: 256 # Width of the mono-temporal auxiliary heads.
      dropout: 0.1 # Auxiliary head dropout probability.
      norm: sync_batchnorm # Auxiliary head normalization, with the same choices as the decoder.
```

Auxiliary heads are used only for modalities whose auxiliary loss is enabled. Mono-temporal auxiliary heads are available with either decoder provider.

The temporal model is configured under `model.multitemporal`:

```yaml
model:
  multitemporal:
    ref_date: 05-15 # Month-day reference (MM-DD) for acquisition-date offsets.
    encoder_widths: [64, 64, 64, 128] # Channel widths of successive temporal encoder and decoder stages.
    decoder_widths: [32, 32, 64, 128] # Channel widths of successive temporal encoder and decoder stages.
    head_hidden_channels: [32, 19] # Hidden output-head widths; the final output count is derived from active tasks and classes.
    str_conv_k: 3 # Kernel size, stride and padding of the spatial downsampling/upsampling blocks.
    str_conv_s: 1 # Kernel size, stride and padding of the spatial downsampling/upsampling blocks.
    str_conv_p: 1 # Kernel size, stride and padding of the spatial downsampling/upsampling blocks.
    agg_mode: att_group # Temporal aggregation: att_group, att_mean or mean.
    encoder_norm: group # Encoder normalization: group, batch or instance.
    n_head: 16 # Number of temporal attention heads.
    d_model: 256 # Attention embedding dimension.
    d_k: 4 # Key/query dimension per attention head.
    pad_value: 0 # Value identifying padded temporal frames.
    padding_mode: reflect # Spatial convolution padding: reflect, zeros, replicate or circular.
```

The optional `model.fusion` block controls multimodal fusion:

```yaml
model:
  fusion:
    feature_source: encoder # Fuse encoder or decoder features; defaults to encoder.
    method: gated_residual # aligned_concat, gated_residual or window_cross_attention; defaults to gated_residual.
    levels: [1, 2, 3] # Zero-based pyramid levels involved in fusion; defaults to [1, 2, 3].
    gate_init: -2.0 # Initial residual gate parameter; defaults to -2.0.
    attention_channels: 128 # Feature dimension for attention fusion; defaults to 128.
    attention_heads: 4 # Number of spatial attention heads; defaults to 4. Must divide attention_channels.
    window_size: 8 # Spatial attention window size; defaults to 8.
```

### config_supervision.yaml

Each entry under `segmentation_tasks` describes a task, such as `AERIAL_LABEL-COSIA` or `ALL_LABEL-LPIS`.

```yaml
segmentation_tasks:
  AERIAL_LABEL-COSIA:
    enabled: true # Include the segmentation task.
    task_weight: 1.0 # Weight applied to the task's training loss. This does not change the equal-task averaging of validation mIoU.
    label_band: 1 # One-based annotation band to read.
    # Mapping from original raster codes to class names. Codes may contain gaps.
    classes:
      0: building
      1: greenhouse
      2: swimming pool
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
    class_weights:
      default: 1.0 # Default class weight.
      # Per-code overrides; a zero weight excludes that class from model outputs and scoring.
      exceptions:
        15: 0.0
        16: 0.0
        17: 0.0
        18: 0.0
      # Per-modality class-weight overrides for auxiliary supervision.
      per_modality_exceptions: {}
    loss:
      ohem:
        enabled: true # Apply online hard-example mining to cross-entropy.
        threshold: 0.7 # Ground-truth probability threshold used for hard-pixel selection.
        minimum_pixels: 100000 # Minimum-pixel setting used to derive the adaptive OHEM threshold.
    nomenclature_version: null # Optional semantic version marker saved in checkpoint metadata alongside active codes and names.
```

Main and auxiliary losses combine cross-entropy with Lovasz-Softmax. Pixels with unknown codes or excluded classes are ignored. Training and validation accept partial task annotations: each row must have at least one annotation, and every active task must have annotations in each split. Prediction can run without labels.

Enable one or several modalities and segmentation tasks independently. A class with weight zero is excluded from supervision and model outputs. Source codes may be non-consecutive: the model uses internal consecutive indices and predictions restore the source codes. Unknown label codes are ignored. The reported mIoU is a macro average over active classes; multi-task checkpoint selection averages task mIoUs equally.

`checkpoints.initialize_from` initializes a new run from pretrained weights; `checkpoints.resume_from` restores training state (`auto` searches the current run's recovery checkpoint). A changed nomenclature reinitializes the affected output projections during fine-tuning. `reset_task_heads: true` can explicitly reset them. Use a new experiment name for an independent run.

`training.duration.total` is the total budget in steps or epochs, not an additional budget on resume. Batch size is per GPU. Pin memory and distributed strategy are selected automatically unless overridden.

Geometric augmentation is shared across modalities and tasks. Radiometric augmentation is enabled per modality; inactive modalities are automatically disabled. Three-band aerial imagery uses HSV on values in [0, 255]; SPOT and other native sensor values use `gain_offset`. Normalization follows augmentation.

### Configuration for flair_zonal_detection :page_facing_up:

See [`configs/config_zonal_detection.yaml`](configs/config_zonal_detection.yaml):


```yaml
training_artifacts:
  resolved_config: /path/to/run/resolved_config.yaml # Saved training configuration defining model, inputs, preprocessing and tasks.
  checkpoint: /path/to/run/checkpoints/best.ckpt # Trained weights matching that configuration.

inputs:
  AERIAL_RGBI:
    raster: /path/to/aerial_mosaic.tif # Input mosaic or date-major temporal raster for an active modality.
  SENTINEL2_TS: # Include only if this modality was active during training.
    raster: /path/to/sentinel2_timeseries.tif
    dates_file: /path/to/dates.txt # Temporal acquisition dates, one YYYYMMDD date per line.
    cloud_mask_raster: /path/to/cloud_masks.tif # Sentinel-2 cloud/snow masks when cloud filtering is enabled.

inference:
  use_gpu: true # Request GPU execution when available.
  precision: auto # auto inherits training precision; explicit choices include 32-true, 16-mixed and bf16-mixed.
  batch_size: 1 # Number of windows predicted together.
  num_workers: 0 # Number of loading workers; zero is suitable for Windows.
  patch_size: 2048 # Width/height of the model input window in pixels of the finest active input grid.
  margin: 512 # Border width removed from each prediction window before writing its central area; must be less than half the patch size.
  output_resolution: 0.2 # Output pixel size in metres.

output:
  folder: ./results/zonal # Destination directory.
  name: aerial_prediction # Output name prefix.
  type: argmax # argmax for original class codes, or class_prob for per-class probabilities encoded as uint8.
  cloud_optimized_geotiff: true # Convert output rasters to cloud-optimized GeoTIFFs.
  write_slicing_grid: false # Export the window grid for inspection.
```

Normalization, band order and task definitions are inherited from the training configuration. Supply only active input modalities. Temporal inputs also require a `dates_file`, with one YYYYMMDD acquisition per line; Sentinel-2 cloud filtering additionally requires `cloud_mask_raster`. Temporal rasters store all bands of each date consecutively. The number of bands per date is inferred from raster bands and dates. Input grids must share a projected CRS in metres, with square north-up pixels.

<hr>

## Baseline results

COSIA aerial IRC experiments on SET1–SET5, using Swin-Large and the UPerNet decoder with 768 channels. Results below are the recorded test metrics, in percent; each experiment uses its own split. They are not five repeated runs on the same test set.

| Model | O.A. (%) | mIoU (%) | F-score (%) |
|---|---:|---:|---:|
| FLAIR-HUB_V2_COSIA_AERIAL_IRC_SET1 | 78.277 | 65.690 | 78.137 |
| FLAIR-HUB_V2_COSIA_AERIAL_IRC_SET2 | 80.457 | 68.728 | 80.570 |
| FLAIR-HUB_V2_COSIA_AERIAL_IRC_SET3 | 79.539 | 68.279 | 80.314 |
| FLAIR-HUB_V2_COSIA_AERIAL_IRC_SET4 | 80.760 | 69.800 | 81.316 |
| FLAIR-HUB_V2_COSIA_AERIAL_IRC_SET5 | 79.983 | 67.072 | 79.386 |
| **Mean across splits** | **79.803** | **67.914** | **79.945** |

mIoU is macro-averaged over the 15 active COSIA classes (codes 0–14); codes 15–18 are excluded. The summary is an unweighted mean of the five split scores, not a pooled confusion-matrix result. Training uses 90,000 steps, 18,000 warmup steps, AdamW at 6e-5 and OHEM with a 50,000-pixel minimum.


## Inference classes and colors

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


## 🔎 Filter dataset with the FLAIR-HUB Dataset Browser

A small desktop GUI to browse and download subsets
of the IGNF/FLAIR-HUB dataset from Hugging Face with filters for: Domain, Year, Modality or Data type.

Requirements:

- Python 3.9+
- Tkinter (usually included; on Linux you may need: sudo apt-get install python3-tk)
- Python packages: pip install `huggingface_hub`

Run:
1. Get the `flair-hub-HF-dl.py` file.
2. In a terminal: ```pip install huggingface_hub```
3. Launch: ```python flair-hub-HF-dl.py```

<br><br>


## Reference
Cite the following article if you use the FLAIR #1 dataset:

```bibtex
@article{GARIOUD2026271,
title = {FLAIR-HUB: Large-scale multimodal dataset for land cover and crop mapping},
author = {Anatol Garioud and Sébastien Giordano and Nicolas David and Nicolas Gonthier},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {237},
pages = {271-300},
year = {2026},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2026.04.017},
url = {https://www.sciencedirect.com/science/article/pii/S0924271626001899},
}
```

```
Anatol Garioud, Sébastien Giordano, Nicolas David, Nicolas Gonthier.
FLAIR-HUB: Large-scale Multimodal Dataset for Land Cover and Crop Mapping.
ISPRS Journal of Photogrammetry and Remote Sensing, Volume 237, 2026.
DOI: https://doi.org/10.1016/j.isprsjprs.2026.04.017
```


## Acknowledgment
The experiments conducted in this study were performed using HPC/AI resources provided by GENCI-IDRIS (Grant 2024-A0161013803, 2024-AD011014286R2 and 2025-A0181013803).

## Dataset license

The "OPEN LICENCE 2.0/LICENCE OUVERTE" is a license created by the French government specifically for the purpose of facilitating the dissemination of open data by public administration.
If you are looking for an English version of this license, you can find it on the [official licence page](https://github.com/etalab/licence-ouverte).<br>
Applicable legislation : This licence is governed by French law.<br>
Compatibility of this licence : This licence has been designed to be compatible with any free licence that at least requires an acknowledgement of authorship, and specifically with the previous version of this licence as well as with the following licences: United Kingdom’s “Open Government Licence” (OGL), Creative Commons’ “Creative Commons Attribution” (CC-BY) and Open Knowledge Foundation’s “Open Data Commons Attribution” (ODC-BY).
