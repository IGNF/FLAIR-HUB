import torch
import torch.nn as nn

from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from flair_inc.models.monotemp_model import FLAIR_MonoTemporal
from flair_inc.models.multitemp_model import UTAE



class FusionFitter(nn.Module):
    """
    Aligns input feature maps to target feature maps in terms of number, channels, and spatial dimensions.
    Specifically, it adjusts the number of feature maps, the number of channels in each map, 
    and the spatial dimensions (height, width) of each map to match the target shapes.
    """

    def __init__(self):
        """
        Initialize the FusionFitter module.
        """
        super(FusionFitter, self).__init__()


    def _align_nb_featmaps(self, input_maps, target_shapes):
        """
        Align the number of input feature maps to match the target shapes by averaging or repeating.
        Args:
            input_maps (list of torch.Tensor): List of input feature maps.
            target_shapes (list of tuple): List of target shapes, where each shape is 
                                           a tuple (C, H, W) for each feature map.
        Returns:
            list of torch.Tensor: List of aligned input feature maps.
        """
        input_count = len(input_maps)
        target_count = len(target_shapes)

        if input_count > target_count:
            # Reduce input feature maps by averaging
            factor = input_count // target_count
            reduced_maps = [
                torch.stack(input_maps[i * factor:(i + 1) * factor]).mean(dim=0)
                for i in range(target_count)
            ]
        elif input_count < target_count:
            # Repeat input maps to match the target count
            repeats = target_count // input_count
            extras = target_count % input_count
            reduced_maps = input_maps * repeats + input_maps[:extras]
        else:
            reduced_maps = input_maps

        return reduced_maps


    def _align_channels(self, fmap, target_channels):
        """
        Align the number of channels in the feature map to the target channels using convolution.
        Args:
            fmap (torch.Tensor): Input feature map with shape (B, C_in, H, W).
            target_channels (int): The target number of channels.
        Returns:
            torch.Tensor: Feature map with aligned channels.
        """
        input_channels = fmap.shape[1]
        if input_channels != target_channels:
            # Convolution to match the target number of channels
            conv = nn.Conv2d(input_channels, target_channels, kernel_size=1).to(fmap.device)
            fmap = conv(fmap)
        return fmap


    def _align_spatial_dims(self, fmap, target_shape):
        """
        Align the spatial dimensions (H, W) of the feature map to the target shape using interpolation.
        Args:
            fmap (torch.Tensor): Input feature map with shape (B, C, H, W).
            target_shape (tuple): Target shape as (C, H_target, W_target).
        Returns:
            torch.Tensor: Feature map with aligned spatial dimensions.
        """
        target_h, target_w = target_shape[2], target_shape[3]
        if fmap.shape[2] != target_h or fmap.shape[3] != target_w:
            fmap = F.interpolate(fmap, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return fmap


    def forward(self, input_maps, target_shapes):
        """
        Align input feature maps to target shapes (number, channels, spatial dimensions).
        Args:
            input_maps (list of torch.Tensor): List of input feature maps to be aligned.
            target_shapes (list of tuple): List of target shapes for each feature map 
                                           in the format (C, H, W).
        Returns:
            list of torch.Tensor: List of aligned feature maps.
        """
        # Align the number of feature maps
        input_maps = self._align_nb_featmaps(input_maps, target_shapes)

        aligned_feature_maps = []
        for fmap, target_shape in zip(input_maps, target_shapes):
            # Align the number of channels
            fmap = self._align_channels(fmap, target_shape[1])  
            # Align the spatial dimensions
            fmap = self._align_spatial_dims(fmap, target_shape)
            aligned_feature_maps.append(fmap)

        return aligned_feature_maps

        
    
    
class FLAIR_TimeTexture(nn.Module):
    """
    Main model. Wrapper for monotemporal and multitemporal models and the fusion module.
    """

    def __init__(self, config):
        super(FLAIR_TimeTexture, self).__init__()

        arch = config['models']['monotemp_model']['arch'].split('_')[0]
        self.arch = arch if arch in ['smp', 'hf'] else None
        self.fuse_operation = config['models']['fusion_operation']
        n_classes = len(config['classes'])

        ########## MONOTEMPORAL MODELS
        self.first_monotemp_initialized = False
        self.monotemp_model_aerial = None
        self.monotemp_model_pan = None
        self.monotemp_model_elev = None
        self.monotemp_model_spot = None
        self.model_keys = {}

        if config['modalities']['inputs']['AERIAL_RGBI']:
            self.monotemp_model_aerial = FLAIR_MonoTemporal(
                config,
                channels=len(config['modalities']['inputs_channels']['aerial']),
                classes=n_classes,
                return_backbone_only=self.first_monotemp_initialized
            )
            if not self.first_monotemp_initialized:
                self.model_keys['AERIAL_RGBI'] = 'full'
                self.first_monotemp_initialized = True
            else:
                self.model_keys['AERIAL_RGBI'] = 'backbone'

        if config['modalities']['inputs']['AERIAL-RLT_PAN']:
            self.monotemp_model_pan = FLAIR_MonoTemporal(
                config,
                channels=1,
                classes=n_classes,
                return_backbone_only=self.first_monotemp_initialized
            )
            if not self.first_monotemp_initialized:
                self.model_keys['AERIAL-RLT_PAN'] = 'full'
                self.first_monotemp_initialized = True
            else:
                self.model_keys['AERIAL-RLT_PAN'] = 'backbone'

        if config['modalities']['inputs']['DEM_ELEV']:
            self.monotemp_model_elev = FLAIR_MonoTemporal(
                config,
                channels=1,
                classes=n_classes,
                return_backbone_only=self.first_monotemp_initialized
            )
            if not self.first_monotemp_initialized:
                self.model_keys['DEM_ELEV'] = 'full'
                self.first_monotemp_initialized = True
            else:
                self.model_keys['DEM_ELEV'] = 'backbone'

        if config['modalities']['inputs']['SPOT_RGBI']:
            self.monotemp_model_spot = FLAIR_MonoTemporal(
                config,
                channels=len(config['modalities']['inputs_channels']['spot']),
                classes=n_classes,
                return_backbone_only=self.first_monotemp_initialized
            )
            if not self.first_monotemp_initialized:
                self.model_keys['SPOT_RGBI'] = 'full'
                self.first_monotemp_initialized = True
            else:
                self.model_keys['SPOT_RGBI'] = 'backbone'

        ########## MULTITEMPORAL MODELS
        self.multitemp_model_sen2 = None
        self.multitemp_model_sen1asc = None
        self.multitemp_model_sen1desc = None

        if config['modalities']['inputs']['SENTINEL2_TS']:
            self.multitemp_model_sen2 = UTAE(
                config,
                input_dim=len(config['modalities']['inputs_channels']['sentinel2']),
                return_maps=True
            )
        if config['modalities']['inputs']['SENTINEL1-ASC_TS']:
            self.multitemp_model_sen1asc = UTAE(
                config,
                input_dim=len(config['modalities']['inputs_channels']['sentinel1']),
                return_maps=True
            )
        if config['modalities']['inputs']['SENTINEL1-DESC_TS']:
            self.multitemp_model_sen1desc = UTAE(
                config,
                input_dim=len(config['modalities']['inputs_channels']['sentinel1']),
                return_maps=True
            )

        self.print_model_parameters()

        ########## MULTITEMPORAL AUX LOSS RESHAPING
        if any([self.multitemp_model_sen2, self.multitemp_model_sen1asc, self.multitemp_model_sen1desc]):
            self.reshape_to_labels = nn.Sequential(
                nn.Upsample(size=(512, 512), mode='nearest'),
                nn.Conv2d(32, n_classes, 1)  # ######### CHECK THIS !!
            )

        ########## METADATA
        if config['modalities']['use_metadata']:
            i = 512
            last_spatial_dim = int([(i := i / 2) for u in range(len(self.arch_vhr.seg_model.encoder.out_channels) - 1)][-1])
            # self.mtd_mlp = MetadataEncodingMLP(config['geo_enc_size'] + 13, last_spatial_dim)  ######### CHECK THIS !!

        ########## FUSION MODULE
        self.fusion_fitter = FusionFitter()

  
    @rank_zero_only
    def print_model_parameters(self):
        """
        Prints the total number of parameters in the model, broken down by each model type.
        """
        models = {
            'AERIAL_RGBI': self.monotemp_model_aerial,
            'AERIAL-RLT_PAN': self.monotemp_model_pan,
            'DEM_ELEV': self.monotemp_model_elev,
            'SPOT_RGBI': self.monotemp_model_spot,
            'SENTINEL2_TS': self.multitemp_model_sen2,
            'SENTINEL1-ASC_TS': self.multitemp_model_sen1asc,
            'SENTINEL1-DESC_TS': self.multitemp_model_sen1desc
        }

        total_params = 0
        table = f"| {'Model Key':<20} | {'Type':<10} | {'Parameters':<15} |\n"
        table += f"| {'-'*20} | {'-'*10} | {'-'*15} |\n"

        for key, model in models.items():
            if model is not None:
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                model_type = self.model_keys.get(key, 'full')
                table += f"| {key:<20} | {model_type:<10} | {num_params:<15} |\n"
        
        table += f"| {'**TOTAL PARAMS**':<20} | {'':<10} | {total_params:<15} |\n"
        print(table)


    def process_monotemp_model(self, model, key, architecture, model_keys, batch):
        """
        Processes a monotemporal model based on the given key, architecture, and batch data.
        """
        if model:
            if model_keys.get(key) == 'full':
                if architecture == 'hf':
                    return list(model.seg_model.backbone(batch[key]).feature_maps)
                elif architecture == 'smp':
                    return list(model.seg_model.encoder(batch[key]))
            else:
                if architecture == 'hf':
                    return list(model.seg_model(batch[key]).feature_maps)
                elif architecture == 'smp':
                    return list(model.seg_model(batch[key]))

        return None


    def decode_model(self, model, fused_feature, architecture):
        """
        Decodes the fused feature using the given model and architecture.
        """
        if architecture == 'hf':
            return model.seg_model.decode_head(fused_feature)
        elif architecture == 'smp':
            out_map = model.seg_model.decoder(*fused_feature)
            return model.seg_model.segmentation_head(out_map)
        
        return None
 
    
    def forward(self, batch):
        """
        Forward pass for FLAIR_TimeTexture.
        Processes monotemporal and multitemporal feature maps, aligns them, 
        and fuses them using the specified operation before decoding.
        """
        
        feature_maps = { 
            'monotemporal': {},
            'multitemporal': {}               
        }
        
        base_feature_shape = None  # To store the base shape for alignment

        # Process monotemporal data
        if self.monotemp_model_aerial:
            feature_maps['monotemporal']['AERIAL_RGBI'] = self.process_monotemp_model(
                self.monotemp_model_aerial, 
                'AERIAL_RGBI', 
                self.arch, 
                self.model_keys, 
                batch,
            )
            
        if self.monotemp_model_spot:
            feature_maps['monotemporal']['SPOT_RGBI'] = self.process_monotemp_model(
                self.monotemp_model_spot, 
                'SPOT_RGBI', 
                self.arch, 
                self.model_keys, 
                batch,
            )
            
        if self.monotemp_model_pan:
            feature_maps['monotemporal']['AERIAL-RLT_PAN'] = self.process_monotemp_model(
                self.monotemp_model_pan, 
                'AERIAL-RLT_PAN', 
                self.arch, 
                self.model_keys, 
                batch,
            )
            
        if self.monotemp_model_elev:
            feature_maps['monotemporal']['DEM_ELEV'] = self.process_monotemp_model(
                self.monotemp_model_elev, 
                'DEM_ELEV', 
                self.arch, 
                self.model_keys, 
                batch,
            )        

        # Initialize base_feature_shape with the first non-None feature map
        if base_feature_shape is None:
            for modality in ['AERIAL_RGBI', 'SPOT_RGBI', 'AERIAL-RLT_PAN', 'DEM_ELEV']:
                if modality in feature_maps['monotemporal']:
                    base_feature_shape = [fm.shape for fm in feature_maps['monotemporal'][modality]]
                    break

        # Process multitemporal data
        if self.multitemp_model_sen2:
            _, fmaps_sen2 = self.multitemp_model_sen2(
                batch['SENTINEL2_TS'], 
                batch_positions=batch['SENTINEL2_TS_DATES']
            )
            feature_maps['multitemporal']['SENTINEL2_TS'] = fmaps_sen2
            out_aux_sen2 = self.reshape_to_labels(fmaps_sen2[-1])

            if base_feature_shape is None and 'SENTINEL2_TS' in feature_maps['multitemporal']:
                base_feature_shape = [fm.shape for fm in fmaps_sen2]            

        if self.multitemp_model_sen1asc:
            _, fmaps_sen1asc = self.multitemp_model_sen1asc(
                batch['SENTINEL1-ASC_TS'], 
                batch_positions=batch['SENTINEL1-ASC_DATES']
            )
            feature_maps['multitemporal']['SENTINEL1-ASC_TS'] = fmaps_sen1asc
            out_aux_sen1asc = self.reshape_to_labels(fmaps_sen1asc[-1])   

            if base_feature_shape is None and 'SENTINEL1-ASC_TS' in feature_maps['multitemporal']:
                base_feature_shape = [fm.shape for fm in fmaps_sen1asc]

        if self.multitemp_model_sen1desc:
            _, fmaps_sen1desc = self.multitemp_model_sen1desc(
                batch['SENTINEL1-DESC_TS'], 
                batch_positions=batch['SENTINEL1-DESC_DATES']
            )
            feature_maps['multitemporal']['SENTINEL1-DESC_TS'] = fmaps_sen1desc
            out_aux_sen1desc = self.reshape_to_labels(fmaps_sen1desc[-1]) 

            if base_feature_shape is None and 'SENTINEL1-DESC_TS' in feature_maps['multitemporal']:
                base_feature_shape = [fm.shape for fm in fmaps_sen1desc]
        
        # Align feature maps using FusionFitter
        for group_key in feature_maps:
            for key in feature_maps[group_key]:             
                curr_shapes = [i.shape for i in feature_maps[group_key][key]]
                if curr_shapes != base_feature_shape:
                    feature_maps[group_key][key] = self.fusion_fitter(
                        feature_maps[group_key][key], 
                        base_feature_shape
                    )

        # Fuse feature maps of different modalities                    
        to_fuse = []
        for group_key in feature_maps:
            for key in feature_maps[group_key]:
                to_fuse.append(feature_maps[group_key][key])

        fused_feature = []
        for u in range(len(to_fuse[0])):
            fmaps_stage = [i[u] for i in to_fuse]
            stacked_fmaps = torch.stack(fmaps_stage, dim=0) 
            if self.fuse_operation == 'add':
                fused_tensor = torch.sum(stacked_fmaps, dim=0)
            elif self.fuse_operation == 'average':
                fused_tensor = torch.mean(stacked_fmaps, dim=0)
            elif self.fuse_operation == 'multiply':
                fused_tensor = torch.prod(stacked_fmaps, dim=0)
            else:
                raise ValueError(f"Unknown fuse operation: {self.fuse_operation}")
            fused_feature.append(fused_tensor)            

        # Decode the fused feature map using the appropriate model
        out_map = None
        if self.monotemp_model_aerial and self.model_keys['AERIAL_RGBI'] == 'full':
            out_map = self.decode_model(self.monotemp_model_aerial, fused_feature, self.arch)
        elif self.monotemp_model_pan and self.model_keys['AERIAL-RLT_PAN'] == 'full':
            out_map = self.decode_model(self.monotemp_model_pan, fused_feature, self.arch)
        elif self.monotemp_model_elev and self.model_keys['DEM_ELEV'] == 'full':
            out_map = self.decode_model(self.monotemp_model_elev, fused_feature, self.arch)
        elif self.monotemp_model_spot and self.model_keys['SPOT_RGBI'] == 'full':
            out_map = self.decode_model(self.monotemp_model_spot, fused_feature, self.arch)

        # Interpolate the output to the desired size if using 'hf' architecture
        if self.arch == 'hf':    
            logits = nn.functional.interpolate(out_map, size=512, mode="bilinear", align_corners=False) 
        else:
            logits = out_map

        return logits


