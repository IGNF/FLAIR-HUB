import numpy as np
import math
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

    def __init__(self, backbone_total_channels, decoder_target_channels):
        """
        Initialize the FusionFitter module.
        """
        super(FusionFitter, self).__init__()
        self.conv1x1_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            for in_ch, out_ch in zip(backbone_total_channels, decoder_target_channels)
        ])

    def _align_spatial_dims(self, fmap, target_shape):
        """
        Align the spatial dimensions (H, W) of the feature map to the target shape using interpolation.
        Args:
            fmap (torch.Tensor): Input feature map with shape (B, C, H, W).
            target_shape (tuple): Target shape as (B, C, H_target, W_target).
        Returns:
            torch.Tensor: Feature map with aligned spatial dimensions.
        """
        target_h, target_w = target_shape[2], target_shape[3]
        if fmap.shape[2] != target_h or fmap.shape[3] != target_w:
            fmap = F.interpolate(fmap, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return fmap

    def forward(self, feature_maps, base_feature_shape):
        """
        Align input feature maps to target shapes (number, channels, spatial dimensions).
        Args:
            feature_maps (dict): Dictionary of feature maps to be aligned.
            base_feature_shape (list): List of target shapes for each feature map in the format (B, C, H, W).
        Returns:
            list of torch.Tensor: List of aligned feature maps.
        """

        aligned_feature_maps_list = []
        for key in feature_maps:
            aligned_feature_maps = [self._align_spatial_dims(fmap, target_shape) for fmap, target_shape in zip(feature_maps[key], base_feature_shape)]
            aligned_feature_maps_list.append(aligned_feature_maps)

        # Stack the feature maps along the channel dimension
        stacked_feature_maps = [torch.cat(fmaps, dim=1) for fmaps in zip(*aligned_feature_maps_list)]

        # Use Conv2d layers to adjust the channel sizes
        adjusted_feature_maps = [conv1x1(fmap) for conv1x1, fmap in zip(self.conv1x1_layers, stacked_feature_maps)]

        return adjusted_feature_maps

    




class FLAIR_TimeTexture(nn.Module):
    """
    Main model. Wrapper for monotemporal and multitemporal models and the fusion module.
    """

    def __init__(self, config):
        super(FLAIR_TimeTexture, self).__init__()
        
    
        ### CHECK IF BOTH MONO & MULTI MODELS ARE ON
        self.mono_keys = ['AERIAL_RGBI', 'AERIAL-RLT_PAN', 'DEM_ELEV', 'SPOT_RGBI']
        self.mono_init = any(config['modalities']['inputs'][key] for key in self.mono_keys)
        self.mono_mod_unique = sum(config['modalities']['inputs'][key] for key in self.mono_keys) == 1

        self.multi_keys = ['SENTINEL2_TS', 'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS']
        self.multi_init = any(config['modalities']['inputs'][key] for key in self.multi_keys)
        self.multi_mod_unique = sum(config['modalities']['inputs'][key] for key in self.multi_keys) == 1
  

        ### MONO ARCH
        if self.mono_init:
            self.arch = config['models']['monotemp_model']['arch'].split('_')[0]

        ### OUT CHANNELS LOGITS 
        self.n_classes = len(config['classes'])

        ### MODELS
        self.models = nn.ModuleDict({
            'AERIAL_RGBI': None,
            'AERIAL-RLT_PAN': None,
            'DEM_ELEV': None,
            'SPOT_RGBI': None,
            'SENTINEL2_TS': None,
            'SENTINEL1-ASC_TS': None,
            'SENTINEL1-DESC_TS': None,
            'FUSION_DECODER': None
        })    

        ### AUX LOSSES ACTIVATED 
        self.aux_losses = config['modalities']['aux_loss'] 

        ### MODALITIES DROPOUT PROB 
        self.modalities_dropout = config['modalities']['modality_dropout']

        ### CHANNELS PER MODALITY
        self.channels_dict = {
            'AERIAL_RGBI': len(config['modalities']['inputs_channels']['aerial']),
            'AERIAL-RLT_PAN': 1,
            'DEM_ELEV': 1 if config['modalities']['pre_processings']['calc_elevation'] else 2,
            'SPOT_RGBI': len(config['modalities']['inputs_channels']['spot']),
            'SENTINEL2_TS': len(config['modalities']['inputs_channels']['sentinel2']),
            'SENTINEL1-ASC_TS': len(config['modalities']['inputs_channels']['sentinel1']),
            'SENTINEL1-DESC_TS': len(config['modalities']['inputs_channels']['sentinel1'])
        }    

        ### INITIALIZE MONOTEMP MODELS -> FULL OR BACKBONE W.R.T. AUX LOSS ASKED
        for modality in self.mono_keys:
            if config['modalities']['inputs'][modality]:
                channels = self.channels_dict[modality]
                return_type = 'full' if self.aux_losses[modality] else 'backbone'
                self.models[modality]  = self.initialize_monotemp_model(
                    config, channels, self.n_classes, return_type
                )
                
        #### IF ANY MONO -> ADJUST LENGTH FEATURE MAPS MULTI
        mono_key_on = next((key for key in self.mono_keys if self.models[key] is not None), None)
        if self.mono_init and self.multi_init:     
            try:
                backbone_channels = self.models[mono_key_on].seg_model.encoder.out_channels if self.arch == 'smp' else self.models[mono_key_on].seg_model.backbone.channels
            except AttributeError:
                backbone_channels = self.models[mono_key_on].seg_model.out_channels if self.arch == 'smp' else self.models[mono_key_on].seg_model.channels            
            config['models']['multitemp_model']['encoder_widths'] = self.adjust_fm_length(config, backbone_channels)
            config['models']['multitemp_model']['decoder_widths'] = self.adjust_fm_length(config, backbone_channels)            
        
        #### IF OUT_CONV[-1] != NB_CLASSES -> APPEND NB_CLASSES 
        if self.n_classes != config['models']['multitemp_model']['out_conv'][-1] and self.multi_init:
            config['models']['multitemp_model']['out_conv'].append(self.n_classes)             
            
        #### INITIALIZE MULTITEMP MODELS -> FULL
        for modality in self.multi_keys:
            if config['modalities']['inputs'][modality]:
                channels = self.channels_dict[modality]
                self.models[modality] = self.initialize_multitemp_model(
                    config, channels
                )

        if self.mono_init:        
            backbones_channels_per_stage = self.calc_backbones_channels(config)                
            self.models['FUSION_DECODER']  = self.initialize_monotemp_model(
                        config, channels, self.n_classes, 'decoder'
            )
            if self.arch == 'hf':
                target_channels = self.models['FUSION_DECODER'].seg_model.in_channels 
            elif self.arch == 'smp':
                try: 
                    target_channels = self.models[mono_key_on].seg_model.encoder.out_channels   ## TODO here mono_key_on ? Channels first block might vary ?                      
                except AttributeError:
                    target_channels = self.models[mono_key_on].seg_model.out_channels   ## TODO here mono_key_on ? Channels first block might vary ?                      
            self.fusion_fitter = FusionFitter(backbones_channels_per_stage, target_channels)

        elif not self.mono_init and self.multi_init:
            pass
            #### INITALIZE FOR SEN FUSION IF NEEDED....
   
        self.print_model_parameters(self.models, self.aux_losses)



    def initialize_monotemp_model(self, config, channels, n_classes, return_type='full'):
        """
        Initialize the monotemporal model.
        Args:
            config (dict): Configuration dictionary.
            channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            return_type (str): Type of return ('full' or 'backbone').
        Returns:
            model: Initialized monotemporal model.
        """
        model = FLAIR_MonoTemporal(
            config,
            channels=channels,
            classes=n_classes,
            return_type=return_type
        )
        if config['hyperparams']['accelerator'] == 'gpu':
            model = model.cuda()
        return model


    def initialize_multitemp_model(self, config, channels):
        """
        Initialize the multitemporal model.
        Args:
            config (dict): Configuration dictionary.
            channels (int): Number of input channels.
        Returns:
            model: Initialized multitemporal model.
        """
        model = UTAE(
            config,
            input_dim=channels,
            return_maps=True
        )
        if config['hyperparams']['accelerator'] == 'gpu':
            model = model.cuda()
        return model


    def adjust_fm_length(self, config, mono_temp_backbone_channels):
        """
        Adjust the feature map length to match the monotemporal backbone channels.
        Args:
            config (dict): Configuration dictionary.
            mono_temp_backbone_channels (list): List of monotemporal backbone channels.
        Returns:
            list: Adjusted feature map lengths.
        """
        def round_to_nearest_power_of_eight(x):
            return 2 ** round(math.log(x, 2))

        encoder_widths = config['models']['multitemp_model']['encoder_widths']
        min_value = min(encoder_widths)
        max_value = max(encoder_widths)

        target_size = len(mono_temp_backbone_channels)
        expanded_widths = np.linspace(min_value - 1, max_value + 1, target_size).astype(int)

        return [round_to_nearest_power_of_eight(value) for value in expanded_widths]


    @rank_zero_only
    def print_model_parameters(self, models, aux_losses):
        """
        Print the total number of parameters in the model, broken down by each model type.
        Args:
            models (dict): Dictionary of models.
            aux_losses (dict): Dictionary of auxiliary losses.
        """
        total_params = 0
        table = f"| {'Model Key':<20} | {'Type':<10} | {'Aux loss':<10} | {'Parameters':<15} |\n"
        table += f"| {'-'*20} | {'-'*10} | {'-'*10} | {'-'*15} |\n"
        for key, model in models.items():
            if model is not None:
                if key != 'FUSION_DECODER':
                    num_params = sum(p.numel() for p in model.parameters())
                    total_params += num_params
                    model_type = 'full' if aux_losses[key] or key in self.multi_keys else 'backbone'
                    aux_loss = 'Yes' if aux_losses[key] else 'No'
                    table += f"| {key:<20} | {model_type:<10} | {aux_loss:<10} | {num_params:>15,} |\n"
                else:
                    num_params = sum(p.numel() for p in model.parameters())
                    total_params += num_params
                    model_type = 'decoder'
                    aux_loss = 'No'
                    table += f"| {key:<20} | {model_type:<10} | {aux_loss:<10} | {num_params:>15,} |\n"
        table += f"| {'**TOTAL PARAMS**':<20} | {'':<10} | {'':<10} | {total_params:>15,} |\n"
        print('')
        print(table)


    def calc_backbones_channels(self, config):
        """
        Calculate the total number of channels for the backbones.
        Args:
            config (dict): Configuration dictionary.
        Returns:
            list: Total number of channels per stage.
        """
        backbones_total_channels = []
        for key in self.mono_keys:
            if self.models[key]:
                try:
                    channels = self.models[key].seg_model.encoder.out_channels if self.arch == 'smp' else self.models[key].seg_model.backbone.channels
                except AttributeError:
                    channels = self.models[key].seg_model.out_channels if self.arch == 'smp' else self.models[key].seg_model.channels
                backbones_total_channels.append(list(channels))
        reversed_decoder_fm = list(config['models']['multitemp_model']['decoder_widths'])[::-1]
        for key in self.multi_keys:
            if self.models[key]:
                backbones_total_channels.append(reversed_decoder_fm)
        total_per_stage = [sum(x) for x in zip(*backbones_total_channels)]

        return total_per_stage


    def process_monotemp_model(self, model, batch, architecture, key, model_type):
        """
        Process a monotemporal model based on the given key, architecture, and batch data.
        Args:
            model: The monotemporal model.
            batch (dict): Batch data.
            architecture (str): Model architecture ('hf' or 'smp').
            key (str): Key for the batch data.
            model_type (str): Type of model ('full' or 'backbone').
        Returns:
            list: Processed feature maps.
        """
        if model_type == 'full':
            if architecture == 'hf':
                return list(model.seg_model.backbone(batch[key]).feature_maps)
            elif architecture == 'smp':
                return list(model.seg_model.encoder(batch[key]))
        elif model_type == 'backbone':
            if architecture == 'hf':
                return list(model.seg_model(batch[key]).feature_maps)
            elif architecture == 'smp':
                return list(model.seg_model(batch[key]))
        return None


    def decode_model(self, model, fused_feature, architecture):
        """
        Decode the fused feature using the given model and architecture.
        Args:
            model: The model to use for decoding.
            fused_feature (torch.Tensor): The fused feature tensor.
            architecture (str): Model architecture ('hf' or 'smp').
        Returns:
            torch.Tensor: Decoded feature map.
        """
        if architecture == 'hf':
            return model.seg_model.decode_head(fused_feature)
        elif architecture == 'smp':
            out_map = model.seg_model.decoder(*fused_feature)
            return model.seg_model.segmentation_head(out_map)
        return None


    def interpolate_map(self, x, size):
        """
        Interpolate the input tensor to the given size.
        Args:
            x (torch.Tensor): Input tensor.
            size (tuple): Target size (H, W).
        Returns:
            torch.Tensor: Interpolated tensor.
        """
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


    def apply_modality_dropout(self, feature_maps, modalities_dropout_dict):
        """
        Apply modality dropout to feature maps based on the configuration.
        Args:
            feature_maps (dict): Dictionary of feature maps to be aligned.
            modalities_dropout_dict (dict): Dictionary containing dropout probabilities for each modality.
        Returns:
            dict: Updated feature maps with modality dropout applied and replaced by parameter tensors.
        """
        for key in feature_maps.keys():
            dropout_prob = modalities_dropout_dict[key]
            if torch.rand(1).item() < dropout_prob:
                param_features = []
                for tensor in feature_maps[key]:
                    param_tensor = nn.Parameter(torch.empty_like(tensor))
                    nn.init.xavier_uniform_(param_tensor)
                    param_features.append(param_tensor)
                feature_maps[key] = param_features
        return feature_maps


    
    def forward(self, batch, apply_mod_dropout=True):
        """
        Forward pass for FLAIR_TimeTexture.
        Processes monotemporal and multitemporal feature maps, aligns them, 
        and fuses them using the specified operation before decoding.
        """

        dict_featuremaps = {}
        dict_logits = {}

        img_label_size = batch['LABELS'][0].shape[-1]

        #### PROCESS MONOTEMP MODELS
        if self.models['AERIAL_RGBI']:
            dict_featuremaps['AERIAL_RGBI'] = self.process_monotemp_model(
                self.models['AERIAL_RGBI'], 
                batch,
                self.arch,
                'AERIAL_RGBI', 
                model_type = 'full' if self.aux_losses['AERIAL_RGBI'] else 'backbone'
            )
            if self.aux_losses['AERIAL_RGBI']:
                if self.arch == 'hf':
                    _decoded = self.models['AERIAL_RGBI'].seg_model.decode_head(dict_featuremaps['AERIAL_RGBI'])
                    dict_logits['AERIAL_RGBI'] = self.interpolate_map(_decoded, img_label_size)
                elif self.arch == 'smp':
                    _decoded = self.models['AERIAL_RGBI'].seg_model.decoder(*dict_featuremaps['AERIAL_RGBI'])
                    dict_logits['AERIAL_RGBI'] = self.models['AERIAL_RGBI'].seg_model.segmentation_head(_decoded)
                
        if self.models['SPOT_RGBI']:
            dict_featuremaps['SPOT_RGBI'] = self.process_monotemp_model(
                self.models['SPOT_RGBI'],
                batch,  
                self.arch, 
                'SPOT_RGBI',
                model_type = 'full' if self.aux_losses['SPOT_RGBI'] else 'backbone'
            )
            if self.aux_losses['SPOT_RGBI']:
                if self.arch == 'hf':
                    _decoded = self.models['SPOT_RGBI'].seg_model.decode_head(dict_featuremaps['SPOT_RGBI'])
                    dict_logits['SPOT_RGBI'] = self.interpolate_map(_decoded, img_label_size)
                elif self.arch == 'smp':
                    _decoded = self.models['SPOT_RGBI'].seg_model.decoder(*dict_featuremaps['SPOT_RGBI'])
                    dict_logits['SPOT_RGBI'] = self.interpolate_map(self.models['SPOT_RGBI'].seg_model.segmentation_head(_decoded), img_label_size)
            
        if self.models['AERIAL-RLT_PAN']:
            dict_featuremaps['AERIAL-RLT_PAN'] = self.process_monotemp_model(
                self.models['AERIAL-RLT_PAN'], 
                batch,
                self.arch,
                'AERIAL-RLT_PAN', 
                model_type = 'full' if self.aux_losses['AERIAL-RLT_PAN'] else 'backbone'
            )
            if self.aux_losses['AERIAL-RLT_PAN']:
                if self.arch == 'hf':
                    _decoded = self.models['AERIAL-RLT_PAN'].seg_model.decode_head(dict_featuremaps['AERIAL-RLT_PAN'])
                    dict_logits['AERIAL-RLT_PAN'] = self.interpolate_map(_decoded, img_label_size)
                elif self.arch == 'smp':
                    _decoded = self.models['AERIAL-RLT_PAN'].seg_model.decoder(*dict_featuremaps['AERIAL-RLT_PAN'])
                    dict_logits['AERIAL-RLT_PAN'] = self.models['AERIAL-RLT_PAN'].seg_model.segmentation_head(_decoded)
            
        if self.models['DEM_ELEV']:
            dict_featuremaps['DEM_ELEV'] = self.process_monotemp_model(
                self.models['DEM_ELEV'], 
                batch,
                self.arch,
                'DEM_ELEV', 
                model_type = 'full' if self.aux_losses['DEM_ELEV'] else 'backbone'
            )
            if self.aux_losses['DEM_ELEV']:
                if self.arch == 'hf':
                    _decoded = self.models['DEM_ELEV'].seg_model.decode_head(dict_featuremaps['DEM_ELEV'])
                    dict_logits['DEM_ELEV'] = self.interpolate_map(_decoded, img_label_size)
                elif self.arch == 'smp':
                    _decoded = self.models['DEM_ELEV'].seg_model.decoder(*dict_featuremaps['DEM_ELEV'])
                    dict_logits['DEM_ELEV'] = self.models['DEM_ELEV'].seg_model.segmentation_head(_decoded)
                
        #### PROCESS MULTITEMP MODELS                
        if self.models['SENTINEL2_TS']:
            outconv_sen2, fmaps_sen2 = self.models['SENTINEL2_TS'](
                batch['SENTINEL2_TS'], 
                batch_positions=batch['SENTINEL2_TS_DATES']
            )
            dict_featuremaps['SENTINEL2_TS'] = list(fmaps_sen2)
            if self.aux_losses['SENTINEL2_TS'] or self.multi_mod_unique:
                dict_logits['SENTINEL2_TS'] =  self.interpolate_map(outconv_sen2, img_label_size)        

        if self.models['SENTINEL1-ASC_TS']:
            outconv_sen1dasc, fmaps_sen1asc = self.models['SENTINEL1-ASC_TS'](
                batch['SENTINEL1-ASC_TS'], 
                batch_positions=batch['SENTINEL1-ASC_DATES']
            )
            dict_featuremaps['SENTINEL1-ASC_TS'] = list(fmaps_sen1asc)
            if self.aux_losses['SENTINEL1-ASC_TS'] or self.multi_mod_unique:
                dict_logits['SENTINEL1-ASC_TS'] =  self.interpolate_map(outconv_sen1dasc, img_label_size)             

        if self.models['SENTINEL1-DESC_TS']:
            outconv_sen1desc, fmaps_sen1desc = self.models['SENTINEL1-DESC_TS'](
                batch['SENTINEL1-DESC_TS'], 
                batch_positions=batch['SENTINEL1-DESC_DATES']
            )
            dict_featuremaps['SENTINEL1-DESC_TS'] = list(fmaps_sen1desc)
            if self.aux_losses['SENTINEL1-DESC_TS'] or self.multi_mod_unique:
                dict_logits['SENTINEL1-DESC_TS'] =  self.interpolate_map(outconv_sen1desc, img_label_size)

        #### MODALITY DROPOUT   
        if apply_mod_dropout and not self.mono_mod_unique and not self.multi_mod_unique :
            dict_featuremaps_dropout = self.apply_modality_dropout(dict_featuremaps, self.modalities_dropout) 
        else:
            dict_featuremaps_dropout = dict_featuremaps       
                
        #### FUSION         
        if self.mono_init:
            mono_key_on = next((key for key in self.mono_keys if self.models[key] is not None), None)
            decode_in_shapes = [fm.shape for fm in dict_featuremaps_dropout[mono_key_on]]

            fused_features = self.fusion_fitter(dict_featuremaps_dropout, decode_in_shapes)
            if self.arch == 'hf':
                out_fusion = self.models['FUSION_DECODER'].seg_model(fused_features)
                dict_logits['FUSED'] = self.interpolate_map(out_fusion, img_label_size)
            elif self.arch == 'smp':
                out_fusion = self.models['FUSION_DECODER'].seg_model(*fused_features)
                dict_logits['FUSED'] = self.models['FUSION_DECODER'].seg_model_head(out_fusion)
                
                
        elif not self.mono_init and self.multi_init and not self.multi_mod_unique:
            out_fusion = [torch.mean(torch.stack(tensors), dim=0) for tensors in zip(*[dict_logits[i] for i in self.multi_keys if i in dict_logits])]
            dict_logits['FUSED'] = self.interpolate_map(torch.stack(out_fusion), img_label_size)

        elif not self.mono_init and self.multi_mod_unique:
            dict_logits['FUSED'] = self.interpolate_map(dict_logits[list(dict_logits.keys())[0]], img_label_size)



        return dict_logits


