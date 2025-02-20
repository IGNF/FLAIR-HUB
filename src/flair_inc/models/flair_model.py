import numpy as np
import math
import random
import torch
import torch.nn as nn

from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from flair_inc.models.monotemp_model import FLAIR_MonoTemporal
from flair_inc.models.multitemp_model import UTAE





class FLAIR_TimeTexture(nn.Module):
    def __init__(self, config, img_input_sizes):
        super(FLAIR_TimeTexture, self).__init__()
        self.config = config
        self.img_input_sizes = img_input_sizes
        
        self.mono_keys = ['AERIAL_RGBI', 'AERIAL-RLT_PAN', 'DEM_ELEV', 'SPOT_RGBI']
        self.multi_keys = ['SENTINEL2_TS', 'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS']
        
        # Auxiliary losses
        aux_losses = config['modalities']['aux_loss']
        self.aux_losses = {
            mod: aux_losses[mod]
            for mod in aux_losses
            if aux_losses[mod] and config['modalities']['inputs'].get(mod, False)
        }
        
        # Tasks
        self.tasks = len(config['labels'])
        self.task_nclasses = sum(
            [len(config['labels_configs'][config['labels'][i]]['value_name']) for i in range(self.tasks)]
        )
        
        # Gather modality channels
        self.channels_dict = {
            mod: (1 if mod in ['AERIAL-RLT_PAN', 'DEM_ELEV'] else len(config['modalities']['inputs_channels'][mod]))
            for mod in config['modalities']['inputs']
        }
        if config['modalities']['inputs']['DEM_ELEV']:
            self.channels_dict['DEM_ELEV'] = (
                1 if config['modalities']['pre_processings']['calc_elevation'] 
                     and not config['modalities']['pre_processings']['calc_elevation_stack_dsm']
                else 2
            )
        
        # Encoders
        self.encoders = nn.ModuleDict()
        for mod in self.mono_keys:
            if config['modalities']['inputs'].get(mod, False):
                self.encoders[mod] = FLAIR_MonoTemporal(
                    config,
                    channels=self.channels_dict[mod],
                    classes=self.task_nclasses,
                    img_size=img_input_sizes[mod],
                    return_type='encoder',
                )
        
        # Adjust UTAE parameters for multi-modal fusion
        if any(config['modalities']['inputs'] for key in self.multi_keys):
            if self.task_nclasses != config['models']['multitemp_model']['out_conv'][-1]:
                config['models']['multitemp_model']['out_conv'].append(self.task_nclasses)

        if any(config['modalities']['inputs'].get(key, False) for key in self.multi_keys) and self.encoders:
            mono_encoder_out_channels = self.encoders[list(self.encoders.keys())[0]].seg_model.out_channels
            config['models']['multitemp_model']['encoder_widths'] = self.adjust_fm_length(config, mono_encoder_out_channels)
            config['models']['multitemp_model']['decoder_widths'] = self.adjust_fm_length(config, mono_encoder_out_channels)
        
        for mod in self.multi_keys:
            if config['modalities']['inputs'].get(mod, False):
                self.encoders[mod] = UTAE(
                    input_dim=len(config['modalities']['inputs_channels'][mod]),
                    encoder_widths=config['models']['multitemp_model']['encoder_widths'],
                    decoder_widths=config['models']['multitemp_model']['decoder_widths'],
                    out_conv=config['models']['multitemp_model']['out_conv'],
                    str_conv_k=config['models']['multitemp_model']['str_conv_k'],
                    str_conv_s=config['models']['multitemp_model']['str_conv_s'],
                    str_conv_p=config['models']['multitemp_model']['str_conv_p'],
                    agg_mode=config['models']['multitemp_model']['agg_mode'],
                    encoder_norm=config['models']['multitemp_model']['encoder_norm'],
                    n_head=config['models']['multitemp_model']['n_head'],
                    d_model=config['models']['multitemp_model']['d_model'],
                    d_k=config['models']['multitemp_model']['d_k'],
                    encoder=False,
                    return_maps=True,
                    pad_value=config['models']['multitemp_model']['pad_value'],
                    padding_mode=config['models']['multitemp_model']['padding_mode'],
                )
        
        if any([key for key in self.mono_keys if key in self.encoders]) : 
            encoders_out_channels = self.calc_backbones_channels(config)
            target_fused_channels = self.encoders[list(self.encoders.keys())[0]].seg_model.out_channels            
            self.fusion_handler = FusionHandler(
                backbones_channels=encoders_out_channels,
                target_fused_channels=target_fused_channels,
                mono_keys=self.mono_keys,
                multi_keys=self.multi_keys,
            )            
        else:
            encoders_out_channels = [1] # Dummy value
            target_fused_channels = [1] # Dummy value         
            self.fusion_handler = FusionHandler(
                backbones_channels=encoders_out_channels,
                target_fused_channels=target_fused_channels,
                mono_keys=self.mono_keys,
                multi_keys=self.multi_keys,
            )                 
        
        # Decoders
        self.main_decoders = nn.ModuleDict()
        for task in config['labels']:
            if any(key in self.encoders for key in self.mono_keys):
                self.main_decoders[task] = FLAIR_MonoTemporal(
                    config, channels=1, classes=len(config['labels_configs'][task]['value_name']), return_type='decoder'
                )
            else:
                self.main_decoders[task] = nn.Conv2d(
                    in_channels=self.task_nclasses, out_channels=len(config['labels_configs'][task]['value_name']), kernel_size=1
                )
        
        self.aux_decoders = nn.ModuleDict()
        for mod in self.aux_losses:
            for task in config['labels']:
                if mod in self.mono_keys:
                    self.aux_decoders[f'{mod}__{task}'] = FLAIR_MonoTemporal(
                        config, channels=1, classes=len(config['labels_configs'][task]['value_name']), return_type='decoder'
                    )
                elif mod in self.multi_keys:
                    self.aux_decoders[f'{mod}__{task}'] = nn.Conv2d(
                        in_channels=self.task_nclasses, out_channels=len(config['labels_configs'][task]['value_name']), kernel_size=1
                    )


        self.print_model_parameters(self.encoders, 
                                    self.main_decoders, 
                                    self.aux_decoders, 
                                    self.mono_keys, 
                                    self.multi_keys, 
                                    self.config
        )

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

        if len(mono_temp_backbone_channels) > 2 and (mono_temp_backbone_channels[0] == 0 or mono_temp_backbone_channels[1] == 0):
            mono_temp_backbone_channels = mono_temp_backbone_channels[2:]
            
        encoder_widths = config['models']['multitemp_model']['encoder_widths']
        min_value = min(encoder_widths)
        max_value = max(encoder_widths)
        
        target_size = len(mono_temp_backbone_channels)
        expanded_widths = np.linspace(min_value - 1, max_value + 1, target_size).astype(int)
        
        return [round_to_nearest_power_of_eight(value) for value in expanded_widths]


    @rank_zero_only
    def print_model_parameters(self, encoders, decoders_main, decoders_aux, mono_keys, multi_keys, config):
        """
        Print the total number of parameters in the encoders and decoders, broken down by type.
        Args:
            encoders (nn.ModuleDict): Dictionary of encoder models.
            decoders_main (nn.ModuleDict): Dictionary of main decoder models.
            decoders_aux (nn.ModuleDict): Dictionary of auxiliary decoder models.
            mono_keys (list): List of keys that define mono-type encoders or decoders.
            multi_keys (list): List of keys that define multi-type decoders.
            config (dict): Configuration dictionary.
        """
        total_params = 0
        table = " " + "-"*113 + "\n"  # Ensure newline is explicitly added
        table += ("| {:<37} | {:<35} | {:<17} | {:<13} |\n"
                "| {} | {} | {} | {} |\n").format("Model modality", "Architecture", "Type", "Parameters",
                                                    "-"*37, "-"*35, "-"*17, "-"*13)

        has_mono_key = any(key in mono_keys for key in encoders)
        default_decoder_arch = 'utae'
        
        for key, model in encoders.items():
            if model is not None:
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                if key in mono_keys:
                    encoder = config['models']['monotemp_model']['arch'].split('-')[0]
                elif key in multi_keys:
                    encoder = 'utae'
                else:
                    encoder = 'Unknown'
                table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format(key, encoder, 'backbone', num_params)

        table += "| {} | {} | {} | {} |\n".format("-"*37, "-"*35, "-"*17, "-"*13)
        for key, model in decoders_aux.items():
            if model is not None:
                first_part_of_key = key.split('__')[0]
                if first_part_of_key in multi_keys:
                    decoder_arch = 'utae'
                else:
                    decoder_arch = config['models']['monotemp_model']['arch'].split('-')[1] if has_mono_key else 'utae'
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format(key, decoder_arch, 'aux loss decoder', num_params)

        table += "| {} | {} | {} | {} |\n".format("-"*37, "-"*35, "-"*17, "-"*13)
        for key, model in decoders_main.items():
            if model is not None:
                if has_mono_key:
                    decoder_arch = config['models']['monotemp_model']['arch'].split('-')[1]
                else:
                    decoder_arch = default_decoder_arch
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format(key, decoder_arch, 'task decoder', num_params)

        # Total Parameters
        table += "|" + "-"*113 +'|'
        table += "\n| {:<37}   {:<35}   {:<17}   {:>13,} |\n".format("Total parameters", "", "", total_params)
        table += ' '+"-"*113

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
        for mod in self.encoders:
            if mod in self.mono_keys:
                out_channels = self.encoders[mod].seg_model.out_channels
                # Check if the first or second item is 0 (conventioon in smp)
                if len(out_channels) > 2 and (out_channels[0] == 0 or out_channels[1] == 0):
                    channels = out_channels[2:]  # Take from index 2 onwards
                else:
                    channels = out_channels  # Keep all channels 
                backbones_total_channels.append(list(channels))

        reversed_decoder_fm = list(self.config['models']['multitemp_model']['decoder_widths'])[::-1] #reversed as order given is top to bottom see : https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py
        for mod in self.encoders:
            if mod in self.multi_keys:
                backbones_total_channels.append(reversed_decoder_fm)
        total_per_stage = [sum(x) for x in zip(*backbones_total_channels)]

        return total_per_stage


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


    def modality_dropout(self, feature_maps, modalities_dropout_dict):
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

    
    def forward(self, batch, apply_mod_dropout=False):

        FMAPS, LOGITS_TASKS, LOGITS_AUX = {}, {}, {}

        active_mono_keys = [key for key in self.mono_keys if key in self.encoders]
        active_multi_keys = [key for key in self.multi_keys if key in self.encoders]
        img_size = batch[self.config['labels'][0]].shape[-1]
        
        for mod, encoder in self.encoders.items():
            
            if mod in self.mono_keys:
                # Process mono modality encoders
                FMAPS[mod] = encoder.seg_model(batch[mod])

                # Populate LOGITS_AUX immediately if aux losses are active
                if self.aux_losses.get(mod):
                    for task in self.config['labels']:             
                        aux_key = f"{mod}__{task}"
                        if aux_key in self.aux_decoders:
                            aux_decoder = self.aux_decoders[aux_key]
                            LOGITS_AUX[f'aux_{mod}_{task}'] = self.interpolate_map(aux_decoder.seg_model(*FMAPS[mod]), img_size)
                            
            else:
                # Process multi-modality encoders (UTAE: return both feature maps and logits)
                logits, fmaps = encoder(batch[mod], batch_positions=batch.get(mod.replace('TS', 'DATES')))
                logits = self.interpolate_map(logits, img_size)
                 
                LOGITS_TASKS[mod] = self.interpolate_map(logits, img_size)  # Ensure consistent size
                FMAPS[mod] = fmaps

                # Populate LOGITS_AUX immediately if aux losses are active
                if self.aux_losses.get(mod):
                    for task in self.config['labels']: 
                        aux_key = f"{mod}__{task}"
                        if aux_key in self.aux_decoders:
                            aux_decoder = self.aux_decoders[aux_key]
                            LOGITS_AUX[f'aux_{mod}_{task}'] = self.interpolate_map(aux_decoder(logits), img_size)



        if apply_mod_dropout and len(self.encoders) > 1:
            print('--- PERFORMING MODALITY DROPOUT ---')
            modalities_dropout_dict = {key: random.uniform(0, 1) for key in FMAPS.keys()}
            FMAPS = self.modality_dropout(FMAPS, modalities_dropout_dict)  


        if active_mono_keys != []:
            fused_features = self.fusion_handler(FMAPS, FMAPS[active_mono_keys[0]])
        else:
            fused_features = self.fusion_handler(LOGITS_TASKS, LOGITS_TASKS[active_multi_keys[0]]) 


        for task in self.config['labels']:
            if active_mono_keys != []:    
                LOGITS_TASKS[task] = self.interpolate_map(self.main_decoders[task].seg_model(*fused_features), img_size)
            else:
                if len(self.config['labels']) > 1:
                    LOGITS_TASKS[task] = self.main_decoders[task](fused_features)    
                else:
                    LOGITS_TASKS[task] = fused_features               


        for mod in list(LOGITS_TASKS.keys()):  
            if mod in self.config['modalities']['inputs']:
                del LOGITS_TASKS[mod]  

        return LOGITS_TASKS, LOGITS_AUX










import torch.nn.functional as F

class FusionHandler(nn.Module):
    def __init__(self, backbones_channels, target_fused_channels, mono_keys, multi_keys):
        """
        Initialize the FusionHandler module to handle different fusion scenarios.
        """
        super(FusionHandler, self).__init__()

        self.mono_keys = mono_keys  # All possible mono_keys
        self.multi_keys = multi_keys  # All possible multi_keys

        # Remove dummy channels if needed
        if len(target_fused_channels) > 2 and (target_fused_channels[0] == 0 or target_fused_channels[1] == 0):
            target_fused_channels = target_fused_channels[2:]

        # Convolution layers for feature fusion
        self.conv_f = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            for in_ch, out_ch in zip(backbones_channels, target_fused_channels)
        ])

    def forward(self, feature_maps, target_fm_maps):
        active_keys = list(feature_maps.keys())  # Only active feature map keys
        mono_key_active = [key for key in active_keys if key in self.mono_keys]
        multi_key_active = [key for key in active_keys if key in self.multi_keys]

        # 1. Only one mono_key is active → No fusion
        if len(mono_key_active) == 1 and len(multi_key_active) == 0:
            return feature_maps[mono_key_active[0]]

        # 2. Only one multi_key is active → No fusion
        if len(mono_key_active) == 0 and len(multi_key_active) == 1:
            return feature_maps[multi_key_active[0]]

        # 3. Multiple multi_keys are active, but no mono_key → Stack & Mean
        if len(mono_key_active) == 0 and len(multi_key_active) > 1:
            stacked_feature_maps = torch.stack([feature_maps[key] for key in multi_key_active], dim=0)
            return torch.mean(stacked_feature_maps, dim=0)  # Mean over stacked maps

        # 4. At least one mono_key and one multi_key are active → Full Fusion Process
        target_shapes = [fm.shape for fm in target_fm_maps]

        if target_shapes[0][1] == 0 or target_shapes[1][1] == 0:
            target_shapes = target_shapes[2:]
            dummy_shapes = target_fm_maps[:2]
        else:
            dummy_shapes = None  # No need for dummy channels

        aligned_fmaps = []

        for mod in active_keys:  # Use only active keys
            mod_fmaps = feature_maps[mod]  # List of tensors

            if mod_fmaps[0].shape[1] == 0 or mod_fmaps[1].shape[1] == 0:
                mod_fmaps = mod_fmaps[2:]

            if len(mod_fmaps) != len(target_shapes):  # Ensure correct number of feature maps
                mod_fmaps = [mod_fmaps[0]] * (len(target_shapes) - len(mod_fmaps)) + mod_fmaps

            # Align feature maps to target shapes
            resized_fmaps = []
            for fmap, target in zip(mod_fmaps, target_shapes):
                target_h, target_w = target[-2], target[-1]

                if fmap.shape[-1] != target_w or fmap.shape[-2] != target_h:
                    fmap = F.interpolate(fmap, size=(target_h, target_w), mode='bilinear', align_corners=False)

                resized_fmaps.append(fmap)

            aligned_fmaps.append(resized_fmaps)  # Store aligned feature maps

        # Stack feature maps along channel dimension
        stacked_feature_maps = [torch.cat(fmaps, dim=1) for fmaps in zip(*aligned_fmaps)]

        # Apply convolution layers
        adjusted_feature_maps = [conv_f(fmap) for conv_f, fmap in zip(self.conv_f, stacked_feature_maps)]

        # If dummy_shapes is not None, prepend a zero tensor with the same shape
        if dummy_shapes is not None:
            adjusted_feature_maps = dummy_shapes + adjusted_feature_maps

        return adjusted_feature_maps








class FusionHandler2(nn.Module):
    def __init__(self, backbones_channels, target_fused_channels, mono_keys, multi_keys):
        """
        Initialize the FusionHandler module.
        """
        super(FusionHandler, self).__init__()

        self.mono_keys = mono_keys  
        self.multi_keys = multi_keys  

        if len(target_fused_channels) > 2 and (target_fused_channels[0] == 0 or target_fused_channels[1] == 0):
            target_fused_channels = target_fused_channels[2:]

        self.conv_f = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            for in_ch, out_ch in zip(backbones_channels, target_fused_channels)
        ])

    def forward(self, feature_maps, target_fm_maps):
        active_keys = list(feature_maps.keys())  # Get active feature map keys
        mono_key_active = [key for key in active_keys if key in self.mono_keys]
        multi_key_active = [key for key in active_keys if key in self.multi_keys]

        # Case 1: A single mono_key is active
        if len(mono_key_active) == 1 and len(multi_key_active) == 0:
            return feature_maps[active_keys[0]]  # No fusion, return the input as-is

        # Case 2: A single multi_key is active
        if len(mono_key_active) == 0 and len(multi_key_active) == 1:
            return feature_maps[active_keys[0]]  # No fusion, return the input as-is
            
        # Case 3: Perform fusion (Mono + Multi or Multiple Mono / Multi)
        else:
            target_shapes = [fm.shape for fm in target_fm_maps]
    
            if target_shapes[0][1] == 0 or target_shapes[1][1] == 0:
                target_shapes = target_shapes[2:]
                dummy_shapes = target_fm_maps[:2]
            else:
                target_shapes = target_shapes  # Keep all channels 
                dummy_shapes = None
    
            aligned_fmaps = []
    
            for mod in feature_maps:
                mod_fmaps = feature_maps[mod]  # This is a list of tensors
    
                if mod_fmaps[0].shape[1] == 0 or mod_fmaps[1].shape[1] == 0:
                    mod_fmaps = mod_fmaps[2:]
                    
                if len(mod_fmaps) != len(target_shapes):  # Repeat first fmap if needed
                    mod_fmaps = [mod_fmaps[0]] * (len(target_shapes) - len(mod_fmaps)) + mod_fmaps
    
                # Align feature maps based on spatial dimensions
                resized_fmaps = []
                for fmap, target in zip(mod_fmaps, target_shapes):
                    target_h, target_w = target[-2], target[-1]  # Extract spatial dimensions
    
                    if fmap.shape[-1] != target_w or fmap.shape[-2] != target_h:
                        fmap = F.interpolate(fmap, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
                    resized_fmaps.append(fmap)
    
                aligned_fmaps.append(resized_fmaps)  # Store aligned feature maps
    
            # Stack feature maps along channel dimension
            stacked_feature_maps = [torch.cat(fmaps, dim=1) for fmaps in zip(*aligned_fmaps)]
            
            # Apply convolution layers
            adjusted_feature_maps = [conv_f(fmap) for conv_f, fmap in zip(self.conv_f, stacked_feature_maps)]
            
            # If dummy_shapes is not None, prepend a zero tensor with the same shape
            if dummy_shapes is not None:
                adjusted_feature_maps = dummy_shapes + adjusted_feature_maps
            
            return adjusted_feature_maps