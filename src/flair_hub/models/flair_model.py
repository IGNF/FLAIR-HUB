import numpy as np
import math
import torch
import torch.nn as nn

from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from flair_hub.models.monotemp_model import FLAIR_Monotemp
from flair_hub.models.multitemp_model import UTAE
from flair_hub.models.flair_upernet import FLAIRUPerDecoder, build_norm
from flair_hub.utils.class_config import model_num_classes


def _resolve_monotemp_decoder(
    monotemp_config: dict,
) -> tuple[str, str, dict]:
    """Resolve the decoder architecture, provider, and applicable options.

    The ``upernet_decoder`` section controls only an ``-upernet`` architecture.
    Every other architecture suffix is built by SMP and ignores that section.
    """
    decoder_arch = monotemp_config['arch'].rsplit('-', 1)[-1].lower()
    if decoder_arch != 'upernet':
        return decoder_arch, 'smp', {}

    upernet_config = monotemp_config.get(
        'upernet_decoder',
        monotemp_config.get('decoder_head', {}),
    )
    requested_implementation = str(
        upernet_config.get('implementation', 'smp')
    ).lower()
    implementation_aliases = {
        'flair_hub': 'flair_hub',
        'custom': 'flair_hub',
        'smp': 'smp',
    }
    if requested_implementation not in implementation_aliases:
        raise ValueError(
            "monotemp_model.upernet_decoder.implementation must be "
            "either 'flair_hub' or 'smp'"
        )
    return (
        decoder_arch,
        implementation_aliases[requested_implementation],
        upernet_config,
    )


class FCNAuxHead(nn.Module):
    """FLAIR-HUB FCN deep-supervision head for one encoder feature map."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_classes: int,
        dropout: float = 0.1,
        norm_type: str = "sync_batchnorm",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        norm = build_norm(norm_type, channels)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            norm,
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Match ConvModule and BaseDecodeHead classifier initialization."""
        nn.init.kaiming_normal_(
            self.conv[0].weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if isinstance(self.conv[1], (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
            nn.init.ones_(self.conv[1].weight)
            nn.init.zeros_(self.conv[1].bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.conv(feature_map)))


class FLAIR_HUB_Model(nn.Module):
    """
    FLAIR_HUB_Model is a multi-modal deep learning model for satellite image processing.
    This model processes both mono-temporal and multi-temporal modalities using different encoders,
    fuses their representations, and applies specific decoders for multiple segmentation tasks.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        img_input_sizes (dict): Dictionary specifying input sizes for different modalities.

    Attributes:
        mono_keys (list): List of mono-temporal modality keys.
        multi_keys (list): List of multi-temporal modality keys.
        aux_losses (dict): Dictionary of auxiliary losses enabled in the configuration.
        tasks (int): Number of tasks defined in the configuration.
        task_nclasses (int): Total number of output classes across all tasks.
        channels_dict (dict): Dictionary mapping modalities to their respective input channel sizes.
        encoders (nn.ModuleDict): Dictionary of encoders for different modalities.
        fusion_handler (FusionHandler): Module that handles feature fusion from different encoders.
        main_decoders (nn.ModuleDict): Dictionary of primary decoders for each task.
        aux_decoders (nn.ModuleDict): Dictionary of auxiliary decoders for additional supervision.

    Methods:
        adjust_fm_length(config, channels):
            Adjusts the feature map length for multi-temporal encoders and decoders.
        calc_backbones_channels(config):
            Computes the number of output channels for the backbone encoders.
        print_model_parameters(encoders, main_decoders, aux_decoders, mono_keys, multi_keys, config):
            Logs model parameter details for debugging and validation.
    """

    def __init__(self, config: dict, img_input_sizes: dict):
        super().__init__()
        self.config = config
        self.img_input_sizes = img_input_sizes

        self.mono_keys = ['AERIAL_RGBI', 'AERIAL-RLT_PAN', 'DEM_ELEV', 'SPOT_RGBI']
        self.multi_keys = ['SENTINEL2_TS', 'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS']


        self.aux_losses = {
            mod: loss
            for mod, loss in config['modalities']['aux_loss'].items()
            if loss and config['modalities']['inputs'].get(mod, False)
        }


        self.tasks = len(config['labels'])
        self.task_nclasses = sum(
            model_num_classes(config['labels_configs'][label])
            for label in config['labels']
        )


        self.channels_dict = {
            mod: (
                1 if mod in ["AERIAL-RLT_PAN", "DEM_ELEV"]
                else len(config["modalities"].get("inputs_channels", {}).get(mod) or [])
            )
            for mod, enabled in config["modalities"]["inputs"].items()
            if enabled
        }


        if config['modalities']['inputs']['DEM_ELEV']:
            self.channels_dict['DEM_ELEV'] = (
                (2 if config['modalities']['pre_processings']['calc_elevation_stack_dsm'] else 1)
                if config['modalities']['pre_processings']['calc_elevation']
                else len(config['modalities']['inputs_channels']['DEM_ELEV'])
            )


        self.encoders = nn.ModuleDict()
        self.mono_encoder_channels = {}
        for mod in self.mono_keys:
            if config['modalities']['inputs'].get(mod, False):
                self.encoders[mod] = FLAIR_Monotemp(
                    config,
                    channels=self.channels_dict[mod],
                    classes=self.task_nclasses,
                    img_size=img_input_sizes[mod],
                    return_type='encoder',
                )
                self.mono_encoder_channels[mod] = self._usable_encoder_channels(
                    self.encoders[mod].seg_model.out_channels
                )

        fusion_config = config['models'].get('fusion', {})
        feature_map_source = fusion_config.get('feature_source', 'encoder')

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
                    feature_map_source=feature_map_source,
                    pad_value=config['models']['multitemp_model']['pad_value'],
                    padding_mode=config['models']['multitemp_model']['padding_mode'],
                )

        if any(mod in self.encoders for mod in self.mono_keys):
            encoders_out_channels = self.calc_backbones_channels()
            target_fused_channels = next(iter(self.encoders.values())).seg_model.out_channels
        else:
            encoders_out_channels = [1]
            target_fused_channels = [1]

        modality_channels = {}
        for mod, encoder in self.encoders.items():
            if mod in self.mono_keys:
                modality_channels[mod] = self._usable_encoder_channels(
                    encoder.seg_model.out_channels
                )[-4:]
            else:
                if feature_map_source == 'encoder':
                    modality_channels[mod] = list(
                        config['models']['multitemp_model']['encoder_widths']
                    )
                else:
                    modality_channels[mod] = list(
                        config['models']['multitemp_model']['decoder_widths']
                    )[::-1]

        self.fusion_handler = FusionHandler(
            backbones_channels=encoders_out_channels,
            target_fused_channels=target_fused_channels,
            modality_channels=modality_channels,
            mono_keys=self.mono_keys,
            multi_keys=self.multi_keys,
            active_modality_count=len(self.encoders),
            method=fusion_config.get('method', 'gated_residual'),
            levels=fusion_config.get('levels', [1, 2, 3]),
            gate_init=float(fusion_config.get('gate_init', -2.0)),
            attention_channels=int(
                fusion_config.get('attention_channels', 128)
            ),
            attention_heads=int(fusion_config.get('attention_heads', 4)),
            window_size=int(fusion_config.get('window_size', 8)),
        )


        self.main_decoders = nn.ModuleDict()
        monotemp_config = config['models']['monotemp_model']
        decoder_arch, decoder_implementation, upernet_config = (
            _resolve_monotemp_decoder(monotemp_config)
        )
        use_flair_upernet = (
            decoder_arch == 'upernet'
            and decoder_implementation == 'flair_hub'
        )

        upernet_in_channels = self._usable_encoder_channels(
            target_fused_channels
        )[-4:]
        for task in config['labels']:
            task_num_classes = model_num_classes(
                config['labels_configs'][task]
            )
            if any(mod in self.encoders for mod in self.mono_keys):
                if use_flair_upernet:
                    self.main_decoders[task] = FLAIRUPerDecoder(
                        in_channels=upernet_in_channels,
                        channels=int(upernet_config.get('channels', 256)),
                        num_classes=task_num_classes,
                        pool_scales=tuple(
                            upernet_config.get('pool_scales', (1, 2, 3, 6))
                        ),
                        dropout=float(upernet_config.get('dropout', 0.1)),
                        norm_type=upernet_config.get(
                            'norm', 'sync_batchnorm'
                        ),
                        align_corners=bool(
                            upernet_config.get('align_corners', False)
                        ),
                    )
                else:
                    self.main_decoders[task] = FLAIR_Monotemp(
                        config,
                        channels=1,
                        classes=task_num_classes,
                        return_type='decoder',
                    )
            else:
                self.main_decoders[task] = nn.Conv2d(
                    in_channels=self.task_nclasses,
                    out_channels=task_num_classes,
                    kernel_size=1,
                )


        self.aux_decoders = nn.ModuleDict()
        aux_head_config = config['models']['monotemp_model'].get(
            'auxiliary_head', {}
        )
        aux_channels = int(aux_head_config.get('channels', 256))
        aux_dropout = float(aux_head_config.get('dropout', 0.1))
        aux_norm = aux_head_config.get('norm', 'sync_batchnorm')

        for task in config['labels']:
            task_num_classes = model_num_classes(
                config['labels_configs'][task]
            )
            for mod in self.aux_losses.keys():
                if mod in self.mono_keys:
                    encoder_channels = self.mono_encoder_channels[mod]
                    if len(encoder_channels) < 2:
                        raise ValueError(
                            f"Auxiliary supervision requires at least two usable "
                            f"encoder stages for {mod}, got {encoder_channels}"
                        )
                    aux_in_channels = encoder_channels[-2]

                    self.aux_decoders[f'{mod}__{task}'] = FCNAuxHead(
                        in_channels=aux_in_channels,
                        channels=aux_channels,
                        num_classes=task_num_classes,
                        dropout=aux_dropout,
                        norm_type=aux_norm,
                    )
                else:
                    self.aux_decoders[f'{mod}__{task}'] = nn.Conv2d(
                        in_channels=self.task_nclasses,
                        out_channels=task_num_classes,
                        kernel_size=1,
                    )

        self.print_model_parameters(
            self.encoders, self.main_decoders, self.aux_decoders,
            self.mono_keys, self.multi_keys, self.config, self.fusion_handler
        )


    def adjust_fm_length(self, config: dict, mono_temp_backbone_channels: list) -> list:
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
    def print_model_parameters(
        self,
        encoders: nn.ModuleDict,
        decoders_main: nn.ModuleDict,
        decoders_aux: nn.ModuleDict,
        mono_keys: list[str],
        multi_keys: list[str],
        config: dict,
        fusion_handler: nn.Module,
    ) -> None:
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
        table = " " + "-"*123 + "\n"
        table += ("| {:<37} | {:<45} | {:<17} | {:<13} |\n"
                "| {} | {} | {} | {} |\n").format("Model modality", "Architecture", "Type", "Parameters",
                                                    "-"*37, "-"*45, "-"*17, "-"*13)

        has_mono_key = any(key in mono_keys for key in encoders)
        default_decoder_arch = 'utae'

        for key, model in encoders.items():
            if model is not None:
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                if key in mono_keys:
                    encoder = config['models']['monotemp_model']['arch'].rsplit('-', 1)[0]
                elif key in multi_keys:
                    encoder = 'utae'
                else:
                    encoder = 'Unknown'
                table += "| {:<37} | {:<45} | {:<17} | {:>13,} |\n".format(key, encoder, 'backbone', num_params)

        fusion_params = sum(p.numel() for p in fusion_handler.parameters())
        if fusion_params:
            total_params += fusion_params
            table += "| {:<37} | {:<45} | {:<17} | {:>13,} |\n".format(
                'MULTIMODAL_FUSION',
                fusion_handler.method,
                'fusion',
                fusion_params,
            )

        table += "| {} | {} | {} | {} |\n".format("-"*37, "-"*45, "-"*17, "-"*13)
        for key, model in decoders_aux.items():
            if model is not None:
                first_part_of_key = key.split('__')[0]
                if isinstance(model, FCNAuxHead):
                    decoder_arch = 'fcn_aux'
                elif first_part_of_key in multi_keys:
                    decoder_arch = 'utae'
                else:
                    decoder_arch = model.__class__.__name__
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                table += "| {:<37} | {:<45} | {:<17} | {:>13,} |\n".format(key, decoder_arch, 'auxiliary head', num_params)

        table += "| {} | {} | {} | {} |\n".format("-"*37, "-"*45, "-"*17, "-"*13)
        for key, model in decoders_main.items():
            if model is not None:
                if has_mono_key:
                    decoder_arch = getattr(
                        model,
                        'architecture_name',
                        config['models']['monotemp_model']['arch'].rsplit('-', 1)[1],
                    )
                else:
                    decoder_arch = default_decoder_arch
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                table += "| {:<37} | {:<45} | {:<17} | {:>13,} |\n".format(key, decoder_arch, 'task decoder', num_params)


        table += "|" + "-"*123 +'|'
        table += "\n| {:<37}   {:<45}   {:<17}   {:>13,} |\n".format("Total parameters", "", "", total_params)
        table += ' '+"-"*123

        print('')
        print(table)


    def calc_backbones_channels(self) -> list[int]:
        """
        Calculate the total number of channels for the backbones.
        Returns:
            list: Total number of channels per stage.
        """
        backbones_total_channels = []
        for mod in self.encoders:
            if mod in self.mono_keys:
                channels = self._usable_encoder_channels(
                    self.encoders[mod].seg_model.out_channels
                )[-4:]
                backbones_total_channels.append(channels)

        fusion_config = self.config['models'].get('fusion', {})
        if fusion_config.get('feature_source', 'encoder') == 'encoder':
            temporal_channels = list(
                self.config['models']['multitemp_model']['encoder_widths']
            )
        else:
            temporal_channels = list(
                self.config['models']['multitemp_model']['decoder_widths']
            )[::-1]
        for mod in self.encoders:
            if mod in self.multi_keys:
                backbones_total_channels.append(temporal_channels)
        total_per_stage = [sum(x) for x in zip(*backbones_total_channels)]

        return total_per_stage

    @staticmethod
    def _usable_encoder_channels(out_channels) -> list[int]:
        """Drop SMP dummy feature entries while retaining real encoder stages."""
        channels = list(out_channels)
        if len(channels) > 2 and (channels[0] == 0 or channels[1] == 0):
            channels = channels[2:]
        return [int(channel) for channel in channels if channel > 0]

    @staticmethod
    def _select_auxiliary_feature(
        feature_maps: list[torch.Tensor],
        aux_head: FCNAuxHead,
    ) -> torch.Tensor:
        """Select the penultimate encoder stage, robust to SMP dummy features."""
        usable_features = [
            feature_map
            for feature_map in feature_maps
            if feature_map.ndim == 4 and feature_map.shape[1] > 0
        ]
        if len(usable_features) >= 2:
            penultimate = usable_features[-2]
            if penultimate.shape[1] == aux_head.in_channels:
                return penultimate

        channel_matches = [
            feature_map
            for feature_map in usable_features
            if feature_map.shape[1] == aux_head.in_channels
        ]
        if channel_matches:
            return channel_matches[-1]

        feature_channels = [feature.shape[1] for feature in usable_features]
        raise ValueError(
            "Cannot resolve the penultimate auxiliary encoder stage with "
            f"{aux_head.in_channels} channels from runtime features {feature_channels}"
        )


    def interpolate_map(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """
        Interpolate the input tensor to the given size.
        Args:
            x (torch.Tensor): Input tensor.
            size (tuple): Target size (H, W).
        Returns:
            torch.Tensor: Interpolated tensor.
        """
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


    def modality_dropout(
        self,
        feature_maps: dict[str, list[torch.Tensor]],
        modalities_dropout_dict: dict[str, float]
    ) -> dict[str, list[torch.Tensor]]:
        """
        Apply modality dropout to feature maps based on the configuration.
        Args:
            feature_maps (dict): Dictionary of feature maps to be aligned.
            modalities_dropout_dict (dict): Dictionary containing dropout probabilities for each modality.
        Returns:
            dict: Updated feature maps with selected modalities replaced by zeros.
        """
        candidates = [
            key for key, probability in modalities_dropout_dict.items()
            if probability > 0 and key in feature_maps
        ]
        dropped = [
            key for key in candidates
            if torch.rand((), device=feature_maps[key][0].device).item()
            < modalities_dropout_dict[key]
        ]

        if len(dropped) == len(feature_maps) and dropped:
            dropped.pop()
        for key in dropped:
            feature_maps[key] = [torch.zeros_like(tensor) for tensor in feature_maps[key]]
        return feature_maps


    def forward(self, batch: dict, apply_mod_dropout: bool = False) -> dict:
        """
        Forward pass through the model.
        Args:
            batch (dict): Input batch.
            apply_mod_dropout (bool): Whether to apply modality dropout (default is False).
        Returns:
            dict: Output logits for tasks and auxiliary losses.
        """

        fmaps, logits_tasks, logits_aux = {}, {}, {}

        active_mono_keys = [key for key in self.mono_keys if key in self.encoders]
        active_multi_keys = [key for key in self.multi_keys if key in self.encoders]
        reference = batch.get(self.config['labels'][0])
        if reference is None:
            reference = max(
                (batch[mod] for mod in self.encoders),
                key=lambda value: value.shape[-2] * value.shape[-1],
            )
        img_size = reference.shape[-2:]

        for mod, encoder in self.encoders.items():
            if mod in self.mono_keys:

                fmaps[mod] = encoder.seg_model(batch[mod])


                if self.aux_losses.get(mod):
                    for task in self.config['labels']:
                        aux_key = f"{mod}__{task}"
                        if aux_key in self.aux_decoders:
                            aux_decoder = self.aux_decoders[aux_key]
                            aux_feature = self._select_auxiliary_feature(
                                fmaps[mod], aux_decoder
                            )
                            logits_aux[f'aux_{mod}_{task}'] = self.interpolate_map(
                                aux_decoder(aux_feature),
                                batch[task].shape[-2:] if task in batch else img_size
                            )

            else:

                logits, fmaps_ = encoder(batch[mod], batch_positions=batch.get(mod.replace('TS', 'DATES')))
                logits = self.interpolate_map(logits, img_size)

                logits_tasks[mod] = self.interpolate_map(logits, img_size)
                fmaps[mod] = fmaps_


                if self.aux_losses.get(mod):
                    for task in self.config['labels']:
                        aux_key = f"{mod}__{task}"
                        if aux_key in self.aux_decoders:
                            aux_decoder = self.aux_decoders[aux_key]
                            logits_aux[f'aux_{mod}_{task}'] = self.interpolate_map(
                                aux_decoder(logits),
                                batch[task].shape[-2:] if task in batch else img_size
                            )

        if apply_mod_dropout and len(self.encoders) > 1 and active_mono_keys:
            modalities_dropout_dict = {
                key: float(
                    self.config['modalities']['modality_dropout'].get(key, 0.0)
                )
                for key in fmaps
            }
            fmaps = self.modality_dropout(fmaps, modalities_dropout_dict)

        if active_mono_keys:
            fused_features = self.fusion_handler(fmaps, fmaps[active_mono_keys[0]])
        else:
            if apply_mod_dropout and len(self.encoders) > 1:

                wrapped = {key: [value] for key, value in logits_tasks.items()}
                wrapped = self.modality_dropout(
                    wrapped, self.config['modalities']['modality_dropout'])
                logits_tasks = {key: value[0] for key, value in wrapped.items()}
            fused_features = self.fusion_handler(logits_tasks, logits_tasks[active_multi_keys[0]])

        for task in self.config['labels']:
            if active_mono_keys:
                logits_tasks[task] = self.interpolate_map(
                    self.main_decoders[task].seg_model(*fused_features),
                    batch[task].shape[-2:] if task in batch else img_size
                )
            else:
                if len(self.config['labels']) > 1:
                    logits_tasks[task] = self.interpolate_map(
                        self.main_decoders[task](fused_features),
                        batch[task].shape[-2:] if task in batch else img_size)
                else:
                    logits_tasks[task] = fused_features

        for mod in list(logits_tasks.keys()):
            if mod in self.config['modalities']['inputs']:
                del logits_tasks[mod]

        return logits_tasks, logits_aux


class WindowCrossAttention2d(nn.Module):
    """Local cross-attention from aerial queries to temporal context."""

    def __init__(
        self,
        aerial_channels: int,
        context_channels: int,
        attention_channels: int = 128,
        attention_heads: int = 4,
        window_size: int = 8,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if attention_channels % attention_heads != 0:
            raise ValueError(
                "attention_channels must be divisible by attention_heads"
            )
        self.window_size = int(window_size)
        self.query_projection = nn.Sequential(
            nn.Conv2d(aerial_channels, attention_channels, 1, bias=False),
            nn.GroupNorm(1, attention_channels),
        )
        self.key_projection = nn.Sequential(
            nn.Conv2d(context_channels, attention_channels, 1, bias=False),
            nn.GroupNorm(1, attention_channels),
        )
        self.value_projection = nn.Conv2d(
            context_channels, attention_channels, 1, bias=False
        )
        self.attention = nn.MultiheadAttention(
            attention_channels,
            attention_heads,
            batch_first=True,
        )
        self.output_projection = nn.Conv2d(
            attention_channels, aerial_channels, 1, bias=False
        )
        hidden_channels = min(64, max(16, aerial_channels // 8))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * aerial_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_init)

    def _to_windows(self, tensor: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        batch, channels, height, width = tensor.shape
        window = self.window_size
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        padded_h, padded_w = tensor.shape[-2:]
        tensor = tensor.view(
            batch,
            channels,
            padded_h // window,
            window,
            padded_w // window,
            window,
        )
        tensor = tensor.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = tensor.view(-1, window * window, channels)
        return windows, (batch, height, width, padded_h, padded_w)

    def _from_windows(
        self,
        windows: torch.Tensor,
        shape: tuple,
    ) -> torch.Tensor:
        batch, height, width, padded_h, padded_w = shape
        window = self.window_size
        channels = windows.shape[-1]
        tensor = windows.view(
            batch,
            padded_h // window,
            padded_w // window,
            window,
            window,
            channels,
        )
        tensor = tensor.permute(0, 5, 1, 3, 2, 4).contiguous()
        tensor = tensor.view(batch, channels, padded_h, padded_w)
        return tensor[:, :, :height, :width]

    def forward(
        self,
        aerial_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        queries, shape = self._to_windows(
            self.query_projection(aerial_feature)
        )
        keys, _ = self._to_windows(self.key_projection(context_feature))
        values, _ = self._to_windows(self.value_projection(context_feature))
        attended, _ = self.attention(
            queries,
            keys,
            values,
            need_weights=False,
        )
        update = self.output_projection(self._from_windows(attended, shape))
        gate = torch.sigmoid(
            self.gate(torch.cat([aerial_feature, update], dim=1))
        )
        return aerial_feature + gate * update


class FusionHandler(nn.Module):
    def __init__(
        self,
        backbones_channels: list[int],
        target_fused_channels: list[int],
        modality_channels: dict[str, list[int]],
        mono_keys: list[str],
        multi_keys: list[str],
        active_modality_count: int = 2,
        method: str = 'gated_residual',
        levels: list[int] | None = None,
        gate_init: float = -2.0,
        attention_channels: int = 128,
        attention_heads: int = 4,
        window_size: int = 8,
    ) -> None:
        """Fuse temporally aggregated features with the aerial encoder pyramid.

        Levels are indexed over the four encoder stages consumed by UPerNet
        (0=/4, 1=/8, 2=/16, 3=/32 for Swin).  Any raw-input prefix exposed by
        the image encoder is retained unchanged.
        """
        super().__init__()

        if method not in {
            'aligned_concat',
            'gated_residual',
            'window_cross_attention',
        }:
            raise ValueError(
                "fusion method must be aligned_concat, gated_residual or "
                "window_cross_attention"
            )

        self.mono_keys = mono_keys
        self.multi_keys = multi_keys
        self.method = method
        self.modality_channels = {
            key: list(channels) for key, channels in modality_channels.items()
        }

        if not any(key in mono_keys for key in modality_channels):
            self.target_channels = []
            self.stage_count = 0
            self.levels = set()
            return
        self.target_channels = FLAIR_HUB_Model._usable_encoder_channels(
            target_fused_channels
        )[-4:]
        self.stage_count = len(self.target_channels)
        self.levels = set(range(self.stage_count) if levels is None else levels)
        invalid_levels = sorted(
            level for level in self.levels
            if level < 0 or level >= self.stage_count
        )
        if invalid_levels:
            raise ValueError(
                f"fusion levels {invalid_levels} are outside 0..{self.stage_count - 1}"
            )

        for modality, channels in self.modality_channels.items():
            if len(channels) != self.stage_count:
                raise ValueError(
                    f"{modality} exposes {len(channels)} fusion stages, expected "
                    f"{self.stage_count}: {channels}"
                )

        self.concat_projections = nn.ModuleDict()
        self.context_projections = nn.ModuleDict()
        self.context_norms = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.cross_attention = nn.ModuleDict()

        if active_modality_count <= 1:
            return

        if self.method == 'aligned_concat':
            for level in sorted(self.levels):
                in_channels = sum(
                    channels[level]
                    for channels in self.modality_channels.values()
                )
                self.concat_projections[str(level)] = nn.Conv2d(
                    in_channels,
                    self.target_channels[level],
                    kernel_size=1,
                )
            return

        primary_mono = next(
            (key for key in self.modality_channels if key in self.mono_keys),
            None,
        )
        if primary_mono is None:
            return

        for modality, channels in self.modality_channels.items():
            if modality == primary_mono:
                continue
            for level in sorted(self.levels):
                out_channels = self.target_channels[level]
                module_key = f'{modality}__{level}'
                if self.method == 'window_cross_attention':
                    self.cross_attention[module_key] = WindowCrossAttention2d(
                        aerial_channels=out_channels,
                        context_channels=channels[level],
                        attention_channels=attention_channels,
                        attention_heads=attention_heads,
                        window_size=window_size,
                        gate_init=gate_init,
                    )
                    continue
                self.context_projections[module_key] = nn.Conv2d(
                    channels[level], out_channels, kernel_size=1, bias=False
                )
                groups = math.gcd(out_channels, 32)
                self.context_norms[module_key] = nn.GroupNorm(
                    max(groups, 1), out_channels
                )
                hidden_channels = min(64, max(16, out_channels // 8))
                gate = nn.Sequential(
                    nn.Conv2d(
                        2 * out_channels,
                        hidden_channels,
                        kernel_size=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, 1, kernel_size=1),
                )
                nn.init.zeros_(gate[-1].weight)
                nn.init.constant_(gate[-1].bias, gate_init)
                self.gates[module_key] = gate

    def _usable_maps(
        self,
        feature_maps: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        usable = [
            feature_map
            for feature_map in feature_maps
            if feature_map.ndim == 4 and feature_map.shape[1] > 0
        ]
        if len(usable) < self.stage_count:
            raise ValueError(
                f"Fusion expected at least {self.stage_count} feature maps, "
                f"received {[tuple(item.shape) for item in usable]}"
            )
        return usable[:-self.stage_count], usable[-self.stage_count:]

    @staticmethod
    def _resize_like(
        feature_map: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if feature_map.shape[-2:] == target.shape[-2:]:
            return feature_map
        return F.interpolate(
            feature_map,
            size=target.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    def forward(self, feature_maps: dict, target_fm_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass for feature map fusion.

        Args:
            feature_maps (dict): Dictionary of feature maps for each modality.
            target_fm_maps (list[torch.Tensor]): List of target feature maps.

        Returns:
            list[torch.Tensor]: Adjusted and fused feature maps.
        """
        active_keys = list(feature_maps.keys())
        mono_key_active = [key for key in active_keys if key in self.mono_keys]
        multi_key_active = [key for key in active_keys if key in self.multi_keys]


        if len(mono_key_active) == 1 and len(multi_key_active) == 0:
            return feature_maps[mono_key_active[0]]


        if len(mono_key_active) == 0 and len(multi_key_active) == 1:
            return feature_maps[multi_key_active[0]]


        if len(mono_key_active) == 0 and len(multi_key_active) > 1:
            stacked_feature_maps = torch.stack(
                [feature_maps[key] for key in multi_key_active], dim=0
            )
            return torch.mean(stacked_feature_maps, dim=0)


        primary_mono = mono_key_active[0]
        prefix, target_stages = self._usable_maps(feature_maps[primary_mono])
        aligned = {}
        for modality in active_keys:
            _, modality_stages = self._usable_maps(feature_maps[modality])
            aligned[modality] = [
                self._resize_like(feature_map, target)
                for feature_map, target in zip(modality_stages, target_stages)
            ]

        fused_stages = []
        for level, aerial_feature in enumerate(aligned[primary_mono]):
            if level not in self.levels:
                fused_stages.append(aerial_feature)
                continue

            if self.method == 'aligned_concat':
                concatenated = torch.cat(
                    [aligned[modality][level] for modality in active_keys],
                    dim=1,
                )
                fused_stages.append(
                    self.concat_projections[str(level)](concatenated)
                )
                continue

            fused = aerial_feature
            for modality in active_keys:
                if modality == primary_mono:
                    continue
                module_key = f'{modality}__{level}'
                if self.method == 'window_cross_attention':
                    fused = self.cross_attention[module_key](
                        fused,
                        aligned[modality][level],
                    )
                    continue
                context = self.context_norms[module_key](
                    self.context_projections[module_key](aligned[modality][level])
                )
                gate = torch.sigmoid(
                    self.gates[module_key](torch.cat([aerial_feature, context], dim=1))
                )
                fused = fused + gate * context
            fused_stages.append(fused)

        return prefix + fused_stages


