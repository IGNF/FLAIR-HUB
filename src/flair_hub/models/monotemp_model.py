import torch.nn as nn
import segmentation_models_pytorch as smp
import timm

from typing import Dict, Any


class TimmFeatureEncoder(nn.Module):
    """Minimal timm feature encoder with explicit stochastic-depth support."""

    def __init__(
        self,
        name: str,
        in_channels: int,
        pretrained: bool,
        drop_path_rate: float | None,
        img_size: int,
    ) -> None:
        super().__init__()
        kwargs = {
            "in_chans": in_channels,
            "features_only": True,
            "pretrained": pretrained,
            "img_size": img_size,
        }
        if drop_path_rate is not None:
            kwargs["drop_path_rate"] = drop_path_rate
        self.model = timm.create_model(name.removeprefix("tu-"), **kwargs)
        self.out_channels = [in_channels] + list(self.model.feature_info.channels())

    def forward(self, inputs):
        features = []
        for feature, channels in zip(self.model(inputs), self.out_channels[1:]):

            if (
                feature.ndim == 4
                and feature.shape[-1] == channels
                and feature.shape[1] != channels
            ):
                feature = feature.permute(0, 3, 1, 2).contiguous()
            features.append(feature)
        return [inputs] + features


class DecoderWrapper(nn.Module):
    """Handles sequential execution of the decoder and segmentation head."""

    def __init__(self, decoder: nn.Module, segmentation_head: nn.Module) -> None:
        """
        Initialize the DecoderWrapper with a decoder and a segmentation head.

        Args:
            decoder (nn.Module): The decoder module.
            segmentation_head (nn.Module): The segmentation head module.
        """
        super().__init__()
        self.decoder = decoder
        self.segmentation_head = segmentation_head

    def forward(self, *features: Any) -> nn.Module:
        """
        Perform a forward pass through the decoder and segmentation head.
        Args:
            *features (Any): Feature maps to be passed to the decoder.
        Returns:
            nn.Module: Output after passing through the segmentation head.
        """
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)


class FLAIR_Monotemp(nn.Module):
    """Monotemporal FLAIR model for segmentation."""

    def __init__(
        self,
        config: Dict[str, Any],
        channels: int = 3,
        classes: int = 19,
        img_size: int = 512,
        return_type: str = 'encoder'
    ) -> None:
        """
        Initialize the FLAIR_Monotemp model with the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for model setup.
            channels (int): Number of input channels (default is 3 for RGB).
            classes (int): Number of output classes for segmentation.
            img_size (int): Image size for model input.
            return_type (str): Specifies the model return type ('encoder', 'decoder').

        Raises:
            AssertionError: If `return_type` is not one of ['encoder', 'decoder'].
        """
        super().__init__()

        self.return_type = return_type
        assert self.return_type in ['encoder', 'decoder'],\
            'return_type should be one of ["encoder", "decoder"]'

        monotemp_config = config['models']['monotemp_model']
        encoder, decoder = monotemp_config['arch'].rsplit('-', 1)
        encoder_config = monotemp_config.get('encoder', {})
        encoder_weights = encoder_config.get('weights', 'imagenet')

        upernet_config = monotemp_config.get('upernet_decoder', {})
        custom_upernet_encoder = (
            self.return_type == 'encoder'
            and decoder.lower() == 'upernet'
            and str(upernet_config.get('implementation', 'smp')).lower()
            in {'flair_hub', 'custom'}
        )
        if custom_upernet_encoder:
            smp_model = None
            for encoder_name in (encoder, 'tu-' + encoder):
                kwargs = {
                    "arch": "upernet",
                    "encoder_name": encoder_name,
                    "classes": classes,
                    "in_channels": channels,
                    "img_size": img_size,
                    "encoder_weights": encoder_weights,
                }
                if encoder_config.get('drop_path_rate') is not None:
                    kwargs['drop_path_rate'] = float(
                        encoder_config['drop_path_rate']
                    )
                try:
                    smp_model = smp.create_model(**kwargs)
                    break
                except (KeyError, TypeError):
                    continue
            if smp_model is not None:
                self.seg_model = smp_model.encoder
            else:
                self.seg_model = TimmFeatureEncoder(
                    encoder,
                    in_channels=channels,
                    pretrained=encoder_weights is not None,
                    drop_path_rate=encoder_config.get('drop_path_rate'),
                    img_size=img_size,
                )
            return

        last_error = None
        for encoder_name in (encoder, 'tu-' + encoder):
            for include_img_size in (True, False):
                kwargs = {
                    "arch": decoder,
                    "encoder_name": encoder_name,
                    "classes": classes,
                    "in_channels": channels,
                    "encoder_weights": encoder_weights,
                }
                if include_img_size:
                    kwargs["img_size"] = img_size
                try:
                    self.seg_model = smp.create_model(**kwargs)
                    last_error = None
                    break
                except (KeyError, TypeError) as exc:
                    last_error = exc
            if last_error is None:
                break
        if last_error is not None:
            raise last_error

        if self.return_type == 'encoder':
            self.seg_model = self.seg_model.encoder
        elif self.return_type == 'decoder':
            self.seg_model = DecoderWrapper(self.seg_model.decoder, self.seg_model.segmentation_head)
