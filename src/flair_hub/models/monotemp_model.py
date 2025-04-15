import torch.nn as nn
import segmentation_models_pytorch as smp

from typing import Dict, Any


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
        assert self.return_type in ['encoder', 'decoder'], \
            'return_type should be one of ["encoder", "decoder"]'

        encoder, decoder = config['models']['monotemp_model']['arch'].split('-')[0], \
                            config['models']['monotemp_model']['arch'].split('-')[1]

        try:
            self.seg_model = smp.create_model(
                arch=decoder,
                encoder_name=encoder,
                classes=classes,
                in_channels=channels,
                img_size=img_size,
            )
        except KeyError:
            self.seg_model = smp.create_model(
                arch=decoder,
                encoder_name='tu-' + encoder,  # timm models of smp 
                classes=classes,
                in_channels=channels,
                img_size=img_size,
            )

        if self.return_type == 'encoder':
            self.seg_model = self.seg_model.encoder
        elif self.return_type == 'decoder':
            self.seg_model = DecoderWrapper(self.seg_model.decoder, self.seg_model.segmentation_head)
