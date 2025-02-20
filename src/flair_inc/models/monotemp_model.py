import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Any


class DecoderWrapper(nn.Module):
    """Handles sequential execution of the decoder and segmentation head."""
    def __init__(self, decoder, segmentation_head):
        super().__init__()
        self.decoder = decoder
        self.segmentation_head = segmentation_head

    def forward(self, *features):
        decoder_output = self.decoder(*features)  
        return self.segmentation_head(decoder_output)  


class FLAIR_MonoTemporal(nn.Module):
    
    def __init__(self, 
                 config: Dict[str, Any],
                 channels: int = 3,
                 classes: int = 19,
                 img_size: int = 512,
                 return_type: str = 'encoder',
                ) -> None:
        """
        Initialize the FLAIR_MonoTemporal model with the provided configuration.
        """
        super(FLAIR_MonoTemporal, self).__init__()

        n_channels = channels
        n_classes = classes
        self.return_type = return_type
        
        assert self.return_type in ['encoder', 'decoder', 'classifier'], \
            'return_type should be one of ["encoder", "decoder", "classifier"]'

        encoder, decoder = config['models']['monotemp_model']['arch'].split('-')[0], config['models']['monotemp_model']['arch'].split('-')[1]
        
        try: 
            self.seg_model = smp.create_model(
                arch=decoder, 
                encoder_name=encoder, 
                classes=n_classes, 
                in_channels=n_channels,
                img_size=img_size,
            )
        except KeyError: 
            self.seg_model = smp.create_model(
                arch=decoder, 
                encoder_name='tu-'+encoder, 
                classes=n_classes, 
                in_channels=n_channels,
                img_size=img_size,
            )            

        if self.return_type == 'encoder':
            self.seg_model = self.seg_model.encoder

        elif self.return_type == 'decoder':
            self.seg_model = DecoderWrapper(self.seg_model.decoder, self.seg_model.segmentation_head)    

