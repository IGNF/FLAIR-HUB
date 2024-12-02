import torch
import segmentation_models_pytorch as smp

from typing import Dict, Any
from transformers import AutoModelForSemanticSegmentation, AutoConfig



class FLAIR_MonoTemporal(torch.nn.Module):
    
    def __init__(self, 
                 config: Dict[str, Any],
                 channels: int = 3,
                 classes: int = 19,
                 return_backbone_only: bool = False
                ) -> None:
        """
        Initialize the FLAIR_MonoTemporal model with the provided configuration.
        Args:
            config (Dict[str, Any]): The configuration dictionary for the model.
            channels (int, optional): The number of input channels. Default is 3 (RGB).
            classes (int, optional): The number of output classes. Default is 19.
            return_backbone_only (bool, optional): If True, returns only the backbone of the model. Default is False.
        """
        super(FLAIR_MonoTemporal, self).__init__()

        n_channels = channels
        n_classes = classes
        self.return_backbone_only = return_backbone_only

        ##### loading architecture from segmentation_models_pytorch
        if config['models']['monotemp_model']['arch'].split('_')[0] == 'smp':
            encoder, decoder = config['models']['monotemp_model']['arch'].split('_')[1].split('-')[0], config['models']['monotemp_model']['arch'].split('_')[1].split('-')[1]
            self.seg_model = smp.create_model(
                arch=decoder, 
                encoder_name=encoder, 
                classes=n_classes, 
                in_channels=n_channels
            )  
            self.cfg_model = None                  

        ##### loading architecture from HuggingFace transformers AutoModelForSemanticSegmentation
        elif config['models']['monotemp_model']['arch'].split('_')[0] == 'hf':  
            self.cfg_model = AutoConfig.from_pretrained(
                config['models']['monotemp_model']['arch'].split('_')[1], 
                num_labels=n_classes
            )
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(
                config['models']['monotemp_model']['arch'].split('_')[1], 
                config=self.cfg_model, 
                ignore_mismatched_sizes=True
            )
            if n_channels != 3:
                self.cfg_model.backbone_config.num_channels = n_channels
                self.seg_model.config.backbone_config.num_channels = n_channels
                self.seg_model = self.update_model_channels(
                    self.seg_model, 
                    num_channels=n_channels, 
                    init_mode=config['models']['monotemp_model']['new_channels_init_mode']
                )                          
        else:
            raise ValueError('Wrong monotemp_model provided. Cannot retrieve model. See readme.')

        if self.return_backbone_only:
            if config['models']['monotemp_model']['arch'].split('_')[0] == 'smp':
                self.seg_model = self.seg_model.encoder
            elif config['models']['monotemp_model']['arch'].split('_')[0] == 'hf': 
                self.seg_model = self.seg_model.backbone


    @staticmethod
    def update_model_channels(model: torch.nn.Module, num_channels: int = 4, init_mode: str = 'third') -> torch.nn.Module:
        """
        Updates the first convolution layer of the encoder to match the specified number of input channels.
        Args:
            model (torch.nn.Module): The model whose encoder needs to be updated.
            num_channels (int, optional): The desired number of input channels. Default is 4.
            init_mode (str, optional): The initialization mode for new channels. Default is 'third'.
        Returns:
            torch.nn.Module: The updated model with modified channels.
        """
        print(f'--- Modifying Swin encoder 1st conv channel number to {num_channels}')
        model.config.backbone_config.num_channels = num_channels
        backbone = model.backbone
        conv1 = backbone.embeddings.patch_embeddings.projection

        new_conv1 = torch.nn.Conv2d(
            in_channels=num_channels, 
            out_channels=conv1.out_channels, 
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None
        )

        with torch.no_grad():
            if num_channels <= conv1.weight.shape[1]:  # Reducing channels
                if num_channels == 1:  # Average the input RGB channels
                    new_conv1.weight.data[:, 0, :, :] = conv1.weight.data.mean(dim=1)
                elif num_channels == 2:  # Drop the third channel (if present)
                    new_conv1.weight.data[:, :2, :, :] = conv1.weight.data[:, :2, :, :]
                else:
                    new_conv1.weight.data[:, :num_channels, :, :] = conv1.weight.data[:, :num_channels, :, :]
            else:  # Adding channels
                new_conv1.weight.data[:, :conv1.weight.shape[1], :, :] = conv1.weight.data

                if num_channels > conv1.weight.shape[1]:
                    if init_mode == 'zeros':
                        new_conv1.weight.data[:, conv1.weight.shape[1]:, :, :] = torch.zeros_like(new_conv1.weight.data[:, conv1.weight.shape[1]:, :, :])
                    elif init_mode == 'random':
                        new_conv1.weight.data[:, conv1.weight.shape[1]:, :, :] = torch.randn_like(new_conv1.weight.data[:, conv1.weight.shape[1]:, :, :])
                    elif init_mode == 'first':
                        new_conv1.weight.data[:, conv1.weight.shape[1]:, :, :] = conv1.weight.data[:, 0:1, :, :].expand(-1, num_channels - conv1.weight.shape[1], -1, -1)
                    elif init_mode == 'second':
                        new_conv1.weight.data[:, conv1.weight.shape[1]:, :, :] = conv1.weight.data[:, 1:2, :, :].expand(-1, num_channels - conv1.weight.shape[1], -1, -1)
                    elif init_mode == 'third':
                        if conv1.weight.shape[1] >= 3:  
                            new_conv1.weight.data[:, conv1.weight.shape[1]:, :, :] = conv1.weight.data[:, 2:3, :, :].expand(-1, num_channels - conv1.weight.shape[1], -1, -1)
                        else:
                            raise ValueError(f"Cannot use 'third' as init_mode since the original conv layer has less than 3 channels.")
                    else:
                        raise ValueError(f"Unsupported init_mode: {init_mode}. Choose from 'zeros', 'random', 'first', 'second', 'third'.")

        backbone.embeddings.patch_embeddings.projection = new_conv1
        patch_embeddings = backbone.embeddings.patch_embeddings
        patch_embeddings.num_channels = num_channels
        return model

