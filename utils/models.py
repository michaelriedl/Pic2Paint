# -*- coding: utf-8 -*-
"""
Neural transfer network definitions

Author: Michael Riedl
Last Updated: December 8, 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def gram_matrix(input):
    """ Function for calculating the normalized Gram matrix from layer activations  
        
    Parameters
    ----------
    input : torch.Tensor
        The layer activation tensor
    
    Returns
    -------
    torch.Tensor
        A tensor containing the Gram matrix
        
    Notes
    -----
    This code was modified from: 
        https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    """
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)

class Normalization(nn.Module):
    """ A neural module that takes an input image tensor and normalizes it for
        the pretrained network
    
    Extended Summary
    ----------------
    A :class:`Normalization` module takes an input image tensor and normalizes it 
    for the pretrained neural network. This module is placed in front of the 
    pretrained network so that the input data does not need to be preprocessed.
    
    Attributes
    ----------
    mean : torch.Tensor
        A tensor of means for each channel
        
    std : torch.Tensor
        A tensor of standard deviations for each channel
        
    Notes
    -----
    The means and variances should be moved to the desired device before 
    initializing the module.
    This code was modified from: 
        https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class StyleTransferNet(nn.Module):
    """ A neural module that implements neural style transfer using the pretrained
        VGG19 network. The implementation follows that of Gatys et. al.
    
    Extended Summary
    ----------------
    A :class:`StyleTransferNet` transfers the sytle of the style image to the
    content image. The inputs are the style image, the style loss weight, the
    content image, the content loss weight, and the device to be used for
    training.
    
    Attributes
    ----------
    device : str
        The device to use for training.
        
    style_img : torch.Tensor
        The style image to be used to transfer style to the content image.
    
    content_img : torch.Tensor
        The content image to be re-styled/
    
    style_weight : float
        The weight applied to the style loss.
    
    content_weight : float
        The weight applied to the content loss.
        
    Notes
    -----
    This module was designed to minimize the memory overhead required when
    training on a GPU.
    This code was modified from: 
        https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    """
    def __init__(self, device, style_img, content_img, style_weight=1e6, content_weight=1):
        super(StyleTransferNet, self).__init__()
        # Store the device to run the network
        self.device = device
        # Move the style and content images to the device
        style_img = style_img.requires_grad_(False).to(self.device)
        content_img = content_img.requires_grad_(False).to(self.device)
        # Store the weights
        self.style_weight = style_weight
        self.content_weight = content_weight
        # Set the default layers to use (CHANGING REQUIRES MODIFICATION OF FORWARD PASS)
        max_conv_layer = 5
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        # Set the normalization parameters for VGG19
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        # Store the needed VGG19 layers
        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.style_loss_dict = dict()
        self.content_loss_dict = dict()
        cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.normalization = Normalization(self.norm_mean, self.norm_std)
        model = nn.Sequential(self.normalization)
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
                self.conv_layers.append(layer)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
                self.relu_layers.append(layer)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
                self.pool_layers.append(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))   
            model.add_module(name, layer.requires_grad_(False))
            # Add content loss
            if name in content_layers:        
                self.content_loss_dict[name] = model(content_img).detach()
            # Add style loss
            if name in style_layers:
                self.style_loss_dict[name] = model(style_img).detach()     
            # Stop after the 5th conv layer
            if(i == max_conv_layer):
                break
        
    def forward(self, img):
        x = self.normalization(img)
        x = self.conv_layers[0](x)
        # Style loss
        self.loss = self.style_weight*F.mse_loss(gram_matrix(x), gram_matrix(self.style_loss_dict['conv_1']))
        x = self.relu_layers[0](x)
        x = self.conv_layers[1](x)
        # Style loss
        self.loss += self.style_weight*F.mse_loss(gram_matrix(x), gram_matrix(self.style_loss_dict['conv_2']))
        x = self.relu_layers[1](x)
        x = self.pool_layers[0](x)
        x = self.conv_layers[2](x)
        # Style loss
        self.loss += self.style_weight*F.mse_loss(gram_matrix(x), gram_matrix(self.style_loss_dict['conv_3']))
        x = self.relu_layers[2](x)
        x = self.conv_layers[3](x)
        # Style loss
        self.loss += self.style_weight*F.mse_loss(gram_matrix(x), gram_matrix(self.style_loss_dict['conv_4']))
        # Content loss
        self.loss += self.content_weight*F.mse_loss(x, self.content_loss_dict['conv_4'])
        x = self.relu_layers[3](x)
        x = self.pool_layers[1](x)
        x = self.conv_layers[4](x)
        # Style loss
        self.loss += self.style_weight*F.mse_loss(gram_matrix(x), gram_matrix(self.style_loss_dict['conv_5']))
        
        return x