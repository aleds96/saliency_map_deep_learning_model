import torch
import torch.nn as nn
import torchvision.models as models
import copy
class FixationPredictorV1(nn.Module):
    def __init__(self, backbone='resnet18',is_multi_task=False):
        super(FixationPredictorV1, self).__init__()
        self.is_multi_task=is_multi_task
        self.model_name = 'v1'
        if backbone == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(base.children())[:-2])  # remove avgpool & fc
            in_channels = 512
        else:
            raise ValueError("Unsupported backbone")
        #decoder input shape: [1, 512, 7, 7])
        self.decoder_first_map= nn.Sequential(
            #in: [1, 512, 7, 7]) out: torch.Size([1, 128, 7, 7])
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            #out: torch.Size([1, 128, 14, 14])
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # → [B, 64, 28, 28]
            nn.ReLU(inplace=True),
            #out: torch.Size([1, 64, 112, 112])
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            #nn.Dropout2d(p=0.3),
        )
            # #out: torch.Size([1, 64, 224, 224])
            #nn.ConvTranspose2d(16, 1,kernel_size=2, stride=2) ,
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            #nn.Conv2d(16, 1, kernel_size=1), )
        self.first_map_head =  nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        )
        #nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        if self.is_multi_task: 
            self.second_map_head = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        )
        else: 
            self.second_map_head = None
        
    def forward(self, x):
        x = self.encoder(x)
        #x shape=>[1, 512, 7, 7])
        #print('x shape=>',x.shape)
        first_decoded_map_out = self.decoder_first_map(x)
        first_map_out = self.first_map_head(first_decoded_map_out)
        if self.is_multi_task==False: 
            return first_map_out,None 
        second_map_out = self.second_map_head(first_decoded_map_out)
        return first_map_out, second_map_out
        
class FixationPredictorV2(nn.Module):
    def __init__(self, backbone='resnet18',is_multi_task=False):
        super(FixationPredictorV2, self).__init__()
        self.is_multi_task=is_multi_task
        self.model_name = 'v2'
        if backbone == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(base.children())[:-2])  # remove avgpool & fc
            in_channels = 512
        else:
            raise ValueError("Unsupported backbone")
        #decoder input shape: [1, 512, 7, 7])
        self.decoder_first_map= nn.Sequential(
            #in: [1, 512, 7, 7]) out: torch.Size([1, 128, 7, 7])
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            #out: torch.Size([1, 128, 14, 14])
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # → [B, 64, 28, 28]
            nn.ReLU(inplace=True),
            #out: torch.Size([1, 64, 112, 112])
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.3),
        )
            # #out: torch.Size([1, 64, 224, 224])
            #nn.ConvTranspose2d(16, 1,kernel_size=2, stride=2) ,
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            #nn.Conv2d(16, 1, kernel_size=1), )
        self.first_map_head =  nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        #nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        if self.is_multi_task: 
            self.second_map_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        else: 
            self.second_map_head = None
        
    def forward(self, x):
        x = self.encoder(x)
        #x shape=>[1, 512, 7, 7])
        #print('x shape=>',x.shape)
        first_decoded_map_out = self.decoder_first_map(x)
        first_map_out = self.first_map_head(first_decoded_map_out)
        if self.is_multi_task==False: 
            return first_map_out,None 
        second_map_out = self.second_map_head(first_decoded_map_out)
        return first_map_out, second_map_out  
class FixationPredictorV3(nn.Module):
    def __init__(self, backbone='resnet18',is_multi_task=False):
        super(FixationPredictorV3, self).__init__()
        self.is_multi_task=is_multi_task
        self.model_name = 'v3'
        if backbone == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(base.children())[:-2])  # remove avgpool & fc
            in_channels = 512
        else:
            raise ValueError("Unsupported backbone")
        #decoder input shape: [1, 512, 7, 7])
        self.decoder_first_map= nn.Sequential(
            #in: [1, 512, 7, 7]) out: torch.Size([1, 128, 7, 7])
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.3),
            #out: torch.Size([1, 128, 14, 14])
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # → [B, 64, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #out: torch.Size([1, 64, 112, 112])
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.3),
        )
            # #out: torch.Size([1, 64, 224, 224])
            #nn.ConvTranspose2d(16, 1,kernel_size=2, stride=2) ,
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            #nn.Conv2d(16, 1, kernel_size=1), )
        self.first_map_head =  nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        #nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        if self.is_multi_task: 
            self.second_map_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        else: 
            self.second_map_head = None
        
    def forward(self, x):
        x = self.encoder(x)
        #x shape=>[1, 512, 7, 7])
        #print('x shape=>',x.shape)
        first_decoded_map_out = self.decoder_first_map(x)
        first_map_out = self.first_map_head(first_decoded_map_out)
        if self.is_multi_task==False: 
            return first_map_out,None 
        second_map_out = self.second_map_head(first_decoded_map_out)
        return first_map_out, second_map_out  
#definisci loss custom 
class MultiTaskLoss(nn.Module):
    def __init__(self, weight_binary=1.0, weight_continuous=1.0,binary_loss_pos_weight=1,continuous_loss_pos_weight=1):
        super().__init__()
        self.weight_binary = weight_binary
        self.weight_continuous = weight_continuous
        self.binary_loss_pos_weight =binary_loss_pos_weight
        self.continuous_loss_pos_weight = continuous_loss_pos_weight
        self.binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([binary_loss_pos_weight]))
        
        self.continuous_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([continuous_loss_pos_weight]))

    def forward(self, binary_pred=None, binary_target=None,
                      continuous_pred=None, continuous_target=None):
        total_loss = 0.0
        USE_BOTH_LOSS = False  if binary_pred is None or continuous_pred is None else True 
        #print('USE BOTH LOSS =>', USE_BOTH_LOSS)
        if binary_pred is not None and binary_target is not None:
            binary_loss = self.binary_loss_fn(binary_pred, binary_target)
            if USE_BOTH_LOSS: 
                total_loss += self.weight_binary  * binary_loss
                
            else:
                total_loss += 1 * binary_loss
                

        if continuous_pred is not None and continuous_target is not None:
            continuous_loss = self.continuous_loss_fn(continuous_pred, continuous_target)
            if USE_BOTH_LOSS:
                total_loss += self.weight_continuous * continuous_loss
            else:
                total_loss += 1 * continuous_loss
        return total_loss