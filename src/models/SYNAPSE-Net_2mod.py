from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .blocks import (
    ConvBnGELU, ResidualBlock, CBAM, LesionGate, EncoderStage, 
    SwinTransformerLayer, FusionBottleneck, UNetPPDecoder5
)

class CustomEncoder5(nn.Module):
    def __init__(self, in_channels=1, chs=(64, 96, 128, 192, 256)):
        super().__init__()
        c1,c2,c3,c4,c5 = chs
        self.s1 = EncoderStage(in_channels, c1, conv_kernels=[5,5])
        self.p1 = nn.MaxPool2d(2,2)
        self.s2 = EncoderStage(c1, c2, conv_kernels=[3,3])
        self.p2 = nn.MaxPool2d(2,2)
        self.s3 = EncoderStage(c2, c3, conv_kernels=[3,3])
        self.p3 = nn.MaxPool2d(2,2)
        self.s4 = EncoderStage(c3, c4, conv_kernels=[3,3])
        self.p4 = nn.MaxPool2d(2,2)
        self.s5 = EncoderStage(c4, c5, conv_kernels=[3,3])

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(self.p1(f1))
        f3 = self.s3(self.p2(f2))
        f4 = self.s4(self.p3(f3))
        f5 = self.s5(self.p4(f4))
        return [f1,f2,f3,f4,f5]

class UNetPPDecoder5(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5, dropout_rate: float = 0.0):
        super().__init__()
        self.up_c5 = nn.ConvTranspose2d(c5, c5, kernel_size=2, stride=2)
        self.up_c4 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2)
        self.up_c3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)
        self.up_c2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)

        self.x3_1 = ResidualBlock(c4 + c5, c4, dropout_rate=dropout_rate)
        self.x2_1 = ResidualBlock(c3 + c4, c3, dropout_rate=dropout_rate)
        self.x1_1 = ResidualBlock(c2 + c3, c2, dropout_rate=dropout_rate)
        self.x0_1 = ResidualBlock(c1 + c2, c1, dropout_rate=dropout_rate)

        self.x2_2 = ResidualBlock(c3 + c3 + c4, c3, dropout_rate=dropout_rate)
        self.x1_2 = ResidualBlock(c2 + c2 + c3, c2, dropout_rate=dropout_rate)
        self.x0_2 = ResidualBlock(c1 + c1 + c2, c1, dropout_rate=dropout_rate)

        self.x1_3 = ResidualBlock(c2 + c2 + c2 + c3, c2, dropout_rate=dropout_rate)
        self.x0_3 = ResidualBlock(c1 + c1 + c1 + c2, c1, dropout_rate=dropout_rate)

        self.x0_4 = ResidualBlock(c1 + c1 + c1 + c1 + c2, c1, dropout_rate=dropout_rate)
    
    def forward(self, f, center):
        pass

class SYNAPSENet_2mod(nn.Module):
    def __init__(self,
                 chs=(64, 96, 128, 192, 256),
                 token_dim=256,
                 bottleneck_heads=8,
                 bottleneck_window=8,
                 adaptive_tokens_h=13,
                 adaptive_tokens_w=13,
                 aux_outs=2,
                 pre_swin_layers=1,
                 dropout_rate: float = 0.4,
                 swin_mlp_ratio=2.0,
                 drop_path_rate: float = 0.2,
                 **kwargs):
        super().__init__()
        
        self.encA = CustomEncoder5(in_channels=1, chs=chs)
        self.encB = CustomEncoder5(in_channels=1, chs=chs)
        c1, c2, c3, c4, c5 = chs

        self.pre_swinA = nn.ModuleList([
            SwinTransformerLayer(
                dim=c5, num_heads=bottleneck_heads, window_size=bottleneck_window,
                mlp_ratio=swin_mlp_ratio, drop_path=drop_path_rate
            ) for _ in range(pre_swin_layers)
        ])
        self.pre_swinB = nn.ModuleList([
            SwinTransformerLayer(
                dim=c5, num_heads=bottleneck_heads, window_size=bottleneck_window,
                mlp_ratio=swin_mlp_ratio, drop_path=drop_path_rate
            ) for _ in range(pre_swin_layers)
        ])

        self.pool_conv = nn.Conv2d(c5, token_dim, 1)
        self.adaptive_tokens = (adaptive_tokens_h, adaptive_tokens_w)
        self.bottleneck = FusionBottleneck(dim=token_dim, num_heads=bottleneck_heads)
        self.token_to_feat = nn.Conv2d(token_dim * 2, c5, 1)
        self.center_res = ResidualBlock(c5, c5, dropout_rate=dropout_rate)

        self.lesion_gate_f3 = LesionGate(in_channels=c5, feature_channels=c3)
        self.lesion_gate_f2 = LesionGate(in_channels=c4, feature_channels=c2)
        self.lesion_gate_f1 = LesionGate(in_channels=c3, feature_channels=c1)

        self.lesion_head = nn.Sequential(
            ConvBnGELU(c5, c5//8), 
            nn.Conv2d(c5//8, 1, 1)
        )
        
        self._fuse_f1 = nn.Sequential(nn.Conv2d(c1*2, c1, 1, bias=False), nn.BatchNorm2d(c1), CBAM(c1), nn.GELU())
        self._fuse_f2 = nn.Sequential(nn.Conv2d(c2*2, c2, 1, bias=False), nn.BatchNorm2d(c2), CBAM(c2), nn.GELU())
        self._fuse_f3 = nn.Sequential(nn.Conv2d(c3*2, c3, 1, bias=False), nn.BatchNorm2d(c3), CBAM(c3), nn.GELU())
        self._fuse_f4 = nn.Sequential(nn.Conv2d(c4*2, c4, 1, bias=False), nn.BatchNorm2d(c4), CBAM(c4), nn.GELU())
        
        self.decoder = UNetPPDecoder5(c1, c2, c3, c4, c5, dropout_rate=dropout_rate)
        self.aux_outs = aux_outs
        self.head_main = nn.Conv2d(c1, 1, 1)
        if aux_outs >= 1: self.head_aux1 = nn.Conv2d(c1, 1, 1)
        if aux_outs >= 2: self.head_aux2 = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        H_in, W_in = x.shape[2], x.shape[3]
        A = x[:, 0:1, :, :]
        B = x[:, 1:2, :, :]

        f1A, f2A, f3A, f4A, f5A = self.encA(A)
        f1B, f2B, f3B, f4B, f5B = self.encB(B)
        
        f1 = self._fuse_f1(torch.cat([f1A, f1B], 1))
        f2 = self._fuse_f2(torch.cat([f2A, f2B], 1))
        f3 = self._fuse_f3(torch.cat([f3A, f3B], 1))
        f4 = self._fuse_f4(torch.cat([f4A, f4B], 1))

        for layer in self.pre_swinA: f5A = layer(f5A)
        for layer in self.pre_swinB: f5B = layer(f5B)

        Ht, Wt = self.adaptive_tokens
        tokA_2d = self.pool_conv(F.adaptive_avg_pool2d(f5A, (Ht, Wt)))
        tokB_2d = self.pool_conv(F.adaptive_avg_pool2d(f5B, (Ht, Wt)))
        tokA = rearrange(tokA_2d, 'b d h w -> b (h w) d')
        tokB = rearrange(tokB_2d, 'b d h w -> b (h w) d')
        
        enriched_tokA, enriched_tokB = self.bottleneck(tokA, tokB)
        
        enriched_2dA = rearrange(enriched_tokA, 'b (h w) d -> b d h w', h=Ht, w=Wt)
        enriched_2dB = rearrange(enriched_tokB, 'b (h w) d -> b d h w', h=Ht, w=Wt)
        
        fused_2d = torch.cat([enriched_2dA, enriched_2dB], dim=1)
        
        center_feat_c5 = self.token_to_feat(fused_2d)
        center = self.center_res(F.interpolate(center_feat_c5, size=f5A.shape[2:], mode='bilinear', align_corners=False))
        lesion_logits = self.lesion_head(center)
        
        x3_1 = self.decoder.x3_1(torch.cat([f4, self.decoder.up_c5(center)], 1))
        f3_gated = self.lesion_gate_f3(center, f3)
        
        x2_1 = self.decoder.x2_1(torch.cat([f3_gated, self.decoder.up_c4(x3_1)], 1))
        x2_2 = self.decoder.x2_2(torch.cat([f3_gated, x2_1, self.decoder.up_c4(x3_1)], 1))
        f2_gated = self.lesion_gate_f2(x3_1, f2)

        x1_1 = self.decoder.x1_1(torch.cat([f2_gated, self.decoder.up_c3(x2_1)], 1))
        x1_2 = self.decoder.x1_2(torch.cat([f2_gated, x1_1, self.decoder.up_c3(x2_2)], 1))
        x1_3 = self.decoder.x1_3(torch.cat([f2_gated, x1_1, x1_2, self.decoder.up_c3(x2_2)], 1))
        f1_gated = self.lesion_gate_f1(x2_2, f1)

        x0_1 = self.decoder.x0_1(torch.cat([f1_gated, self.decoder.up_c2(x1_1)], 1))
        x0_2 = self.decoder.x0_2(torch.cat([f1_gated, x0_1, self.decoder.up_c2(x1_2)], 1))
        x0_3 = self.decoder.x0_3(torch.cat([f1_gated, x0_1, x0_2, self.decoder.up_c2(x1_3)], 1))
        x0_4 = self.decoder.x0_4(torch.cat([f1_gated, x0_1, x0_2, x0_3, self.decoder.up_c2(x1_3)], 1))
        
        main_logits = F.interpolate(self.head_main(x0_4), size=(H_in,W_in), mode='bilinear', align_corners=False)
        auxs=[]
        if self.aux_outs >= 1: auxs.append(F.interpolate(self.head_aux1(x0_3), size=(H_in,W_in), mode='bilinear', align_corners=False))
        if self.aux_outs >= 2: auxs.append(F.interpolate(self.head_aux2(x0_2), size=(H_in,W_in), mode='bilinear', align_corners=False))
        return main_logits, auxs, lesion_logits
