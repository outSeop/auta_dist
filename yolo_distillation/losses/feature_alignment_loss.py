"""
Feature-level 증류를 위한 정렬 손실 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class FeatureAlignmentLoss(nn.Module):
    """Feature-level 증류를 위한 정렬 손실"""
    
    def __init__(self, 
                 spatial_weight: float = 1.0,
                 channel_weight: float = 1.0,
                 use_attention: bool = True):
        """
        Args:
            spatial_weight: Spatial attention 가중치
            channel_weight: Channel attention 가중치
            use_attention: Attention 메커니즘 사용 여부
        """
        super().__init__()
        self.spatial_weight = spatial_weight
        self.channel_weight = channel_weight
        self.use_attention = use_attention
        
    def forward(self, student_features: List[torch.Tensor], 
                teacher_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Feature alignment 손실 계산
        
        Returns:
            total_loss: 전체 특징 손실
            loss_dict: 레이어별 손실
        """
        loss_dict = {}
        
        # Feature가 없는 경우 처리
        if not student_features or not teacher_features:
            print("⚠️ Student 또는 Teacher features가 비어있음")
            total_loss = torch.tensor(0.0, device=teacher_features[0].device if teacher_features else 'cpu')
            loss_dict['feature_total'] = 0.0
            return total_loss, loss_dict
        
        # Tensor로 초기화 (첫 번째 feature의 device 사용)
        device = student_features[0].device
        total_loss = torch.tensor(0.0, device=device)
        
        for idx, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # 크기 맞추기
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], 
                                      mode='bilinear', align_corners=False)
            
            if self.use_attention:
                # Spatial Attention Transfer
                s_spatial_attn = self.spatial_attention(s_feat)
                t_spatial_attn = self.spatial_attention(t_feat).detach()
                spatial_loss = F.mse_loss(s_spatial_attn, t_spatial_attn)
                
                # Channel Attention Transfer  
                s_channel_attn = self.channel_attention(s_feat)
                t_channel_attn = self.channel_attention(t_feat).detach()
                channel_loss = F.mse_loss(s_channel_attn, t_channel_attn)
                
                layer_loss = (self.spatial_weight * spatial_loss + 
                             self.channel_weight * channel_loss)
            else:
                # Simple MSE loss
                layer_loss = F.mse_loss(s_feat, t_feat.detach())
            
            total_loss += layer_loss
            loss_dict[f'feature_layer_{idx}'] = layer_loss.item()
        
        # Tensor인지 확인 후 item() 호출
        if isinstance(total_loss, torch.Tensor):
            loss_dict['feature_total'] = total_loss.item()
            return total_loss / len(student_features), loss_dict
        else:
            print(f"⚠️ total_loss가 Tensor가 아님: {type(total_loss)}")
            loss_dict['feature_total'] = float(total_loss)
            return torch.tensor(float(total_loss), device=device) / len(student_features), loss_dict
    
    def spatial_attention(self, features):
        """Spatial attention map 생성"""
        # Channel 차원으로 평균내어 spatial attention 생성
        return torch.mean(features, dim=1, keepdim=True)
    
    def channel_attention(self, features):
        """Channel attention map 생성"""
        # Spatial 차원으로 평균내어 channel attention 생성
        batch_size, channels = features.size(0), features.size(1)
        return features.view(batch_size, channels, -1).mean(dim=2)
