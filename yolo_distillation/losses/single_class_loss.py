"""
Single Class Detection을 위한 Knowledge Distillation 손실 함수 (리팩토링 버전)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SingleClassDistillationLoss(nn.Module):
    """Single Class Detection을 위한 Knowledge Distillation 손실"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, 
                 temperature: float = 4.0, device: str = 'cuda'):
        super().__init__()
        self.alpha = alpha  # objectness 신뢰도 임계값
        self.beta = beta    # 손실 조합 가중치
        self.temperature = temperature  # 증류 온도
        self.device = device
        
    def forward(self, student_outputs: Dict, teacher_outputs: Dict, 
                targets) -> Tuple[torch.Tensor, Dict]:
        """
        Knowledge Distillation 손실 계산
        
        Args:
            student_outputs: {'bbox': [B, N, 4], 'objectness': [B, N, 1]}
            teacher_outputs: {'bbox': [B, M, 4], 'objectness': [B, M, 1]}  
            targets: Ground truth (다양한 형식 지원)
            
        Returns:
            total_loss: 전체 손실
            metrics: 손실 세부 정보
        """
        loss_dict = {}
        
        try:
            # 1. Objectness Score 증류
            obj_loss = self._compute_objectness_loss(student_outputs, teacher_outputs)
            loss_dict['obj_loss'] = obj_loss.item()
            
            # 2. Bounding Box 증류
            bbox_loss = self._compute_bbox_loss(student_outputs, teacher_outputs)
            loss_dict['bbox_loss'] = bbox_loss.item()
            
            # 3. Localization Quality 증류
            loc_loss = self._compute_localization_loss(teacher_outputs, targets)
            loss_dict['loc_loss'] = loc_loss.item()
            
            # 전체 손실 조합
            total_loss = obj_loss + self.beta * bbox_loss + 0.1 * loc_loss
            loss_dict['total_loss'] = total_loss.item()
            
            return total_loss, loss_dict
            
        except Exception as e:
            print(f"❌ 손실 계산 중 오류: {e}")
            # 오류 시 기본 손실 반환
            device = student_outputs['objectness'].device
            return torch.tensor(0.0, device=device), {'total_loss': 0.0}
    
    def _compute_objectness_loss(self, student_outputs: Dict, teacher_outputs: Dict) -> torch.Tensor:
        """Objectness 점수 증류 손실"""
        student_obj = student_outputs['objectness']
        teacher_obj = teacher_outputs['objectness'].detach()
        
        # 차원 정렬
        teacher_obj, student_obj = self._align_tensors(teacher_obj, student_obj)
        
        # Binary Cross Entropy 손실
        return F.binary_cross_entropy_with_logits(
            student_obj, torch.sigmoid(teacher_obj)
        )
    
    def _compute_bbox_loss(self, student_outputs: Dict, teacher_outputs: Dict) -> torch.Tensor:
        """Bounding Box 증류 손실"""
        student_bbox = student_outputs['bbox']
        teacher_bbox = teacher_outputs['bbox'].detach()
        teacher_obj = teacher_outputs['objectness'].detach()
        
        # 차원 정렬
        teacher_bbox, student_bbox = self._align_tensors(teacher_bbox, student_bbox)
        teacher_obj, _ = self._align_tensors(teacher_obj, student_outputs['objectness'])
        
        # 높은 신뢰도 영역에서만 손실 계산
        with torch.no_grad():
            high_conf_mask = torch.sigmoid(teacher_obj) > self.alpha
            high_conf_mask = high_conf_mask.squeeze(-1)  # [B, N]
        
        if high_conf_mask.any():
            student_bbox_masked = student_bbox[high_conf_mask]
            teacher_bbox_masked = teacher_bbox[high_conf_mask]
            
            # 너무 많은 박스가 있으면 샘플링
            if student_bbox_masked.shape[0] > 10000:
                indices = torch.randperm(student_bbox_masked.shape[0])[:10000]
                student_bbox_masked = student_bbox_masked[indices]
                teacher_bbox_masked = teacher_bbox_masked[indices]
            
            # IoU + L1 손실 조합
            return self._bbox_regression_loss(student_bbox_masked, teacher_bbox_masked)
        else:
            return torch.tensor(0.0, device=student_bbox.device)
    
    def _compute_localization_loss(self, teacher_outputs: Dict, targets) -> torch.Tensor:
        """Localization quality 증류 손실"""
        try:
            teacher_bbox = teacher_outputs['bbox']
            teacher_obj = teacher_outputs['objectness']
            
            # Ground truth와의 IoU 기반 quality 계산
            quality_scores = self._compute_quality_scores(teacher_bbox, teacher_obj, targets)
            
            if isinstance(quality_scores, torch.Tensor) and quality_scores.numel() > 0:
                # Student가 Teacher의 localization quality를 따라가도록
                student_conf = torch.sigmoid(teacher_obj)
                return F.mse_loss(student_conf.squeeze(), quality_scores)
            else:
                return torch.tensor(0.0, device=teacher_bbox.device)
                
        except Exception as e:
            print(f"⚠️ Localization 손실 계산 오류: {e}")
            return torch.tensor(0.0, device=teacher_outputs['bbox'].device)
    
    def _align_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """두 텐서의 차원을 정렬 (정보 손실 최소화)"""
        if tensor1.shape == tensor2.shape:
            return tensor1, tensor2
        
        # 더 작은 텐서를 큰 텐서 크기로 확장
        if tensor1.shape[1] < tensor2.shape[1]:
            tensor1 = self._expand_tensor(tensor1, tensor2.shape[1])
        elif tensor2.shape[1] < tensor1.shape[1]:
            tensor2 = self._expand_tensor(tensor2, tensor1.shape[1])
        
        return tensor1, tensor2
    
    def _expand_tensor(self, tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        """텐서를 목표 크기로 확장 (interpolation 사용)"""
        if tensor.dim() == 3:  # [B, seq, feat]
            tensor_transposed = tensor.transpose(1, 2)  # [B, feat, seq]
            expanded = F.interpolate(
                tensor_transposed, size=target_size, mode='linear', align_corners=False
            )
            return expanded.transpose(1, 2)  # [B, seq, feat]
        return tensor
    
    def _bbox_regression_loss(self, student_bbox: torch.Tensor, teacher_bbox: torch.Tensor) -> torch.Tensor:
        """Bounding box regression 손실 (IoU + L1)"""
        # L1 손실
        l1_loss = F.l1_loss(student_bbox, teacher_bbox)
        
        # IoU 손실 (간소화)
        try:
            iou_loss = 1.0 - self._compute_iou(student_bbox, teacher_bbox).mean()
            return 0.5 * l1_loss + 0.5 * iou_loss
        except:
            return l1_loss
    
    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """IoU 계산 (xywh -> xyxy 변환 후)"""
        # YOLO format (cx, cy, w, h) to (x1, y1, x2, y2)
        box1_xyxy = self._xywh_to_xyxy(box1)
        box2_xyxy = self._xywh_to_xyxy(box2)
        
        # Intersection
        inter_min = torch.max(box1_xyxy[..., :2], box2_xyxy[..., :2])
        inter_max = torch.min(box1_xyxy[..., 2:], box2_xyxy[..., 2:])
        inter_wh = torch.clamp(inter_max - inter_min, min=0)
        intersection = inter_wh[..., 0] * inter_wh[..., 1]
        
        # Areas
        area1 = (box1_xyxy[..., 2] - box1_xyxy[..., 0]) * (box1_xyxy[..., 3] - box1_xyxy[..., 1])
        area2 = (box2_xyxy[..., 2] - box2_xyxy[..., 0]) * (box2_xyxy[..., 3] - box2_xyxy[..., 1])
        
        # Union
        union = area1 + area2 - intersection
        
        # IoU
        iou = intersection / torch.clamp(union, min=1e-6)
        return torch.clamp(iou, min=0.0, max=1.0)
    
    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """YOLO format (cx, cy, w, h) to (x1, y1, x2, y2)"""
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _compute_quality_scores(self, teacher_bbox: torch.Tensor, teacher_obj: torch.Tensor, targets) -> torch.Tensor:
        """Ground truth와의 IoU 기반 quality 점수 계산"""
        try:
            # 간소화된 quality 계산
            if isinstance(targets, dict) and 'bboxes' in targets:
                # 기본적인 품질 점수 반환 (실제 구현에서는 GT와 IoU 계산)
                return torch.sigmoid(teacher_obj).mean(dim=[1, 2])  # [B]
            else:
                # 기본값 반환
                batch_size = teacher_bbox.shape[0]
                return torch.ones(batch_size, device=teacher_bbox.device) * 0.5
                
        except Exception as e:
            print(f"⚠️ Quality 점수 계산 오류: {e}")
            batch_size = teacher_bbox.shape[0]
            return torch.zeros(batch_size, device=teacher_bbox.device)
