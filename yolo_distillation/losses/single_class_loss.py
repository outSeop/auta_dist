"""
단일 클래스 객체 탐지를 위한 증류 손실 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SingleClassDistillationLoss(nn.Module):
    """단일 클래스 객체 탐지를 위한 증류 손실"""
    
    def __init__(self, 
                 bbox_weight: float = 2.0,
                 objectness_weight: float = 1.0,
                 feature_weight: float = 1.0,
                 iou_threshold: float = 0.5):
        """
        Args:
            bbox_weight: Bounding box regression 증류 가중치
            objectness_weight: Objectness score 증류 가중치  
            feature_weight: Feature 증류 가중치
            iou_threshold: Positive sample을 위한 IoU 임계값
        """
        super().__init__()
        self.bbox_weight = bbox_weight
        self.objectness_weight = objectness_weight
        self.feature_weight = feature_weight
        self.iou_threshold = iou_threshold
        
        # 손실 함수들
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, 
                student_outputs: Dict,
                teacher_outputs: Dict,
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        단일 클래스 증류 손실 계산
        
        Args:
            student_outputs: Student 모델 출력 (bbox, objectness)
            teacher_outputs: Teacher 모델 출력 (bbox, objectness)
            targets: Ground truth 타겟
        
        Returns:
            total_loss: 전체 손실
            loss_dict: 개별 손실 딕셔너리
        """
        
        # 1. Objectness Score 증류 (객체 존재 확률)
        student_obj = student_outputs['objectness']  # [B, N, 1]
        teacher_obj = teacher_outputs['objectness'].detach()
        
        print(f"🔍 Student objectness shape: {student_obj.shape}")
        print(f"🔍 Teacher objectness shape: {teacher_obj.shape}")
        
        # 지능적 차원 정렬 (정보 손실 최소화)
        if student_obj.shape != teacher_obj.shape:
            teacher_obj, student_obj = self.align_outputs_intelligently(teacher_obj, student_obj)
            print(f"🔧 지능적 차원 조정 후 - Student: {student_obj.shape}, Teacher: {teacher_obj.shape}")
        
        # Teacher의 objectness를 soft label로 사용
        try:
            obj_loss = F.binary_cross_entropy_with_logits(
                student_obj,
                torch.sigmoid(teacher_obj),
                reduction='mean'
            )
        except Exception as obj_error:
            print(f"❌ Objectness 손실 계산 오류: {obj_error}")
            obj_loss = torch.tensor(0.0, device=student_obj.device)
        
        # 2. Bounding Box Regression 증류
        student_bbox = student_outputs['bbox']  # [B, N, 4]
        teacher_bbox = teacher_outputs['bbox'].detach()
        
        print(f"🔍 Student bbox shape: {student_bbox.shape}")
        print(f"🔍 Teacher bbox shape: {teacher_bbox.shape}")
        
        # 지능적 차원 정렬 (BBox도 동일하게)
        if student_bbox.shape != teacher_bbox.shape:
            teacher_bbox, student_bbox = self.align_outputs_intelligently(teacher_bbox, student_bbox)
            print(f"🔧 BBox 지능적 차원 조정 후 - Student: {student_bbox.shape}, Teacher: {teacher_bbox.shape}")
        
        # Teacher confidence가 높은 예측만 사용
        try:
            high_conf_mask = torch.sigmoid(teacher_obj) > 0.5
            
            if high_conf_mask.any():
                # IoU loss + L1 loss 조합
                bbox_loss = self.bbox_distillation_loss(
                    student_bbox[high_conf_mask],
                    teacher_bbox[high_conf_mask]
                )
            else:
                bbox_loss = torch.tensor(0.0, device=student_obj.device)
        except Exception as bbox_error:
            print(f"❌ BBox 손실 계산 오류: {bbox_error}")
            bbox_loss = torch.tensor(0.0, device=student_obj.device)
        
        # 3. Localization Quality 증류
        # Teacher의 localization quality를 전달
        loc_quality = self.compute_localization_quality(
            teacher_bbox, teacher_obj, targets
        )
        
        loc_loss = self.localization_quality_loss(
            student_bbox, student_obj, loc_quality
        )
        
        # 전체 손실 계산
        total_loss = (
            self.objectness_weight * obj_loss +
            self.bbox_weight * bbox_loss +
            self.feature_weight * loc_loss
        )
        
        loss_dict = {
            'objectness_loss': obj_loss.item(),
            'bbox_loss': bbox_loss.item() if isinstance(bbox_loss, torch.Tensor) else bbox_loss,
            'localization_loss': loc_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def bbox_distillation_loss(self, student_bbox, teacher_bbox):
        """Bounding Box 증류 손실 (IoU + L1)"""
        # CIoU/DIoU loss 계산
        iou_loss = self.compute_ciou_loss(student_bbox, teacher_bbox)
        
        # L1 smooth loss
        l1_loss = self.smooth_l1(student_bbox, teacher_bbox).mean()
        
        return iou_loss + l1_loss
    
    def compute_ciou_loss(self, pred_boxes, target_boxes):
        """Complete IoU Loss 계산"""
        # 간단한 IoU loss 구현 (실제로는 더 정교한 CIoU 필요)
        iou = self.box_iou(pred_boxes, target_boxes)
        return (1 - iou).mean()
    
    def box_iou(self, box1, box2):
        """IoU 계산"""
        # box1, box2: [N, 4] (x1, y1, x2, y2)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    def compute_localization_quality(self, bbox, objectness, targets):
        """Teacher의 localization quality 계산"""
        # Teacher 예측과 GT 간의 IoU를 quality score로 사용
        quality_scores = []
        
        for i in range(len(targets)):
            if len(targets[i]) > 0:
                # Teacher 예측과 GT 매칭
                teacher_conf = torch.sigmoid(objectness[i])
                high_conf_idx = teacher_conf > 0.5
                
                if high_conf_idx.any():
                    teacher_boxes = bbox[i][high_conf_idx.squeeze()]
                    gt_boxes = targets[i][:, 1:5]  # [class, x, y, w, h] 형식 가정
                    
                    # IoU 계산하여 quality score 생성
                    ious = self.box_iou(teacher_boxes, gt_boxes)
                    quality = ious.max(dim=1)[0]
                    quality_scores.append(quality)
        
        if quality_scores:
            return torch.cat(quality_scores)
        else:
            return torch.tensor(0.0, device=bbox.device)
    
    def localization_quality_loss(self, student_bbox, student_obj, quality_scores):
        """Localization quality 전달을 위한 손실"""
        if isinstance(quality_scores, torch.Tensor) and quality_scores.numel() > 0:
            # Student의 confidence가 Teacher의 localization quality를 따라가도록
            return F.mse_loss(
                torch.sigmoid(student_obj).mean(),
                quality_scores.mean()
            )
        return torch.tensor(0.0, device=student_bbox.device)
    
    def align_outputs_intelligently(self, teacher_out, student_out):
        """
        Teacher와 Student 출력을 지능적으로 정렬 (정보 손실 최소화)
        """
        print(f"🔧 정렬 전 - Teacher: {teacher_out.shape}, Student: {student_out.shape}")
        
        # 다차원 텐서 안전하게 처리
        batch_size = teacher_out.shape[0]
        teacher_seq_len = teacher_out.shape[1]
        student_seq_len = student_out.shape[1]
        feature_dim = teacher_out.shape[2]  # objectness: 1, bbox: 4
        
        # Teacher가 더 작은 경우 (일반적인 경우)
        if teacher_seq_len < student_seq_len:
            # Teacher를 Student 크기로 확장 (interpolation)
            # [B, seq, feat] → [B, feat, seq] → interpolate → [B, feat, new_seq] → [B, new_seq, feat]
            teacher_transposed = teacher_out.transpose(1, 2)  # [B, feat, seq]
            teacher_expanded = F.interpolate(
                teacher_transposed,
                size=student_seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [B, new_seq, feat]
            
            print(f"📈 Teacher 확장: {teacher_out.shape} → {teacher_expanded.shape}")
            return teacher_expanded, student_out
            
        # Student가 더 작은 경우 
        elif student_seq_len < teacher_seq_len:
            # Student를 Teacher 크기로 확장
            student_transposed = student_out.transpose(1, 2)
            student_expanded = F.interpolate(
                student_transposed,
                size=teacher_seq_len,
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            
            print(f"📈 Student 확장: {student_out.shape} → {student_expanded.shape}")
            return teacher_out, student_expanded
            
        else:
            # 이미 같은 크기
            print(f"✅ 크기 동일: {teacher_out.shape}")
            return teacher_out, student_out
