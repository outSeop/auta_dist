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
            high_conf_mask = torch.sigmoid(teacher_obj) > 0.5  # [B, N, 1]
            high_conf_mask_bbox = high_conf_mask.squeeze(-1)   # [B, N] for bbox indexing
            
            print(f"🔍 high_conf_mask shape: {high_conf_mask.shape}")
            print(f"🔍 high_conf_mask_bbox shape: {high_conf_mask_bbox.shape}")
            
            if high_conf_mask.any():
                # BBox 마스킹 - 차원 맞춤
                student_bbox_masked = student_bbox[high_conf_mask_bbox]  # [num_valid, 4]
                teacher_bbox_masked = teacher_bbox[high_conf_mask_bbox]  # [num_valid, 4]
                
                print(f"🔍 Masked student_bbox shape: {student_bbox_masked.shape}")
                print(f"🔍 Masked teacher_bbox shape: {teacher_bbox_masked.shape}")
                
                # IoU loss + L1 loss 조합
                bbox_loss = self.bbox_distillation_loss(
                    student_bbox_masked,
                    teacher_bbox_masked
                )
            else:
                bbox_loss = torch.tensor(0.0, device=student_obj.device)
        except Exception as bbox_error:
            print(f"❌ BBox 손실 계산 오류: {bbox_error}")
            bbox_loss = torch.tensor(0.0, device=student_obj.device)
        
        # 3. Localization Quality 증류
        # Teacher의 localization quality를 전달
        print(f"🔍 Forward에서 targets 전달 - 타입: {type(targets)}")
        if hasattr(targets, 'shape'):
            print(f"🔍 Forward에서 targets shape: {targets.shape}")
        
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
        print(f"🔍 BBox 손실 계산 - Student shape: {student_bbox.shape}, Teacher shape: {teacher_bbox.shape}")
        
        # 너무 많은 박스가 있으면 샘플링 (메모리 절약)
        if student_bbox.shape[0] > 10000:
            indices = torch.randperm(student_bbox.shape[0])[:10000]
            student_bbox = student_bbox[indices]
            teacher_bbox = teacher_bbox[indices]
            print(f"🔧 샘플링 후 - Student: {student_bbox.shape}, Teacher: {teacher_bbox.shape}")
        
        try:
            # YOLO bbox 형식 (cx, cy, w, h) → (x1, y1, x2, y2) 변환
            student_bbox_xyxy = self.xywh_to_xyxy(student_bbox)
            teacher_bbox_xyxy = self.xywh_to_xyxy(teacher_bbox)
            
            # CIoU/DIoU loss 계산
            iou_loss = self.compute_ciou_loss(student_bbox_xyxy, teacher_bbox_xyxy)
            
            # L1 smooth loss (원본 좌표에서)
            l1_loss = self.smooth_l1(student_bbox, teacher_bbox).mean()
            
            return iou_loss + l1_loss
            
        except Exception as e:
            print(f"❌ BBox 손실 세부 오류: {e}")
            return torch.tensor(0.0, device=student_bbox.device)
    
    def compute_ciou_loss(self, pred_boxes, target_boxes):
        """Complete IoU Loss 계산"""
        # 간단한 IoU loss 구현 (실제로는 더 정교한 CIoU 필요)
        iou = self.box_iou(pred_boxes, target_boxes)
        return (1 - iou).mean()
    
    def box_iou(self, box1, box2):
        """IoU 계산"""
        # box1, box2: [N, 4] (x1, y1, x2, y2)
        print(f"🔍 IoU 계산 - box1: {box1.shape}, box2: {box2.shape}")
        
        # Shape 확인
        if box1.shape != box2.shape:
            print(f"⚠️ Shape 불일치: {box1.shape} vs {box2.shape}")
            min_len = min(box1.shape[0], box2.shape[0])
            box1 = box1[:min_len]
            box2 = box2[:min_len]
            print(f"🔧 Shape 조정 후: {box1.shape}, {box2.shape}")
        
        # 유효한 박스만 처리 (너비와 높이가 양수인 것)
        valid_mask1 = (box1[:, 2] > box1[:, 0]) & (box1[:, 3] > box1[:, 1])
        valid_mask2 = (box2[:, 2] > box2[:, 0]) & (box2[:, 3] > box2[:, 1])
        valid_mask = valid_mask1 & valid_mask2
        
        if not valid_mask.any():
            print("⚠️ 유효한 박스가 없음")
            return torch.zeros(box1.shape[0], device=box1.device)
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1 + area2 - inter_area
        iou = torch.zeros_like(area1)
        iou[valid_mask] = inter_area[valid_mask] / (union_area[valid_mask] + 1e-6)
        
        return iou
    
    def compute_localization_quality(self, bbox, objectness, targets):
        """Teacher의 localization quality 계산"""
        # Teacher 예측과 GT 간의 IoU를 quality score로 사용
        quality_scores = []
        
        print(f"🔍 Targets 타입: {type(targets)}")
        print(f"🔍 Targets 형태: {targets.shape if hasattr(targets, 'shape') else 'No shape'}")
        if hasattr(targets, 'keys'):
            print(f"🔍 Targets keys: {targets.keys()}")
        if isinstance(targets, (list, tuple)):
            print(f"🔍 Targets 길이: {len(targets)}")
            if len(targets) > 0:
                print(f"🔍 첫 번째 Target 타입: {type(targets[0])}")
                print(f"🔍 첫 번째 Target: {targets[0]}")
        
        # targets 형태에 따른 처리
        if isinstance(targets, torch.Tensor):
            # Tensor 형태인 경우 배치별로 처리
            batch_size = targets.shape[0] if targets.dim() > 0 else 1
            for i in range(batch_size):
                if targets.dim() > 1 and targets.shape[1] > 0:
                        target_i = targets[i] if targets.dim() > 1 else targets
                        if len(target_i.shape) > 0 and target_i.shape[0] > 0:
                            # Teacher 예측과 GT 매칭
                            teacher_conf = torch.sigmoid(objectness[i])  # [N, 1]
                            high_conf_idx = teacher_conf.squeeze(-1) > 0.5  # [N] for bbox indexing
                            
                            if high_conf_idx.any():
                                teacher_boxes = bbox[i][high_conf_idx]  # [num_valid, 4]
                                # targets가 [batch, max_labels, 6] 형태일 수 있음 (class, x, y, w, h, conf)
                                if target_i.shape[-1] >= 5:
                                    gt_boxes = target_i[:, 1:5] if target_i.shape[-1] > 5 else target_i[:, :4]  
                                else:
                                    gt_boxes = target_i[:, :4]  # 이미 bbox만 있는 경우
                                
                                # GT 박스가 있는 경우만 IoU 계산
                                if gt_boxes.shape[0] > 0:
                                    ious = self.box_iou(teacher_boxes, gt_boxes)
                                    quality = ious.max(dim=1)[0]
                                    quality_scores.append(quality)
        
        elif isinstance(targets, (list, tuple)):
            # List 형태인 경우
            for i in range(len(targets)):
                if len(targets[i]) > 0:
                    # Teacher 예측과 GT 매칭
                    teacher_conf = torch.sigmoid(objectness[i])  # [N, 1]
                    high_conf_idx = teacher_conf.squeeze(-1) > 0.5  # [N] for bbox indexing
                    
                    if high_conf_idx.any():
                        teacher_boxes = bbox[i][high_conf_idx]  # [num_valid, 4]
                        gt_boxes = targets[i][:, 1:5] if targets[i].shape[-1] > 4 else targets[i][:, :4]
                        # 디바이스 및 dtype 정렬
                        gt_boxes = gt_boxes.to(teacher_boxes.device, dtype=teacher_boxes.dtype)
                        
                        # IoU 계산하여 quality score 생성
                        ious = self.box_iou(teacher_boxes, gt_boxes)
                        quality = ious.max(dim=1)[0]
                        quality_scores.append(quality)
        
        elif isinstance(targets, dict):
            # 딕셔너리 형태인 경우 - YOLODataset 표준 형태
            print("🔍 딕셔너리 형태 targets 처리")
            batch_idx = targets.get('batch_idx', None)
            bboxes = targets.get('bboxes', None)
            cls = targets.get('cls', None)
            
            if batch_idx is not None and bboxes is not None:
                print(f"🔍 Batch indices: {batch_idx.shape if hasattr(batch_idx, 'shape') else batch_idx}")
                print(f"🔍 BBoxes: {bboxes.shape if hasattr(bboxes, 'shape') else bboxes}")
                print(f"🔍 Classes: {cls.shape if hasattr(cls, 'shape') else cls}")
                
                # 배치별로 GT 박스 그룹화
                unique_batch_idx = torch.unique(batch_idx)
                for batch_i in unique_batch_idx:
                    # 현재 배치에 속하는 GT들
                    mask = batch_idx == batch_i
                    batch_bboxes = bboxes[mask]  # [num_gt, 4]
                    
                    if batch_bboxes.shape[0] > 0:
                        batch_i_int = int(batch_i.item())
                        # Teacher 예측과 GT 매칭
                        teacher_conf = torch.sigmoid(objectness[batch_i_int])  # [N, 1]
                        high_conf_idx = teacher_conf.squeeze(-1) > 0.5  # [N] for bbox indexing
                        
                        if high_conf_idx.any():
                            teacher_boxes = bbox[batch_i_int][high_conf_idx]  # [num_valid, 4]
                            
                            # GT 박스가 있는 경우만 IoU 계산
                            if batch_bboxes.shape[0] > 0:
                                # 좌표 형식이 이미 xyxy인지 xywh인지 확인 필요
                                # 디바이스 및 dtype 정렬
                                batch_bboxes = batch_bboxes.to(teacher_boxes.device, dtype=teacher_boxes.dtype)
                                ious = self.box_iou(teacher_boxes, batch_bboxes)
                                quality = ious.max(dim=1)[0]
                                quality_scores.append(quality)
            else:
                print("⚠️ batch_idx 또는 bboxes가 targets에 없음")
                
        else:
            # 다른 형태인 경우 기본값 반환
            print(f"⚠️ 지원하지 않는 targets 형태: {type(targets)}")
            return torch.tensor(0.0, device=bbox.device)
        
        if quality_scores:
            return torch.cat(quality_scores)
        else:
            return torch.tensor(0.0, device=bbox.device)
    
    def xywh_to_xyxy(self, bbox):
        """YOLO bbox 형식 (cx, cy, w, h) → (x1, y1, x2, y2) 변환"""
        if bbox.shape[-1] != 4:
            print(f"⚠️ 예상과 다른 bbox 형식: {bbox.shape}")
            return bbox
            
        cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
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
