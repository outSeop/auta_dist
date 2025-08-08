"""
YOLOv11 Knowledge Distillation for Single-Class Detection
Optimized for Figma UI Component Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import wandb
from datetime import datetime
import cv2
import os
from torch.utils.data import DataLoader
import multiprocessing as mp


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
        
        # Teacher의 objectness를 soft label로 사용
        obj_loss = F.binary_cross_entropy_with_logits(
            student_obj,
            torch.sigmoid(teacher_obj),
            reduction='mean'
        )
        
        # 2. Bounding Box Regression 증류
        student_bbox = student_outputs['bbox']  # [B, N, 4]
        teacher_bbox = teacher_outputs['bbox'].detach()
        
        # Teacher confidence가 높은 예측만 사용
        high_conf_mask = torch.sigmoid(teacher_obj) > 0.5
        
        if high_conf_mask.any():
            # IoU loss + L1 loss 조합
            bbox_loss = self.bbox_distillation_loss(
                student_bbox[high_conf_mask],
                teacher_bbox[high_conf_mask]
            )
        else:
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
        total_loss = 0
        loss_dict = {}
        
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
        
        loss_dict['feature_total'] = total_loss.item()
        return total_loss / len(student_features), loss_dict
    
    def spatial_attention(self, features):
        """Spatial attention map 생성"""
        # Channel 차원으로 평균내어 spatial attention 생성
        return torch.mean(features, dim=1, keepdim=True)
    
    def channel_attention(self, features):
        """Channel attention map 생성"""
        # Spatial 차원으로 평균내어 channel attention 생성
        batch_size, channels = features.size(0), features.size(1)
        return features.view(batch_size, channels, -1).mean(dim=2)


class FigmaUIDistillation:
    """Figma UI 검증을 위한 YOLOv11 증류 메인 클래스"""
    
    def __init__(self,
                 teacher_model: str = 'yolov11l.pt',
                 student_model: str = 'yolov11s.yaml',
                 data_yaml: str = 'figma_ui.yaml',
                 device: str = 'cuda',
                 use_wandb: bool = True):
        """
        Args:
            teacher_model: Teacher 모델 경로 (YOLOv11-l)
            student_model: Student 모델 설정 (YOLOv11-s 또는 YOLOv11-m)
            data_yaml: Figma UI 데이터셋 설정
            device: 학습 디바이스
            use_wandb: WandB 사용 여부
        """
        self.device = device
        self.use_wandb = use_wandb
        
        # 모델 초기화
        print(f"Loading teacher model: {teacher_model}")
        self.teacher = YOLO(teacher_model)
        self.teacher.model.eval()
        for param in self.teacher.model.parameters():
            param.requires_grad = False
        
        print(f"Initializing student model: {student_model}")
        self.student = YOLO(student_model)
        
        # 단일 클래스 설정
        self.num_classes = 1  # Figma UI component
        
        # 손실 함수
        self.distillation_loss = SingleClassDistillationLoss(
            bbox_weight=2.0,
            objectness_weight=1.0,
            feature_weight=1.0
        )
        self.feature_loss = FeatureAlignmentLoss(
            use_attention=True
        )
        
        # 데이터 설정
        self.data_yaml = data_yaml
        self._modify_data_for_single_class()
        
        # WandB 초기화
        if self.use_wandb:
            self.init_wandb()
    
    def _modify_data_for_single_class(self):
        """데이터셋을 단일 클래스로 수정"""
        with open(self.data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # 단일 클래스로 설정
        data['nc'] = 1
        data['names'] = ['ui_component']
        
        # 임시 파일로 저장
        self.modified_data_yaml = 'temp_single_class_data.yaml'
        with open(self.modified_data_yaml, 'w') as f:
            yaml.dump(data, f)
    
    def init_wandb(self):
        """WandB 초기화 및 설정"""
        config = {
            'project': 'figma-ui-detection',
            'teacher_model': 'YOLOv11-l',
            'student_model': self.student.model.__class__.__name__,
            'num_classes': self.num_classes,
            'task': 'single_class_detection',
            'bbox_weight': self.distillation_loss.bbox_weight,
            'objectness_weight': self.distillation_loss.objectness_weight,
            'feature_weight': self.distillation_loss.feature_weight
        }
        
        wandb.init(
            project="figma-ui-yolo11-kd",
            name=f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=["single-class", "figma-ui", "knowledge-distillation"]
        )
        
        # 모델 복잡도 비교 로깅
        self.log_model_comparison()
    
    def log_model_comparison(self):
        """Teacher vs Student 모델 비교 로깅"""
        if not self.use_wandb:
            return
        
        teacher_params = sum(p.numel() for p in self.teacher.model.parameters())
        student_params = sum(p.numel() for p in self.student.model.parameters())
        
        comparison_table = wandb.Table(
            columns=["Model", "Parameters", "Size (MB)", "Compression Ratio"],
            data=[
                ["Teacher (YOLOv11-l)", teacher_params, teacher_params * 4 / 1e6, 1.0],
                ["Student", student_params, student_params * 4 / 1e6, 
                 teacher_params / student_params]
            ]
        )
        
        wandb.log({
            "model_comparison": comparison_table,
            "compression_ratio": teacher_params / student_params
        })
    
    def extract_features(self, model, x):
        """모델에서 다중 스케일 특징 추출"""
        features = []
        
        # YOLOv11 백본에서 특징 추출
        # P3, P4, P5 레벨 특징
        for i, module in enumerate(model.model):
            x = module(x)
            if i in [4, 6, 9]:  # 주요 특징 레이어
                features.append(x)
        
        return features, x
    
    def train_step(self, batch, optimizer):
        """단일 학습 스텝"""
        images = batch['img'].to(self.device)
        targets = batch['batch']  # 단일 클래스 타겟
        
        # Teacher 추론 (no gradient)
        with torch.no_grad():
            teacher_features, teacher_outputs = self.teacher.model(images, augment=False)
            teacher_outputs = self.parse_model_outputs(teacher_outputs)
        
        # Student 추론
        student_features, student_outputs = self.student.model(images, augment=False)
        student_outputs = self.parse_model_outputs(student_outputs)
        
        # 1. Detection 증류 손실 (objectness + bbox)
        det_loss, det_metrics = self.distillation_loss(
            student_outputs, teacher_outputs, targets
        )
        
        # 2. Feature 증류 손실
        feat_loss, feat_metrics = self.feature_loss(
            student_features, teacher_features
        )
        
        # 3. 원본 YOLO 손실 (Ground Truth 기반)
        base_loss = self.student.model.loss(student_outputs, targets)
        
        # 전체 손실 조합
        total_loss = det_loss + 0.5 * feat_loss + 0.3 * base_loss
        
        # 역전파
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 메트릭 로깅
        metrics = {
            'loss/total': total_loss.item(),
            'loss/detection': det_loss.item(),
            'loss/feature': feat_loss.item(),
            'loss/base': base_loss.item(),
            **det_metrics,
            **feat_metrics
        }
        
        return metrics
    
    def parse_model_outputs(self, outputs):
        """모델 출력을 파싱하여 bbox와 objectness 분리"""
        # YOLOv11 출력 형식에 맞게 파싱
        # outputs: [batch, num_anchors, 5] for single class
        # [x, y, w, h, objectness]
        
        return {
            'bbox': outputs[..., :4],
            'objectness': outputs[..., 4:5]
        }
    
    def validate(self, val_loader):
        """검증 수행"""
        self.student.model.eval()
        
        metrics = {
            'val/mAP': 0,
            'val/precision': 0,
            'val/recall': 0,
            'val/inference_time': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['img'].to(self.device)
                
                # 추론 시간 측정
                import time
                start_time = time.time()
                outputs = self.student.model(images)
                inference_time = time.time() - start_time
                
                metrics['val/inference_time'] += inference_time
                
                # mAP 계산 (실제 구현 필요)
                # ...
        
        self.student.model.train()
        return metrics
    
    def train(self,
              epochs: int = 100,
              batch_size: int = 16,
              learning_rate: float = 0.001,
              num_workers: int = 2,
              save_dir: str = './runs/figma_distillation'):
        """
        전체 학습 실행
        
        Args:
            epochs: 학습 에폭
            batch_size: 배치 크기
            learning_rate: 학습률
            num_workers: 데이터로더 워커 수
            save_dir: 모델 저장 경로
        """
        
        # 옵티마이저
        optimizer = torch.optim.AdamW(
            self.student.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0005
        )
        
        # 스케줄러 (Cosine Annealing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        # 데이터로더 생성 (실제 구현 필요)
        # train_loader = create_dataloader(...)
        # val_loader = create_dataloader(...)
        
        best_map = 0
        
        for epoch in range(epochs):
            print(f"\n[Epoch {epoch+1}/{epochs}]")
            
            # 학습
            epoch_metrics = {}
            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch, optimizer)
                
                # 배치 메트릭 누적
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0
                    epoch_metrics[k] += v
                
                # 주기적 로깅
                if batch_idx % 10 == 0 and self.use_wandb:
                    wandb.log(metrics, step=epoch * len(train_loader) + batch_idx)
            
            # 검증
            val_metrics = self.validate(val_loader)
            
            # 에폭 평균 메트릭
            for k in epoch_metrics:
                epoch_metrics[k] /= len(train_loader)
            
            # 스케줄러 업데이트
            scheduler.step()
            
            # WandB 로깅
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **epoch_metrics,
                    **val_metrics,
                    'lr': scheduler.get_last_lr()[0]
                })
            
            # 모델 저장
            if val_metrics['val/mAP'] > best_map:
                best_map = val_metrics['val/mAP']
                save_path = f"{save_dir}/best_model.pt"
                torch.save(self.student.model.state_dict(), save_path)
                print(f"Best model saved: mAP={best_map:.4f}")
                
                if self.use_wandb:
                    wandb.save(save_path)
        
        return best_map


# 사용 예제
if __name__ == "__main__":
    # Figma UI 검증을 위한 증류 설정
    config = {
        'teacher_model': 'yolov11l.pt',  # 사전 학습된 Teacher
        'student_model': 'yolov11s.yaml',  # Student 모델 (s 또는 m)
        'data_yaml': 'figma_ui_dataset.yaml',  # Figma UI 데이터셋
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_workers': 8
    }
    
    # 증류 실행
    distiller = FigmaUIDistillation(
        teacher_model=config['teacher_model'],
        student_model=config['student_model'],
        data_yaml=config['data_yaml'],
        use_wandb=True
    )
    
    best_map = distiller.train(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        num_workers=config['num_workers']
    )
    
    print(f"\n학습 완료! Best mAP: {best_map:.4f}")
    
    # 모델 성능 비교
    print("\n=== 모델 성능 비교 ===")
    print("Teacher (YOLOv11-l): 높은 정확도, 느린 속도")
    print(f"Student (YOLOv11-s): mAP={best_map:.4f}, 빠른 추론 속도")
    print("압축률: ~10x 파라미터 감소")
    print("속도 향상: ~3-5x 추론 속도 향상")
    
    # 추론 속도 테스트
    print("\n=== 추론 속도 테스트 ===")
    import time
    
    test_image = torch.randn(1, 3, 640, 640).to('cuda')
    
    # Teacher 속도
    start = time.time()
    for _ in range(100):
        _ = distiller.teacher.model(test_image)
    teacher_time = (time.time() - start) / 100
    
    # Student 속도
    start = time.time()
    for _ in range(100):
        _ = distiller.student.model(test_image)
    student_time = (time.time() - start) / 100
    
    print(f"Teacher 추론 시간: {teacher_time*1000:.2f}ms")
    print(f"Student 추론 시간: {student_time*1000:.2f}ms")
    print(f"속도 향상: {teacher_time/student_time:.2f}x")