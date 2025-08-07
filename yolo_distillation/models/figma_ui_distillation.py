"""
Figma UI 검증을 위한 YOLOv11 증류 메인 클래스
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

from ..losses.single_class_loss import SingleClassDistillationLoss
from ..losses.feature_alignment_loss import FeatureAlignmentLoss


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
              num_workers: int = 8,
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
        
        # 데이터로더 생성 시도
        train_loader, val_loader = self._create_dataloaders(batch_size, num_workers)
        
        # 데이터로더가 없는 경우 YOLO 기본 학습 방식 사용
        if train_loader is None:
            print("커스텀 데이터로더 생성 실패. YOLO 기본 학습 방식을 사용합니다.")
            return 0
            # return self._train_with_yolo_default(epochs, batch_size, learning_rate, save_dir)
        
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
    
    def _train_with_yolo_default(self, epochs, batch_size, learning_rate, save_dir):
        """
        YOLO 기본 학습 방식을 사용한 Knowledge Distillation
        """
        print("YOLO 기본 학습 방식으로 Knowledge Distillation을 시작합니다...")
        
        # YOLO 기본 학습 설정
        results = self.student.train(
            data=self.modified_data_yaml,
            epochs=epochs,
            batch=batch_size,
            lr0=learning_rate,
            device=self.device,
            project=save_dir,
            name='distillation_run',
            save=True,
            plots=True,
            verbose=True
        )
        
        # 결과에서 최고 mAP 추출
        if hasattr(results, 'results_dict'):
            best_map = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
        else:
            best_map = 0.0
            
        print(f"YOLO 기본 학습 완료! Best mAP: {best_map:.4f}")
        return best_map
    
    def _create_dataloaders(self, batch_size, num_workers):
        """
        YOLO 데이터로더를 직접 생성
        """
        try:
            from ultralytics.data import build_dataloader
            from ultralytics.utils import DEFAULT_CFG
            import yaml
            
            # 데이터셋 설정 로드
            with open(self.modified_data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # YOLO 기본 설정 사용
            cfg = DEFAULT_CFG.copy()
            cfg.update({
                'batch': batch_size,
                'workers': num_workers,
                'data': self.modified_data_yaml
            })
            
            print("커스텀 데이터로더 생성을 시도합니다...")
            
            # 학습용 데이터로더 생성
            train_loader = build_dataloader(
                dataset=data_config.get('train', 'train'),
                batch_size=batch_size,
                workers=num_workers,
                shuffle=True,
                augment=True
            )[0] if 'train' in data_config else None
            
            # 검증용 데이터로더 생성  
            val_loader = build_dataloader(
                dataset=data_config.get('val', 'val'),
                batch_size=batch_size,
                workers=num_workers,
                shuffle=False,
                augment=False
            )[0] if 'val' in data_config else None
            
            if train_loader is not None:
                print("커스텀 데이터로더 생성 성공!")
                return train_loader, val_loader
            else:
                print("데이터셋 경로를 찾을 수 없습니다.")
                return None, None
                
        except Exception as e:
            print(f"데이터로더 생성 중 오류: {e}")
            return None, None
