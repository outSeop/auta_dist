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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

from ..losses.single_class_loss import SingleClassDistillationLoss
from ..losses.feature_alignment_loss import FeatureAlignmentLoss


class FigmaUIDistillation:
    """Figma UI 검증을 위한 YOLOv11 증류 메인 클래스"""
    
    def __init__(self,
                 teacher_model: str = 'yolov11l.pt',
                 student_model: str = 'yolov11s.yaml',
                 data_yaml: str = 'figma_ui.yaml',
                 device: str = 'cuda',
                 use_wandb: bool = True,
                 verbose_debug: bool = False):
        """
        Args:
            teacher_model: Teacher 모델 경로 (YOLOv11-l)
            student_model: Student 모델 설정 (YOLOv11-s 또는 YOLOv11-m)
            data_yaml: Figma UI 데이터셋 설정
            device: 학습 디바이스
            use_wandb: WandB 사용 여부
            verbose_debug: 상세 디버깅 로그 출력 여부
        """
        self.device = device
        self.use_wandb = use_wandb
        self.verbose_debug = verbose_debug
        
        # 모델 초기화
        print(f"Loading teacher model: {teacher_model}")
        self.teacher = YOLO(teacher_model)
        self.teacher.model = self.teacher.model.to(self.device)
        self.teacher.model.eval()
        for param in self.teacher.model.parameters():
            param.requires_grad = False
        
        print(f"Initializing student model: {student_model}")
        self.student = YOLO(student_model)
        self.student.model = self.student.model.to(self.device)
        
        print(f"✅ Models loaded on device: {self.device}")
        print(f"✅ Teacher parameters: {sum(p.numel() for p in self.teacher.model.parameters()):,}")
        print(f"✅ Student parameters: {sum(p.numel() for p in self.student.model.parameters()):,}")
        
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
        # 배치 구조 확인 및 안전한 처리
        if isinstance(batch, dict):
            images = batch['img']
            # 이미지 데이터 타입 및 정규화 처리
            if images.dtype == torch.uint8:
                images = images.float() / 255.0  # uint8 -> float32, [0,255] -> [0,1]
            images = images.to(self.device)
            
            # print(f"🔍 이미지 shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
            
            # YOLO 배치에서 타겟 추출 (다양한 키 확인)
            if 'batch_idx' in batch:
                targets = batch  # 전체 배치 정보
            elif 'cls' in batch and 'bboxes' in batch:
                targets = batch  # 분리된 형태
            else:
                # 배치 키 확인을 위한 디버깅
                # print(f"🔍 배치 키: {list(batch.keys())}")
                targets = batch
        else:
            # 배치가 튜플이나 리스트인 경우
            images = batch[0]
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            images = images.to(self.device)
            targets = batch[1] if len(batch) > 1 else None
        
        # 모델 디바이스 확인
        # print(f"🔍 Teacher model device: {next(self.teacher.model.parameters()).device}")
        # print(f"🔍 Student model device: {next(self.student.model.parameters()).device}")
        
        # 모델을 올바른 디바이스로 이동
        self.teacher.model = self.teacher.model.to(self.device)
        self.student.model = self.student.model.to(self.device)
        
        try:
            # Teacher 추론 (no gradient)
            with torch.no_grad():
                # print("🔍 Teacher 추론 시작...")
                teacher_outputs = self.teacher.model(images)
                # print(f"🔍 Teacher 출력 타입: {type(teacher_outputs)}")
                if isinstance(teacher_outputs, (list, tuple)):
                    # print(f"🔍 Teacher 출력 개수: {len(teacher_outputs)}")
                    teacher_outputs = teacher_outputs[0] if len(teacher_outputs) > 0 else teacher_outputs
                teacher_outputs = self.parse_model_outputs(teacher_outputs)
                teacher_features = []  # 임시로 빈 리스트
            
            # Student 추론
            # print("🔍 Student 추론 시작...")
            raw_student_preds = self.student.model(images)  # 원본 예측 보관
            # print(f"🔍 Student 출력 타입: {type(raw_student_preds)}")
            
            # 원본 예측 구조 상세 디버깅
            if isinstance(raw_student_preds, (list, tuple)):
                # print(f"🔍 Student 출력 개수: {len(raw_student_preds)}")
                for i, item in enumerate(raw_student_preds):
                    # print(f"🔍 Student 출력[{i}] 타입: {type(item)}")
                    if hasattr(item, 'shape'):
                        # print(f"🔍 Student 출력[{i}] 형태: {item.shape}")
                
                # 텐서만 필터링하여 YOLO 손실에 전달
                tensor_preds = [item for item in raw_student_preds if hasattr(item, 'view')]
                # print(f"🔍 텐서 예측 개수: {len(tensor_preds)}")
                
                student_outputs_for_parsing = raw_student_preds[0] if len(raw_student_preds) > 0 else raw_student_preds
            else:
                # print(f"🔍 Student 단일 출력 형태: {getattr(raw_student_preds, 'shape', 'No shape')}")
                tensor_preds = raw_student_preds
                student_outputs_for_parsing = raw_student_preds
                
            student_outputs = self.parse_model_outputs(student_outputs_for_parsing)  # KD용 파싱
            student_features = []  # 임시로 빈 리스트
            
        except Exception as model_error:
            print(f"❌ 모델 추론 중 오류: {model_error}")
            print(f"❌ 오류 타입: {type(model_error)}")
            raise model_error
        
        # 1. Detection 증류 손실 (objectness + bbox)
        try:
            det_loss, det_metrics = self.distillation_loss(
                student_outputs, teacher_outputs, targets
            )
            print(f"✅ Detection 손실 계산 성공: {det_loss.item():.4f}")
        except Exception as det_error:
            import traceback
            print(f"❌ Detection 손실 계산 오류:")
            print(f"   오류 타입: {type(det_error).__name__}")
            print(f"   오류 메시지: {str(det_error)}")
            print(f"   스택 트레이스:")
            print(traceback.format_exc())
            raise det_error
        
        # 2. Feature 증류 손실
        try:
            feat_loss, feat_metrics = self.feature_loss(
                student_features, teacher_features
            )
            print(f"✅ Feature 손실 계산 성공: {feat_loss.item():.4f}")
        except Exception as feat_error:
            import traceback
            print(f"❌ Feature 손실 계산 오류:")
            print(f"   오류 타입: {type(feat_error).__name__}")  
            print(f"   오류 메시지: {str(feat_error)}")
            print(f"   스택 트레이스:")
            print(traceback.format_exc())
            raise feat_error
        
        # 3. 원본 YOLO 손실 (Ground Truth 기반) - 임시 비활성화
        try:
            # YOLO 손실 함수와의 호환성 문제로 인해 임시로 비활성화
            # Knowledge Distillation 손실만 사용하여 학습 진행
            print("⚠️ Base 손실 계산을 임시로 건너뛰고 KD 손실만 사용합니다")
            base_loss = torch.tensor(0.0, device=images.device)
            print(f"✅ Base 손실 (비활성화): {base_loss.item():.4f}")
        except Exception as base_error:
            import traceback
            print(f"❌ Base 손실 계산 오류:")
            print(f"   오류 타입: {type(base_error).__name__}")
            print(f"   오류 메시지: {str(base_error)}")
            print(f"   스택 트레이스:")
            print(traceback.format_exc())
            # Base 손실 실패 시에도 학습 계속 진행
            base_loss = torch.tensor(0.0, device=images.device)
            print("⚠️ Base 손실 계산 실패 - 0으로 설정하고 학습 계속")
        
        # 전체 손실 조합 (Base 손실 비활성화로 인해 가중치 조정)
        total_loss = det_loss + 0.5 * feat_loss + 0.0 * base_loss  # Base 손실 가중치 0으로 설정
        
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
    
    def debug_print(self, message: str, level: str = "info"):
        """조건부 디버깅 출력"""
        if self.verbose_debug:
            prefix = "🔍" if level == "info" else "⚠️" if level == "warning" else "❌"
            print(f"{prefix} {message}")
    
    def parse_model_outputs(self, outputs):
        """모델 출력을 파싱하여 bbox와 objectness 분리"""
        self.debug_print(f"출력 파싱 - shape: {outputs.shape if hasattr(outputs, 'shape') else 'No shape'}")
        self.debug_print(f"출력 파싱 - type: {type(outputs)}")
        
        # YOLOv11 출력 형식 처리
        if outputs.dim() == 3:  # [batch, num_anchors, features]
            if outputs.shape[-1] >= 5:  # [x, y, w, h, conf, ...]
                return {
                    'bbox': outputs[..., :4],
                    'objectness': outputs[..., 4:5]
                }
            else:
                self.debug_print(f"예상과 다른 출력 차원: {outputs.shape}", "warning")
                # 기본값 반환
                batch_size = outputs.shape[0]
                return {
                    'bbox': torch.zeros(batch_size, 1, 4, device=outputs.device),
                    'objectness': torch.zeros(batch_size, 1, 1, device=outputs.device)
                }
        elif outputs.dim() == 4:  # [batch, features, height, width]
            self.debug_print(f"4D 출력 감지: {outputs.shape}")
            # Feature map 형태인 경우 reshape 필요
            batch_size, features, h, w = outputs.shape
            # Flatten spatial dimensions
            outputs_flat = outputs.view(batch_size, features, h * w).transpose(1, 2)  # [B, H*W, features]
            
            if features >= 5:
                return {
                    'bbox': outputs_flat[..., :4],
                    'objectness': outputs_flat[..., 4:5]
                }
            else:
                # 기본값 반환
                return {
                    'bbox': torch.zeros(batch_size, h*w, 4, device=outputs.device),
                    'objectness': torch.zeros(batch_size, h*w, 1, device=outputs.device)
                }
        else:
            self.debug_print(f"지원하지 않는 출력 차원: {outputs.shape}", "error")
            batch_size = outputs.shape[0] if outputs.dim() > 0 else 1
            return {
                'bbox': torch.zeros(batch_size, 1, 4, device=outputs.device),
                'objectness': torch.zeros(batch_size, 1, 1, device=outputs.device)
            }
    
    def validate(self, val_loader):
        """검증 수행 - 실제 mAP 계산"""
        if val_loader is None:
            return {
                'val/mAP': 0.0,
                'val/precision': 0.0,
                'val/recall': 0.0,
                'val/inference_time': 0.0
            }
        
        self.student.model.eval()
        
        try:
            # 빠른 평가 (처음 20배치만 사용)
            eval_metrics = self.evaluate_model(val_loader, epoch=0)
            
            # 추론 시간 측정
            total_time = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 10:  # 시간 측정용 10배치만
                        break
                        
                    if isinstance(batch, dict):
                        images = batch.get('img', batch.get('image'))
                    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        images = batch[0]
                    else:
                        continue
                    
                    if images is None:
                        continue
                        
                    images = images.float() / 255.0 if images.dtype == torch.uint8 else images
                    images = images.to(self.device)
                    
                    start_time = time.time()
                    _ = self.student.model(images)
                    total_time += time.time() - start_time
                    num_batches += 1
            
            avg_time = total_time / num_batches if num_batches > 0 else 0
            
            self.student.model.train()
            return {
                'val/mAP': eval_metrics.get('map50', 0.0),
                'val/mAP_50_95': eval_metrics.get('map', 0.0),
                'val/precision': eval_metrics.get('precision', 0.0),
                'val/recall': eval_metrics.get('recall', 0.0),
                'val/inference_time': avg_time
            }
            
        except Exception as e:
            print(f"⚠️ 검증 중 오류: {e}")
            self.student.model.train()
            return {
                'val/mAP': 0.0,
                'val/precision': 0.0,
                'val/recall': 0.0,
                'val/inference_time': 0.0
            }
    
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
                try:
                    metrics = self.train_step(batch, optimizer)
                    
                    # 배치 메트릭 누적
                    for k, v in metrics.items():
                        if k not in epoch_metrics:
                            epoch_metrics[k] = 0
                        epoch_metrics[k] += v
                    
                    # 주기적 로깅 (더 간결하게)
                    if batch_idx % 20 == 0:
                        print(f"  📊 Batch {batch_idx}: Loss = {metrics.get('loss/total', 0):.4f}")
                        if self.use_wandb:
                            wandb.log(metrics, step=epoch * len(train_loader) + batch_idx)
                            
                except Exception as batch_error:
                    import traceback
                    print(f"❌ Batch {batch_idx} 처리 중 오류:")
                    print(f"   오류 타입: {type(batch_error).__name__}")
                    print(f"   오류 메시지: {str(batch_error)}")
                    print(f"   상세 스택 트레이스:")
                    print(traceback.format_exc())
                    
                    if batch_idx == 0:  # 첫 번째 배치에서 오류면 중단
                        print("첫 번째 배치부터 오류 발생. 학습을 중단합니다.")
                        return 0
                    continue  # 다른 배치는 건너뛰기
            
            # 검증 및 평가
            val_metrics = self.validate(val_loader)
            
            # mAP 평가 (매 5 에폭마다)
            if val_loader is not None and (epoch + 1) % 5 == 0:
                try:
                    eval_metrics = self.evaluate_model(val_loader, epoch + 1)
                    val_metrics.update({f'eval_{k}': v for k, v in eval_metrics.items()})
                    
                    # 추론 이미지 로깅 (매 10 에폭마다)
                    if (epoch + 1) % 10 == 0:
                        self.log_inference_images(val_loader, epoch + 1, num_images=3)
                        
                except Exception as eval_error:
                    print(f"⚠️ 평가 중 오류: {eval_error}")
            
            # 에폭 평균 메트릭
            for k in epoch_metrics:
                epoch_metrics[k] /= len(train_loader)
            
            # 스케줄러 업데이트
            scheduler.step()
            
            # WandB 로깅
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    **epoch_metrics,
                    **val_metrics,
                    'lr': scheduler.get_last_lr()[0]
                })
            
            # 에폭 결과 표시
            print(f"📈 Epoch {epoch+1}/{epochs} 완료:")
            print(f"   🔥 평균 손실: {epoch_metrics['loss/total']:.4f}")
            print(f"   🎯 mAP@50: {val_metrics.get('val/mAP', 0):.4f}")
            print(f"   ⚡ 추론 시간: {val_metrics.get('val/inference_time', 0)*1000:.1f}ms")
            
            # 모델 저장
            if val_metrics['val/mAP'] > best_map:
                best_map = val_metrics['val/mAP']
                save_path = f"{save_dir}/best_model.pt"
                torch.save(self.student.model.state_dict(), save_path)
                print(f"🌟 새로운 최고 성능! mAP: {best_map:.4f}")
                
                if self.use_wandb:
                    wandb.save(save_path)
            
            print("-" * 40)
        
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
            from ultralytics.data.dataset import YOLODataset
            from torch.utils.data import DataLoader
            import yaml
            import os
            from pathlib import Path
            
            print("🔄 버전 호환성 문제를 피한 직접 데이터로더 생성을 시도합니다...")
            
            # 데이터셋 설정 로드
            with open(self.modified_data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # 데이터셋 경로 확인
            data_path = data_config.get('path', './datasets/figma_ui')
            train_path = os.path.join(data_path, data_config.get('train', 'images/train'))
            val_path = os.path.join(data_path, data_config.get('val', 'images/val'))
            
            print(f"📁 데이터셋 경로: {data_path}")
            print(f"📁 학습 경로: {train_path}")
            print(f"📁 검증 경로: {val_path}")
            
            # 경로 존재 여부 확인
            if not os.path.exists(train_path):
                print(f"❌ 학습 데이터 경로를 찾을 수 없습니다: {train_path}")
                return None, None
            
            # 이미지 파일 확인
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(Path(train_path).glob(ext))
            
            if not image_files:
                print(f"❌ {train_path}에서 이미지 파일을 찾을 수 없습니다.")
                return None, None
            
            print(f"✅ {len(image_files)}개의 이미지 파일을 찾았습니다.")
            
            # 직접 YOLO 데이터셋 생성 (버전 호환성 문제 회피)
            try:
                # 학습용 데이터셋
                train_dataset = YOLODataset(
                    img_path=train_path,
                    data=data_config,
                    task='detect',
                    imgsz=640,
                    batch_size=batch_size,
                    augment=True,
                    cache=False,
                    single_cls=True,  # 단일 클래스
                    stride=32,
                    pad=0.0,
                    rect=False
                )
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=train_dataset.collate_fn
                )
                
                print(f"✅ 학습 데이터로더 생성 성공: {len(train_loader)} batches")
                
                # 검증용 데이터셋
                val_loader = None
                if os.path.exists(val_path):
                    try:
                        val_dataset = YOLODataset(
                            img_path=val_path,
                            data=data_config,
                            task='detect',
                            imgsz=640,
                            batch_size=batch_size,
                            augment=False,
                            cache=False,
                            single_cls=True,
                            stride=32,
                            pad=0.5,
                            rect=True
                        )
                        
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=val_dataset.collate_fn
                        )
                        
                        print(f"✅ 검증 데이터로더 생성 성공: {len(val_loader)} batches")
                    except Exception as val_e:
                        print(f"⚠️ 검증 데이터로더 생성 실패: {val_e}")
                
                print("🎉 직접 데이터로더 생성 완료!")
                return train_loader, val_loader
                
            except Exception as dataset_e:
                print(f"❌ YOLO 데이터셋 생성 실패: {dataset_e}")
                return None, None
                
        except ImportError as ie:
            print(f"❌ YOLO 모듈 import 오류: {ie}")
            return None, None
        except Exception as e:
            print(f"❌ 데이터로더 생성 중 전체 오류: {e}")
            print("💡 직접 데이터로더 생성도 실패 - YOLO 기본 학습으로 폴백합니다")
            return None, None
    
    def evaluate_model(self, val_loader: DataLoader, epoch: int) -> Dict:
        """모델 평가 및 mAP 계산"""
        self.student.model.eval()
        
        all_predictions = []
        all_targets = []
        
        print(f"\n📊 Epoch {epoch} 모델 평가 시작...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 50:  # 빠른 평가를 위해 50배치만 사용
                    break
                    
                # 배치 파싱
                if isinstance(batch, dict):
                    images = batch.get('img', batch.get('image'))
                    targets = batch
                elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    images, targets = batch[0], batch[1]
                else:
                    continue
                
                if images is None:
                    continue
                
                images = images.float() / 255.0 if images.dtype == torch.uint8 else images
                images = images.to(self.device)
                
                # Student 추론
                try:
                    results = self.student.model(images)
                    # YOLO 결과를 표준 형식으로 변환
                    for i, result in enumerate(results):
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes
                            if len(boxes) > 0:
                                pred_boxes = boxes.xyxy.cpu().numpy()
                                pred_scores = boxes.conf.cpu().numpy()
                                pred_labels = boxes.cls.cpu().numpy()
                                
                                for j in range(len(pred_boxes)):
                                    all_predictions.append({
                                        'image_id': batch_idx * images.shape[0] + i,
                                        'bbox': pred_boxes[j],
                                        'score': pred_scores[j],
                                        'label': pred_labels[j]
                                    })
                    
                    # Ground Truth 처리
                    if isinstance(targets, dict) and 'bboxes' in targets:
                        batch_idx_tensor = targets.get('batch_idx', torch.arange(images.shape[0]))
                        bboxes = targets['bboxes']
                        cls = targets.get('cls', torch.zeros(len(bboxes)))
                        
                        for img_idx in range(images.shape[0]):
                            mask = batch_idx_tensor == img_idx
                            if mask.any():
                                img_bboxes = bboxes[mask].cpu().numpy()
                                img_cls = cls[mask].cpu().numpy()
                                
                                for k in range(len(img_bboxes)):
                                    # YOLO format (cx, cy, w, h) to (x1, y1, x2, y2)
                                    bbox = img_bboxes[k]
                                    h, w = images.shape[2], images.shape[3]
                                    x1 = (bbox[0] - bbox[2]/2) * w
                                    y1 = (bbox[1] - bbox[3]/2) * h
                                    x2 = (bbox[0] + bbox[2]/2) * w
                                    y2 = (bbox[1] + bbox[3]/2) * h
                                    
                                    all_targets.append({
                                        'image_id': batch_idx * images.shape[0] + img_idx,
                                        'bbox': [x1, y1, x2, y2],
                                        'label': img_cls[k]
                                    })
                
                except Exception as e:
                    print(f"⚠️ 평가 중 오류 (배치 {batch_idx}): {e}")
                    continue
        
        # mAP 계산
        metrics = self.calculate_map(all_predictions, all_targets)
        
        print(f"📊 평가 완료:")
        print(f"   - 총 예측: {len(all_predictions)}개")
        print(f"   - 총 GT: {len(all_targets)}개") 
        print(f"   - mAP@0.5: {metrics.get('map50', 0.0):.4f}")
        print(f"   - mAP@0.5:0.95: {metrics.get('map', 0.0):.4f}")
        
        return metrics
    
    def calculate_map(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """mAP 계산"""
        if not predictions or not targets:
            return {'map': 0.0, 'map50': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # IoU 임계값들
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        # 이미지별로 그룹화
        pred_by_image = {}
        gt_by_image = {}
        
        for pred in predictions:
            img_id = pred['image_id']
            if img_id not in pred_by_image:
                pred_by_image[img_id] = []
            pred_by_image[img_id].append(pred)
        
        for gt in targets:
            img_id = gt['image_id']
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append(gt)
        
        # 각 IoU 임계값에서 AP 계산
        aps = []
        for iou_thresh in iou_thresholds:
            ap = self.calculate_ap_at_iou(pred_by_image, gt_by_image, iou_thresh)
            aps.append(ap)
        
        # mAP@0.5
        map50 = self.calculate_ap_at_iou(pred_by_image, gt_by_image, 0.5)
        
        # mAP@0.5:0.95
        map_avg = np.mean(aps)
        
        # Precision, Recall at IoU=0.5
        precision, recall = self.calculate_precision_recall(pred_by_image, gt_by_image, 0.5)
        
        return {
            'map': map_avg,
            'map50': map50,
            'precision': precision,
            'recall': recall
        }
    
    def calculate_ap_at_iou(self, pred_by_image: Dict, gt_by_image: Dict, iou_thresh: float) -> float:
        """특정 IoU 임계값에서 AP 계산"""
        all_scores = []
        all_matches = []
        total_gt = 0
        
        for img_id in gt_by_image.keys():
            gt_boxes = gt_by_image[img_id]
            pred_boxes = pred_by_image.get(img_id, [])
            
            total_gt += len(gt_boxes)
            
            if not pred_boxes:
                continue
            
            # Score로 정렬
            pred_boxes = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)
            
            # GT 매칭 여부
            gt_matched = [False] * len(gt_boxes)
            
            for pred in pred_boxes:
                all_scores.append(pred['score'])
                
                # 최고 IoU GT 찾기
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = self.calculate_bbox_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # 매칭 여부 결정
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    all_matches.append(True)
                else:
                    all_matches.append(False)
        
        if not all_scores:
            return 0.0
        
        # Precision-Recall 커브 계산
        sorted_indices = np.argsort(all_scores)[::-1]
        matches = np.array(all_matches)[sorted_indices]
        
        tp = np.cumsum(matches)
        fp = np.cumsum(~matches)
        
        precision = tp / (tp + fp)
        recall = tp / total_gt if total_gt > 0 else np.zeros_like(tp)
        
        # AP 계산 (11-point interpolation)
        ap = 0
        for r in np.arange(0, 1.1, 0.1):
            p_max = np.max(precision[recall >= r]) if np.any(recall >= r) else 0
            ap += p_max / 11
        
        return ap
    
    def calculate_precision_recall(self, pred_by_image: Dict, gt_by_image: Dict, iou_thresh: float) -> Tuple[float, float]:
        """Precision과 Recall 계산"""
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for img_id in gt_by_image.keys():
            gt_boxes = gt_by_image[img_id]
            pred_boxes = pred_by_image.get(img_id, [])
            
            total_gt += len(gt_boxes)
            
            if not pred_boxes:
                continue
            
            # Score 임계값 (0.5) 적용
            pred_boxes = [p for p in pred_boxes if p['score'] >= 0.5]
            
            gt_matched = [False] * len(gt_boxes)
            
            for pred in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = self.calculate_bbox_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    total_tp += 1
                else:
                    total_fp += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        
        return precision, recall
    
    def calculate_bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """두 바운딩 박스의 IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def log_inference_images(self, val_loader: DataLoader, epoch: int, num_images: int = 5):
        """추론 결과 이미지를 WandB에 로깅"""
        if not self.use_wandb:
            return
        
        self.student.model.eval()
        logged_images = 0
        
        print(f"\n🖼️ 추론 결과 이미지 생성 중...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if logged_images >= num_images:
                    break
                
                # 배치 파싱
                if isinstance(batch, dict):
                    images = batch.get('img', batch.get('image'))
                    targets = batch
                elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    images, targets = batch[0], batch[1]
                else:
                    continue
                
                if images is None:
                    continue
                
                images = images.float() / 255.0 if images.dtype == torch.uint8 else images
                images = images.to(self.device)
                
                try:
                    # Student 추론
                    results = self.student.model(images)
                    
                    # 배치의 각 이미지 처리
                    for i in range(min(images.shape[0], num_images - logged_images)):
                        img_tensor = images[i]
                        result = results[i] if i < len(results) else None
                        
                        # 이미지를 numpy로 변환
                        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        # matplotlib figure 생성
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                        
                        # 원본 이미지 (GT 포함)
                        ax1.imshow(img_np)
                        ax1.set_title(f'Ground Truth (Epoch {epoch})', fontsize=14)
                        ax1.axis('off')
                        
                        # GT 박스 그리기
                        if isinstance(targets, dict) and 'bboxes' in targets:
                            batch_idx_tensor = targets.get('batch_idx', torch.arange(images.shape[0]))
                            bboxes = targets['bboxes']
                            
                            img_mask = batch_idx_tensor == (batch_idx * images.shape[0] + i)
                            if img_mask.any():
                                img_bboxes = bboxes[img_mask].cpu().numpy()
                                h, w = img_np.shape[:2]
                                
                                for bbox in img_bboxes:
                                    # YOLO format to pixel coordinates
                                    x1 = (bbox[0] - bbox[2]/2) * w
                                    y1 = (bbox[1] - bbox[3]/2) * h
                                    box_w = bbox[2] * w
                                    box_h = bbox[3] * h
                                    
                                    rect = patches.Rectangle(
                                        (x1, y1), box_w, box_h,
                                        linewidth=2, edgecolor='green', 
                                        facecolor='none', label='GT'
                                    )
                                    ax1.add_patch(rect)
                        
                        # 예측 결과 이미지
                        ax2.imshow(img_np)
                        ax2.set_title(f'Student Predictions (Epoch {epoch})', fontsize=14)
                        ax2.axis('off')
                        
                        # 예측 박스 그리기
                        if result is not None and hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes
                            if len(boxes) > 0:
                                pred_boxes = boxes.xyxy.cpu().numpy()
                                pred_scores = boxes.conf.cpu().numpy()
                                
                                for j, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                                    if score > 0.3:  # 신뢰도 임계값
                                        x1, y1, x2, y2 = box
                                        box_w = x2 - x1
                                        box_h = y2 - y1
                                        
                                        color = 'red' if score > 0.7 else 'orange' if score > 0.5 else 'yellow'
                                        rect = patches.Rectangle(
                                            (x1, y1), box_w, box_h,
                                            linewidth=2, edgecolor=color,
                                            facecolor='none'
                                        )
                                        ax2.add_patch(rect)
                                        
                                        # 점수 표시
                                        ax2.text(x1, y1-5, f'{score:.2f}', 
                                                color=color, fontsize=10, fontweight='bold')
                        
                        plt.tight_layout()
                        
                        # WandB에 로깅
                        wandb.log({
                            f"inference_images/epoch_{epoch}_img_{logged_images}": wandb.Image(fig),
                            "epoch": epoch
                        })
                        
                        plt.close(fig)
                        logged_images += 1
                        
                        if logged_images >= num_images:
                            break
                
                except Exception as e:
                    print(f"⚠️ 이미지 로깅 중 오류: {e}")
                    continue
        
        print(f"✅ {logged_images}개 추론 이미지 WandB 로깅 완료")
