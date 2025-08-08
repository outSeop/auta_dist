"""
Figma UI 검증을 위한 YOLOv11 증류 메인 클래스 (리팩토링 버전)
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
import time

from ..losses.single_class_loss import SingleClassDistillationLoss
from ..losses.feature_alignment_loss import FeatureAlignmentLoss


class FigmaUIDistillation:
    """Figma UI 검증을 위한 YOLOv11 증류 메인 클래스 (리팩토링 버전)"""
    
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
        
        # 데이터셋 설정 로드
        self.load_dataset_config(data_yaml)
        
        # Knowledge Distillation 손실 함수들
        self.distillation_loss = SingleClassDistillationLoss(
            alpha=0.5, beta=0.5, temperature=4.0, device=self.device
        )
        self.feature_loss = FeatureAlignmentLoss()
        
        # WandB 초기화
        if self.use_wandb:
            self.init_wandb()
    
    def load_dataset_config(self, data_yaml: str):
        """데이터셋 설정 로드 및 수정"""
        if not os.path.exists(data_yaml):
            print(f"⚠️ 데이터 설정 파일을 찾을 수 없습니다: {data_yaml}")
            self.data_yaml = data_yaml
            self.modified_data_yaml = data_yaml
            return
        
        # 원본 설정 로드
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # single-class 설정으로 수정
        self.data_config['nc'] = 1
        self.data_config['names'] = {0: 'ui_component'}
        
        if 'channels' not in self.data_config:
            self.data_config['channels'] = 3
        if 'imgsz' not in self.data_config:
            self.data_config['imgsz'] = 640
        
        # 수정된 설정 저장
        modified_path = data_yaml.replace('.yaml', '_modified.yaml')
        with open(modified_path, 'w') as f:
            yaml.dump(self.data_config, f)
        
        self.data_yaml = data_yaml
        self.modified_data_yaml = modified_path
    
    def init_wandb(self):
        """WandB 초기화"""
        try:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb.init(
                project="figma-ui-yolo11-kd",
                name=f"distill_{current_time}",
                config={
                    'teacher': 'yolov11l',
                    'student': 'yolov11s',
                    'task': 'figma_ui_detection',
                    'method': 'knowledge_distillation'
                }
            )
        except Exception as e:
            print(f"⚠️ WandB 초기화 실패: {e}")
            self.use_wandb = False
    
    def train_step(self, batch, optimizer) -> Dict:
        """단일 학습 스텝 (리팩토링)"""
        # 배치 파싱
        images, targets = self._parse_batch(batch)
        if images is None:
            return {}
        
        # 모델 추론
        try:
            teacher_outputs = self._get_teacher_predictions(images)
            student_outputs, raw_student_preds = self._get_student_predictions(images)
        except Exception as e:
            print(f"❌ 모델 추론 중 오류: {e}")
            return {}
        
        # 손실 계산
        try:
            # 1. Detection 증류 손실
            det_loss, det_metrics = self.distillation_loss(
                student_outputs, teacher_outputs, targets
            )
            
            # 2. Feature 증류 손실 (현재 비활성화)
            feat_loss = torch.tensor(0.0, device=images.device)
            feat_metrics = {'feature_total': 0.0}
            
            # 3. Base 손실 (현재 비활성화)
            base_loss = torch.tensor(0.0, device=images.device)
            
            # 전체 손실 조합
            total_loss = det_loss + 0.5 * feat_loss + 0.0 * base_loss
            
            # 역전파
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 메트릭 반환
            return {
                'loss/total': total_loss.item(),
                'loss/detection': det_loss.item(),
                'loss/feature': feat_loss.item(),
                'loss/base': base_loss.item(),
                **det_metrics,
                **feat_metrics
            }
            
        except Exception as e:
            print(f"❌ 손실 계산 중 오류: {e}")
            return {}
    
    def _parse_batch(self, batch):
        """배치 데이터 파싱"""
        if isinstance(batch, dict):
            images = batch.get('img', batch.get('image'))
            targets = batch
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:
            return None, None
        
        if images is None:
            return None, None
        
        # 이미지 전처리
        images = images.float() / 255.0 if images.dtype == torch.uint8 else images
        images = images.to(self.device)
        
        return images, targets
    
    def _get_teacher_predictions(self, images):
        """Teacher 모델 예측"""
        teacher_outputs = self.teacher.model(images)
        if isinstance(teacher_outputs, (list, tuple)) and len(teacher_outputs) > 0:
            teacher_outputs = teacher_outputs[0]
        return self.parse_model_outputs(teacher_outputs)
    
    def _get_student_predictions(self, images):
        """Student 모델 예측"""
        raw_student_preds = self.student.model(images)
        
        # 텐서만 필터링
        if isinstance(raw_student_preds, (list, tuple)):
            tensor_preds = [item for item in raw_student_preds if hasattr(item, 'view')]
            student_outputs_for_parsing = raw_student_preds[0] if len(raw_student_preds) > 0 else raw_student_preds
        else:
            tensor_preds = raw_student_preds
            student_outputs_for_parsing = raw_student_preds
        
        parsed_outputs = self.parse_model_outputs(student_outputs_for_parsing)
        return parsed_outputs, tensor_preds
    
    def parse_model_outputs(self, outputs):
        """모델 출력을 파싱하여 bbox와 objectness 분리"""
        if outputs.dim() == 3:  # [batch, num_anchors, features]
            if outputs.shape[-1] >= 5:  # [x, y, w, h, conf, ...]
                return {
                    'bbox': outputs[..., :4],
                    'objectness': outputs[..., 4:5]
                }
            else:
                batch_size = outputs.shape[0]
                return {
                    'bbox': torch.zeros(batch_size, 1, 4, device=outputs.device),
                    'objectness': torch.zeros(batch_size, 1, 1, device=outputs.device)
                }
        elif outputs.dim() == 4:  # [batch, features, height, width]
            batch_size, features, h, w = outputs.shape
            outputs_flat = outputs.view(batch_size, features, h * w).transpose(1, 2)
            
            if features >= 5:
                return {
                    'bbox': outputs_flat[..., :4],
                    'objectness': outputs_flat[..., 4:5]
                }
            else:
                return {
                    'bbox': torch.zeros(batch_size, h*w, 4, device=outputs.device),
                    'objectness': torch.zeros(batch_size, h*w, 1, device=outputs.device)
                }
        else:
            batch_size = outputs.shape[0] if outputs.dim() > 0 else 1
            return {
                'bbox': torch.zeros(batch_size, 1, 4, device=outputs.device),
                'objectness': torch.zeros(batch_size, 1, 1, device=outputs.device)
            }
    
    def validate(self, val_loader):
        """검증 수행 - bbox 출력 및 실제 mAP 계산"""
        if val_loader is None:
            print("⚠️ Validation loader가 없습니다.")
            return {
                'val/mAP': 0.0,
                'val/precision': 0.0,
                'val/recall': 0.0,
                'val/inference_time': 0.0
            }
        
        print(f"\n🔍 Validation 시작...")
        self.student.model.eval()
        
        try:
            # 상세한 평가 수행
            eval_metrics = self.evaluate_model(val_loader, epoch=0)
            
            # 추론 시간 측정
            print(f"⏱️ 추론 시간 측정 중...")
            total_time = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 5:  # 적은 수로 빠른 측정
                        break
                        
                    images, _ = self._parse_batch(batch)
                    if images is None:
                        continue
                    
                    start_time = time.time()
                    results = self.student.model(images)
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    num_batches += 1
                    
                    # 첫 번째 배치에서 상세 출력
                    if batch_idx == 0:
                        print(f"💡 배치 {batch_idx}: {inference_time*1000:.2f}ms, 이미지 {images.shape[0]}개")
                        if hasattr(results, '__iter__') and not isinstance(results, torch.Tensor):
                            for i, result in enumerate(results):
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    num_detections = len(result.boxes)
                                    print(f"   📦 이미지 {i}: {num_detections}개 검출")
            
            avg_time = total_time / num_batches if num_batches > 0 else 0
            
            print(f"✅ Validation 완료!")
            print(f"   🎯 mAP@0.5: {eval_metrics.get('map50', 0.0):.4f}")
            print(f"   🎯 mAP@0.5:0.95: {eval_metrics.get('map', 0.0):.4f}")
            print(f"   🔍 Precision: {eval_metrics.get('precision', 0.0):.4f}")
            print(f"   🔍 Recall: {eval_metrics.get('recall', 0.0):.4f}")
            print(f"   ⏱️ 평균 추론 시간: {avg_time*1000:.2f}ms")
            
            # WandB에 bbox 이미지 로깅
            if self.use_wandb:
                self.log_inference_images(val_loader, epoch=0, num_images=3)
            
            self.student.model.train()
            return {
                'val/mAP': eval_metrics.get('map50', 0.0),
                'val/mAP_50_95': eval_metrics.get('map', 0.0),
                'val/precision': eval_metrics.get('precision', 0.0),
                'val/recall': eval_metrics.get('recall', 0.0),
                'val/inference_time': avg_time
            }
            
        except Exception as e:
            print(f"❌ 검증 중 오류: {e}")
            import traceback
            traceback.print_exc()
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
              num_workers: int = 2,
              save_dir: str = './runs'):
        """Knowledge Distillation 학습 실행"""
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 데이터로더 생성
        train_loader, val_loader = self._create_dataloaders(batch_size, num_workers)
        
        if train_loader is None:
            print("⚠️ 데이터로더 생성 실패 - YOLO 기본 학습으로 폴백")
            return self._train_with_yolo_default(epochs, batch_size, learning_rate, save_dir)
        
        # 옵티마이저 및 스케줄러 설정
        optimizer = torch.optim.AdamW(self.student.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_map = 0.0
        
        print(f"🎯 Knowledge Distillation 학습 시작: {epochs} epochs")
        print("-" * 50)
        
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
                    
                    # 주기적 로깅 (간결하게)
                    if batch_idx % 20 == 0:
                        print(f"  📊 Batch {batch_idx}: Loss = {metrics.get('loss/total', 0):.4f}")
                        if self.use_wandb:
                            wandb.log(metrics, step=epoch * len(train_loader) + batch_idx)
                            
                except Exception as batch_error:
                    print(f"❌ Batch {batch_idx} 처리 중 오류: {batch_error}")
                    if batch_idx == 0:
                        print("첫 번째 배치부터 오류 발생. 학습을 중단합니다.")
                        return 0
                    continue
            
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
            
            # 에폭 결과 표시
            print(f"📈 Epoch {epoch+1}/{epochs} 완료:")
            print(f"   🔥 평균 손실: {epoch_metrics['loss/total']:.4f}")
            print(f"   🎯 mAP@50: {val_metrics.get('val/mAP', 0):.4f}")
            print(f"   ⚡ 추론 시간: {val_metrics.get('val/inference_time', 0)*1000:.1f}ms")
            
            # WandB 로깅
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    **epoch_metrics,
                    **val_metrics,
                    'lr': scheduler.get_last_lr()[0]
                })
            
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
    
    def _create_dataloaders(self, batch_size, num_workers):
        """데이터로더 생성 (간소화)"""
        try:
            from ultralytics.data.dataset import YOLODataset
            from torch.utils.data import DataLoader
            import yaml
            
            with open(self.modified_data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            data_path = data_config.get('path', './datasets/figma_ui')
            train_path = os.path.join(data_path, data_config.get('train', 'images/train'))
            val_path = os.path.join(data_path, data_config.get('val', 'images/val'))
            
            if not os.path.exists(train_path):
                print(f"❌ 학습 데이터 경로를 찾을 수 없습니다: {train_path}")
                return None, None
            
            # 데이터셋 생성
            train_dataset = YOLODataset(
                img_path=train_path,
                data=data_config,
                task='detect',
                imgsz=640,
                batch_size=batch_size,
                augment=True,
                cache=False,
                single_cls=True,
                stride=32
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=min(num_workers, 2),  # 시스템 권장 worker 수
                pin_memory=True,
                drop_last=True,
                collate_fn=train_dataset.collate_fn
            )
            
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
                        stride=32
                    )
                    
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=min(num_workers, 2),
                        pin_memory=True,
                        drop_last=False,
                        collate_fn=val_dataset.collate_fn
                    )
                    
                except Exception as val_e:
                    print(f"⚠️ 검증 데이터로더 생성 실패: {val_e}")
            
            print(f"✅ 데이터로더 생성 완료!")
            print(f"   학습: {len(train_loader)} batches")
            if val_loader:
                print(f"   검증: {len(val_loader)} batches")
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"❌ 데이터로더 생성 실패: {e}")
            return None, None
    
    def _train_with_yolo_default(self, epochs, batch_size, learning_rate, save_dir):
        """YOLO 기본 학습 방식 폴백"""
        print("🔄 YOLO 기본 학습 방식으로 전환...")
        
        try:
            results = self.student.train(
                data=self.modified_data_yaml,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                device=self.device,
                project=save_dir,
                name='student_model',
                exist_ok=True,
                verbose=True
            )
            return float(results.results_dict.get('metrics/mAP50(B)', 0))
        except Exception as e:
            print(f"❌ YOLO 기본 학습도 실패: {e}")
            return 0.0
    
    # 평가 및 시각화 메서드들은 기존과 동일하게 유지
    def evaluate_model(self, val_loader, epoch: int) -> Dict:
        """모델 평가 및 mAP 계산"""
        self.student.model.eval()
        
        all_predictions = []
        all_targets = []
        
        print(f"\n📊 Epoch {epoch} 모델 평가 시작...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 20:  # 빠른 평가를 위해 20배치만 사용
                    break
                    
                # 배치 파싱
                images, targets = self._parse_batch(batch)
                if images is None:
                    continue
                
                # Student 추론
                try:
                    results = self.student.model(images)
                    
                    # YOLO 결과를 표준 형식으로 변환
                    if hasattr(results, '__iter__') and not isinstance(results, torch.Tensor):
                        # Ultralytics YOLO 결과
                        for i, result in enumerate(results):
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                boxes = result.boxes
                                if len(boxes) > 0:
                                    pred_boxes = boxes.xyxy.cpu().numpy()
                                    pred_scores = boxes.conf.cpu().numpy()
                                    pred_labels = boxes.cls.cpu().numpy()
                                    
                                    # bbox 출력 로깅
                                    print(f"🎯 이미지 {batch_idx}_{i}: {len(pred_boxes)}개 객체 검출")
                                    for j in range(min(3, len(pred_boxes))):  # 상위 3개만 출력
                                        bbox = pred_boxes[j]
                                        print(f"   - Box {j}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}] conf:{pred_scores[j]:.3f}")
                                    
                                    for j in range(len(pred_boxes)):
                                        all_predictions.append({
                                            'image_id': batch_idx * images.shape[0] + i,
                                            'bbox': pred_boxes[j],
                                            'score': pred_scores[j],
                                            'label': pred_labels[j]
                                        })
                    else:
                        # Raw tensor 출력 처리
                        print(f"⚠️ Raw tensor 출력 감지 - 형식 변환 필요")
                    
                    # Ground Truth 처리
                    if isinstance(targets, dict):
                        self._process_ground_truth(targets, images, batch_idx, all_targets)
                
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
        
        self.student.model.train()
        return metrics
    
    def _process_ground_truth(self, targets, images, batch_idx, all_targets):
        """Ground Truth 데이터 처리"""
        try:
            if 'bboxes' in targets:
                batch_idx_tensor = targets.get('batch_idx', torch.arange(images.shape[0]))
                bboxes = targets['bboxes']
                cls = targets.get('cls', torch.zeros(len(bboxes)))
                
                for img_idx in range(images.shape[0]):
                    mask = batch_idx_tensor == img_idx
                    if mask.any():
                        img_bboxes = bboxes[mask].cpu().numpy()
                        img_cls = cls[mask].cpu().numpy()
                        
                        print(f"📋 GT 이미지 {batch_idx}_{img_idx}: {len(img_bboxes)}개 객체")
                        
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
            print(f"⚠️ GT 처리 중 오류: {e}")
    
    def calculate_map(self, predictions, targets):
        """mAP 계산"""
        if not predictions or not targets:
            print("⚠️ 예측 또는 GT가 없어서 mAP를 계산할 수 없습니다.")
            return {'map': 0.0, 'map50': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        import numpy as np
        
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
    
    def calculate_ap_at_iou(self, pred_by_image, gt_by_image, iou_thresh):
        """특정 IoU 임계값에서 AP 계산"""
        import numpy as np
        
        all_scores = []
        all_matches = []
        total_gt = 0
        
        for img_id in gt_by_image:
            gt_boxes = gt_by_image[img_id]
            pred_boxes = pred_by_image.get(img_id, [])
            
            total_gt += len(gt_boxes)
            
            if not pred_boxes:
                continue
            
            # 예측을 신뢰도 순으로 정렬
            pred_boxes = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)
            
            matched_gt = [False] * len(gt_boxes)
            
            for pred in pred_boxes:
                all_scores.append(pred['score'])
                
                # 가장 높은 IoU를 가진 GT 찾기
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if matched_gt[gt_idx]:
                        continue
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # 매칭 확인
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    matched_gt[best_gt_idx] = True
                    all_matches.append(True)
                else:
                    all_matches.append(False)
        
        if not all_scores:
            return 0.0
        
        # Precision-Recall 곡선 계산
        all_scores = np.array(all_scores)
        all_matches = np.array(all_matches)
        
        # 정렬
        sort_idx = np.argsort(-all_scores)
        all_matches = all_matches[sort_idx]
        
        # 누적 TP, FP 계산
        tp = np.cumsum(all_matches)
        fp = np.cumsum(~all_matches)
        
        # Precision, Recall 계산
        precision = tp / (tp + fp)
        recall = tp / total_gt if total_gt > 0 else np.zeros_like(tp)
        
        # AP 계산 (11-point interpolation)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            p_max = np.max(precision[recall >= t]) if np.any(recall >= t) else 0
            ap += p_max / 11
        
        return ap
    
    def calculate_precision_recall(self, pred_by_image, gt_by_image, iou_thresh=0.5):
        """Precision과 Recall 계산"""
        import numpy as np
        
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for img_id in gt_by_image:
            gt_boxes = gt_by_image[img_id]
            pred_boxes = pred_by_image.get(img_id, [])
            
            total_gt += len(gt_boxes)
            
            if not pred_boxes:
                continue
            
            matched_gt = [False] * len(gt_boxes)
            
            for pred in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if matched_gt[gt_idx]:
                        continue
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    matched_gt[best_gt_idx] = True
                    total_tp += 1
                else:
                    total_fp += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        
        return precision, recall
    
    def calculate_iou(self, box1, box2):
        """IoU 계산"""
        # box format: [x1, y1, x2, y2]
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

    def log_inference_images(self, val_loader, epoch: int, num_images: int = 5):
        """추론 결과 이미지를 bbox와 함께 WandB에 로깅"""
        if not self.use_wandb:
            return
        
        print(f"🖼️ {num_images}개 추론 이미지 로깅 (에폭 {epoch})")
        
        try:
            import wandb
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            from PIL import Image
            
            self.student.model.eval()
            wandb_images = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if len(wandb_images) >= num_images:
                        break
                    
                    # 배치 파싱
                    images, targets = self._parse_batch(batch)
                    if images is None:
                        continue
                    
                    # Student 추론
                    results = self.student.model(images)
                    
                    # 각 이미지 처리
                    for img_idx in range(min(images.shape[0], num_images - len(wandb_images))):
                        # 원본 이미지 복원 (0-1 정규화된 이미지를 0-255로)
                        img_tensor = images[img_idx].cpu()
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        
                        # matplotlib figure 생성
                        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                        ax.imshow(img_np)
                        ax.set_title(f'Epoch {epoch} - Image {batch_idx}_{img_idx}')
                        
                        # 예측 결과 bbox 그리기 (빨간색)
                        if hasattr(results, '__iter__') and not isinstance(results, torch.Tensor):
                            if img_idx < len(results):
                                result = results[img_idx]
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    boxes = result.boxes
                                    if len(boxes) > 0:
                                        pred_boxes = boxes.xyxy.cpu().numpy()
                                        pred_scores = boxes.conf.cpu().numpy()
                                        
                                        for j, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                                            if score > 0.3:  # 신뢰도 임계값
                                                x1, y1, x2, y2 = box
                                                width = x2 - x1
                                                height = y2 - y1
                                                
                                                # 예측 bbox (빨간색)
                                                rect = patches.Rectangle(
                                                    (x1, y1), width, height,
                                                    linewidth=2, edgecolor='red', 
                                                    facecolor='none', linestyle='-'
                                                )
                                                ax.add_patch(rect)
                                                
                                                # 신뢰도 점수 표시
                                                ax.text(x1, y1-5, f'Pred: {score:.2f}', 
                                                       color='red', fontsize=10, 
                                                       bbox=dict(boxstyle="round,pad=0.3", 
                                                               facecolor='white', alpha=0.7))
                        
                        # Ground Truth bbox 그리기 (초록색)
                        if isinstance(targets, dict) and 'bboxes' in targets:
                            batch_idx_tensor = targets.get('batch_idx', torch.arange(images.shape[0]))
                            bboxes = targets['bboxes']
                            
                            # 현재 이미지의 GT bbox 찾기
                            mask = batch_idx_tensor == img_idx
                            if mask.any():
                                img_bboxes = bboxes[mask].cpu().numpy()
                                
                                for k, bbox in enumerate(img_bboxes):
                                    # YOLO format (cx, cy, w, h) to (x1, y1, x2, y2)
                                    h, w = img_np.shape[:2]
                                    cx, cy, bw, bh = bbox
                                    x1 = (cx - bw/2) * w
                                    y1 = (cy - bh/2) * h
                                    x2 = (cx + bw/2) * w
                                    y2 = (cy + bh/2) * h
                                    
                                    width = x2 - x1
                                    height = y2 - y1
                                    
                                    # GT bbox (초록색)
                                    rect = patches.Rectangle(
                                        (x1, y1), width, height,
                                        linewidth=2, edgecolor='green', 
                                        facecolor='none', linestyle='--'
                                    )
                                    ax.add_patch(rect)
                                    
                                    # GT 라벨 표시
                                    ax.text(x1, y2+15, f'GT', 
                                           color='green', fontsize=10,
                                           bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor='white', alpha=0.7))
                        
                        # 범례 추가
                        red_patch = patches.Patch(color='red', label='Prediction')
                        green_patch = patches.Patch(color='green', label='Ground Truth')
                        ax.legend(handles=[red_patch, green_patch], loc='upper right')
                        
                        ax.axis('off')
                        plt.tight_layout()
                        
                        # WandB에 추가
                        wandb_images.append(wandb.Image(fig, caption=f"Epoch {epoch} - Inference {batch_idx}_{img_idx}"))
                        plt.close(fig)  # 메모리 정리
                        
                        if len(wandb_images) >= num_images:
                            break
            
            # WandB에 로깅
            if wandb_images:
                wandb.log({f"validation_images_epoch_{epoch}": wandb_images}, step=epoch)
                print(f"✅ {len(wandb_images)}개 이미지를 WandB에 로깅했습니다.")
            else:
                print("⚠️ 로깅할 이미지가 없습니다.")
            
            self.student.model.train()
            
        except Exception as e:
            print(f"❌ WandB 이미지 로깅 중 오류: {e}")
            import traceback
            traceback.print_exc()
