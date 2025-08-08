"""
간결한 출력으로 Knowledge Distillation 실행
"""

import torch
import time
import sys
import os
from yolo_distillation.models.figma_ui_distillation import FigmaUIDistillation


def run_distillation_clean():
    """디버깅 출력을 최소화한 KD 실행"""
    
    # 설정
    config = {
        'teacher_model': '/content/auta_dist/weight/best_forest.pt',
        'student_model': 'yolo11s.yaml',
        'data_yaml': '/content/drive/MyDrive/Colab Notebooks/AUTA/data/yolo_dataset_webforest/data.yaml',
        'epochs': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_workers': 8
    }
    
    print("🚀 Figma UI Knowledge Distillation 시작")
    print("=" * 50)
    
    try:
        # 모델 초기화 (디버깅 비활성화)
        print("📦 모델 로딩 중...")
        distiller = FigmaUIDistillation(
            teacher_model=config['teacher_model'],
            student_model=config['student_model'], 
            data_yaml=config['data_yaml'],
            use_wandb=True,
            verbose_debug=False  # 디버깅 출력 완전 비활성화
        )
        print("✅ 모델 로딩 완료")
        print(f"   Teacher: {distiller.teacher.model.model[-1].nc if hasattr(distiller.teacher.model.model[-1], 'nc') else 'N/A'} classes")
        print(f"   Student: {distiller.student.model.model[-1].nc if hasattr(distiller.student.model.model[-1], 'nc') else 'N/A'} classes")
        
    except Exception as e:
        print(f"❌ 모델 초기화 실패: {e}")
        return
    
    try:
        # 학습 시작
        print(f"\n🎯 학습 시작 ({config['epochs']} epochs)")
        print("-" * 30)
        
        best_map = distiller.train(
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_workers=config['num_workers']
        )
        
        print("\n🎉 학습 완료!")
        print(f"📊 Best mAP: {best_map:.4f}")
        
    except Exception as e:
        print(f"❌ 학습 중 오류: {e}")
        best_map = 0.0
    
    # 성능 요약
    print("\n" + "=" * 50)
    print("📈 Knowledge Distillation 결과 요약")
    print("-" * 30)
    print(f"Teacher: YOLOv11-l (25M params)")
    print(f"Student: YOLOv11-s (9M params) - mAP: {best_map:.4f}")
    print(f"압축률: ~62% 파라미터 감소")
    print(f"WandB: https://wandb.ai/ 에서 상세 결과 확인")


if __name__ == "__main__":
    run_distillation_clean()
