"""
빠른 YOLO 지식 증류 테스트 스크립트
"""
import os
import torch
from yolo_distillation.models.figma_ui_distillation import FigmaUIDistillation

# MPS 문제 방지
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def quick_test():
    print("=" * 50)
    print("🚀 빠른 YOLO 지식 증류 테스트")
    print("=" * 50)
    
    config = {
        'teacher_model': './weight/best_forest.pt',
        'student_model': 'yolo11s.yaml',
        'data_yaml': './data/yolo_dataset_unified/data.yaml',
        'device': 'cpu',  # CPU 강제 사용
        'epochs': 1,      # 1 에폭만
        'batch_size': 2,  # 매우 작은 배치
        'learning_rate': 0.001,
        'num_workers': 1,
        'max_train_samples': 10,  # 학습 10개만
        'max_val_samples': 5      # 검증 5개만
    }
    
    try:
        print("🔄 모델 초기화 중...")
        distiller = FigmaUIDistillation(
            teacher_model=config['teacher_model'],
            student_model=config['student_model'],
            data_yaml=config['data_yaml'],
            device=config['device'],
            use_wandb=False  # WandB 비활성화로 속도 향상
        )
        
        print("✅ 모델 초기화 완료!")
        print(f"🎯 매우 짧은 학습 시작: {config['epochs']} epoch, {config['max_train_samples']} samples")
        
        best_map = distiller.train(
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_workers=config['num_workers'],
            max_train_samples=config['max_train_samples'],
            max_val_samples=config['max_val_samples']
        )
        
        print(f"✅ 테스트 완료! Best mAP: {best_map:.4f}")
        print("🎉 Knowledge Distillation 파이프라인이 정상 작동합니다!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()