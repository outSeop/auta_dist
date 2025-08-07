"""
Figma UI 검증을 위한 YOLOv11 Knowledge Distillation 실행 스크립트
"""

import torch
import time
from yolo_distillation.models.figma_ui_distillation import FigmaUIDistillation


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
