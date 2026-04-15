import os
import torch
from torch.utils.data import DataLoader

from dataset import get_dataset, CLASS_NAMES
from models import LogisticRegression, MLP

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(ROOT, 'results')


# ─────────────────────────────────────────
# 저장된 모델 불러오기
# ─────────────────────────────────────────
def load_model(model_name):
    if model_name == 'LR':
        model = LogisticRegression()
    else:
        model = MLP()

    path = os.path.join(RESULTS_DIR, f'{model_name}_model.pth')
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    print(f'모델 불러오기 완료: {path}')
    return model


# ─────────────────────────────────────────
# 테스트셋 전체 평가
# ─────────────────────────────────────────
def evaluate(model, test_loader):
    correct = 0
    total = 0

    # 클래스별 정답/전체 카운트
    class_correct = [0] * 10
    class_total   = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

            # 클래스별 집계
            for label, pred in zip(labels, predicted):
                class_total[label.item()]   += 1
                if label == pred:
                    class_correct[label.item()] += 1

    overall_acc = correct / total * 100
    print(f'\n전체 정확도: {overall_acc:.2f}%\n')
    print(f'{"클래스":<15} {"정확도":>8}')
    print('-' * 25)
    for i, name in enumerate(CLASS_NAMES):
        acc = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
        print(f'{name:<15} {acc:>7.2f}%')

    return overall_acc


# ─────────────────────────────────────────
if __name__ == '__main__':
    _, test_dataset = get_dataset()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for model_name in ['LR', 'MLP']:
        print(f'\n{"="*40}')
        print(f'  {model_name} 평가')
        print(f'{"="*40}')
        model = load_model(model_name)
        evaluate(model, test_loader)
