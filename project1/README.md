# Fashion-MNIST 분류: Logistic Regression vs MLP

## 프로젝트 소개

Fashion-MNIST(의류 이미지 10종류)를 Logistic Regression과 MLP로 각각 분류하고 성능을 비교합니다.

## 목표

- 베이스라인 모델(Logistic Regression)과 MLP 직접 구현
- 두 모델의 정확도, 학습 곡선 비교
- 오류 분석을 통해 모델 한계 파악

## 환경

- Python 3.11
- PyTorch, torchvision, matplotlib

## 설치 및 실행

pip install torch torchvision matplotlib jupyter

# 데이터 확인

python src/dataset.py

# 학습

python src/train.py

# 평가

python src/evaluate.py

## 결과
| Model | Train Acc | Test Acc | Train Loss | Test Loss |
|-------|-----------|----------|------------|-----------|
| LR    | 85.80%    | 83.10%   | 0.4039     | 0.4764    |
| MLP   | **91.49%**| **87.58%**| **0.2242** | **0.3573** |

| 클래스 | LR | MLP |
|:-------|---:|----:|
| T-shirt | 74.90% | 83.80% |
| Trouser | 97.00% | 97.10% |
| Pullover | 73.30% | 78.90% |
| Dress | 84.40% | 90.00% |
| Coat | 66.70% | 77.10% |
| Sandal | 91.50% | 96.40% |
| Shirt | 65.30% | 70.50% |
| Sneaker | 96.60% | 91.60% |
| Bag | 93.50% | 95.60% |
| Ankle boot | 87.80% | 94.80% |

주요 분석 결과

- MLP가 모든 지표에서 LR을 능가 (Test Acc 기준 +4.48%p)
- 공통 약점: Shirt, Coat, Pullover, T-shirt 등 상의류끼리 혼동이 잦음
- 특이점: Sneaker 클래스에서만 LR(96.60%)이 MLP(91.60%)보다 높음
- LR은 Epoch이 늘어도 성능 향상이 거의 없는 반면, MLP는 지속적으로 향상

## 배운 점

- 선형 모델(LR)은 구조가 단순해 학습이 빠르게 포화되고, 픽셀 간의 관계를 학습하지 못해 비슷한 형태의 클래스 구분에 한계가 있음
- MLP는 은닉층을 통해 비선형 패턴을 학습할 수 있어 전반적으로 높은 성능을 보임
- 두 모델 모두 상의류 혼동 문제가 남아 있으며, 이를 해결하려면 CNN처럼 이미지의 공간적 구조를 활용하는 모델이 필요함
- 학습 곡선을 보면 MLP는 아직 수렴 전으로, Epoch 증가나 Dropout 추가 시 추가 성능 향상 여지가 있음
