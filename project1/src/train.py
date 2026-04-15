import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 경로 설정 (src/ 한 단계 위가 프로젝트 루트)
# ─────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
LOGS_DIR    = os.path.join(RESULTS_DIR, 'logs')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)


# ─────────────────────────────────────────
# 학습 함수
# ─────────────────────────────────────────

#만들어둔 함수 불러오기
from dataset import get_dataset
#만들어둔 모델 클래스 불러오기
from models import LogisticRegression,MLP

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100
    
    print(f'  [Train] Epoch {epoch} | Loss: {avg_loss:.4f} | 정확도: {accuracy:.2f}%')
    return avg_loss, accuracy


# ─────────────────────────────────────────
# 테스트 함수
# ─────────────────────────────────────────
def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total * 100

    print(f'  [Test]  Loss: {avg_loss:.4f} | 정확도: {accuracy:.2f}%')
    return avg_loss, accuracy

# ─────────────────────────────────────────
# 학습 결과 그래프
# ─────────────────────────────────────────
def plot_results(history_lr, history_mlp):
    epochs = range(1, len(history_lr['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Logistic Regression vs MLP', fontsize=16, fontweight='bold')

    # Loss 그래프
    ax = axes[0]
    ax.plot(epochs, history_lr['train_loss'],  'b-',  label='LR Train')
    ax.plot(epochs, history_lr['test_loss'],   'b--', label='LR Test')
    ax.plot(epochs, history_mlp['train_loss'], 'r-',  label='MLP Train')
    ax.plot(epochs, history_mlp['test_loss'],  'r--', label='MLP Test')
    ax.set_title('Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy 그래프
    ax = axes[1]
    ax.plot(epochs, history_lr['train_acc'],  'b-',  label='LR Train')
    ax.plot(epochs, history_lr['test_acc'],   'b--', label='LR Test')
    ax.plot(epochs, history_mlp['train_acc'], 'r-',  label='MLP Train')
    ax.plot(epochs, history_mlp['test_acc'],  'r--', label='MLP Test')
    ax.set_title('Accuracy per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'그래프 저장 완료: {save_path}')


# ─────────────────────────────────────────
# 성능 비교 표
# ─────────────────────────────────────────
def plot_comparison_table(results):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')

    headers = ['Model', 'Train Acc (%)', 'Test Acc (%)', 'Train Loss', 'Test Loss']
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            f"{r['train_acc']:.2f}",
            f"{r['test_acc']:.2f}",
            f"{r['train_loss']:.4f}",
            f"{r['test_loss']:.4f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.4, 2.0)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        color = '#eaf0fb' if i % 2 == 0 else '#ffffff'
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'comparison_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'비교 표 저장 완료: {save_path}')


# ─────────────────────────────────────────
# 오류 분석 이미지
# ─────────────────────────────────────────
def plot_error_analysis(model, test_loader, model_name, n=12):
    model.eval()
    wrong_images, wrong_labels, wrong_preds = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            mask = (preds != labels)
            wrong_images.append(images[mask])
            wrong_labels.append(labels[mask])
            wrong_preds.append(preds[mask])
            if sum(len(w) for w in wrong_images) >= n:
                break

    wrong_images = torch.cat(wrong_images)[:n]
    wrong_labels = torch.cat(wrong_labels)[:n]
    wrong_preds  = torch.cat(wrong_preds)[:n]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f'{model_name} — Error Analysis', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis('off')
            continue
        img = wrong_images[i].squeeze().numpy()
        true_name = CLASS_NAMES[wrong_labels[i].item()]
        pred_name = CLASS_NAMES[wrong_preds[i].item()]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_name}\nPred: {pred_name}', fontsize=9, color='red')
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f'{model_name}_errors.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'오류 분석 저장 완료: {save_path}')


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────
def main(model_name='MLP', epochs=10, batch_size=32):
    train_dataset, test_dataset = get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    if model_name == 'LR':
        model = LogisticRegression()
    else:
        model = MLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print(f'\n{"="*40}')
    print(f'  {model_name} 학습 시작  (epochs={epochs})')
    print(f'{"="*40}')

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, criterion, optimizer, epoch)
        te_loss, te_acc = test(model, test_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

    # 모델 저장
    save_path = os.path.join(RESULTS_DIR, f'{model_name}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f'\n모델 저장 완료: {save_path}')

    return model, history, test_loader


# ─────────────────────────────────────────
if __name__ == '__main__':
    EPOCHS = 10

    model_lr,  history_lr,  test_loader = main('LR',  epochs=EPOCHS)
    model_mlp, history_mlp, _           = main('MLP', epochs=EPOCHS)

    plot_results(history_lr, history_mlp)

    final_results = {
        'LR': {
            'train_acc':  history_lr['train_acc'][-1],
            'test_acc':   history_lr['test_acc'][-1],
            'train_loss': history_lr['train_loss'][-1],
            'test_loss':  history_lr['test_loss'][-1],
        },
        'MLP': {
            'train_acc':  history_mlp['train_acc'][-1],
            'test_acc':   history_mlp['test_acc'][-1],
            'train_loss': history_mlp['train_loss'][-1],
            'test_loss':  history_mlp['test_loss'][-1],
        },
    }
    plot_comparison_table(final_results)

    plot_error_analysis(model_lr,  test_loader, 'LR',  n=12)
    plot_error_analysis(model_mlp, test_loader, 'MLP', n=12)
