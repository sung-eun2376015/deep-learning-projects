import torch.nn as nn
#nn=neural network

class LogisticRegression(nn.Model):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(28*28,10) #fc=fully connected
    def forward(self,x):
        x=x.view(-1,28*28) #1개의 이미지는 (28,28)으로 2차원인데 이걸 하나로 만들기
        return self.fc(x)
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),  # 입력층 → 은닉층1
            nn.ReLU(),
            nn.Linear(256, 128),      # 은닉층1 → 은닉층2
            nn.ReLU(),
            nn.Linear(128, 10)        # 은닉층2 → 출력층
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 이미지 펼치기
        return self.model(x)