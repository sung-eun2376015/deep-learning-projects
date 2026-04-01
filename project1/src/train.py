import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_dataset
#만들어둔 모델 클래스 불러오기
from models import LogisticRegression,MLP

def train(model,train_loader,criterion,optimizer,epoch):
    model.train()
    total_loss=0
    correct=0
    total=0

    #이걸 한번 돌면 배치 1번 처리하는거고 다 돌면 1에포크가 끝난것
    for images,labels in train_loader:
        #기울기 초기화
        optimizer.zero_grad()
        #예측
        outputs=model(images)
        #손실 계산
        loss=criterion(outputs,labels)
        #역전파
        loss.backward()
        #가중치 업데이트
        optimizer.step()

        total_loss+=loss.item() #이번 배치 32장의 loss를 평균낸값을 꺼냄-> 다 더함
        predicted=outputs.argmax(dim=1) #ouputs배열=(32,10) -> predicted배열=(1,32)
        
        correct+=(predicted == labels).sum().item() #이번 배치에 정답과 동일한 이미지가 몇개인지-> 다 더함
        total += labels.size(0) #이번 배치에 이미지가 몇장인지-> 다 더함
    
    # 정확도
    accuracy = correct / total * 100
    # 1에포크의 loss 평균 
    avg_loss = total_loss / len(train_loader)
    
    print(f'Epoch {epoch} | Loss: {avg_loss:.4f} | 정확도: {accuracy:.2f}%')
    
    return avg_loss, accuracy




def main(model_name='MLP',epochs=10,batch_size=32):
    train_dataset,_=get_dataset()
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    if model_name=='LR':
        model=LogisticRegression()
    else:
        model=MLP()
    
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters())

    print(f'\n--- {model_name} 학습 시작 ---')
    
    #에포크만큼 학습을 반복
    for epoch in range(1,epoch+1):
        train(model,train_loader,criterion,optimizer)
    
    torch.save(model.state_dict(), f'./results/{model_name}_model.pth')
    print(f'\n모델 저장 완료!')


if __name__ == '__main__':
    main('LR')   # Logistic Regression 학습
    main('MLP')  # MLP 학습