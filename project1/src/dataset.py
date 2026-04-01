import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 클래스 이름
CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#데이터를 가져와서 저장하고 train,test용 반환
def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),           # 이미지 → 텐서
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    #train데이터
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    #test데이터
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset

#샘플 출력하는 함수
def show_samples(dataset, n=9):
    fig, axes = plt.subplots(3, 3, figsize=(6, 6)) #fig=전체 캔버스/ axes= 3*3 격자칸들
    # axes.flat= 격자칸을 일차원으로 펼친것/ ax= 이때의 각 칸
    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(CLASS_NAMES[label])
        ax.axis('off')
    plt.tight_layout() #여백 조정
    plt.savefig('./results/figures/samples.png') #파일로 저장하기
    plt.show() #화면에 띄우기
    

#이 파일을 직접 실행할때만 작동(import하면 작동 안함)
if __name__ == '__main__':
    train_dataset, test_dataset = get_dataset()
    print(f'학습 데이터: {len(train_dataset)}장')
    print(f'테스트 데이터: {len(test_dataset)}장')
    show_samples(train_dataset)