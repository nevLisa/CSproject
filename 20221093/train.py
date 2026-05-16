import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import timm

# 우리가 만든 파일과 클래스 불러오기
from Utils.data_manager import PlantDataManager
from Utils.dataset import PlantDataset


# 뇌(ResNet) 하나에 출구만 2개로 개조하는 네트워크 클래스
class MultiTaskPlantModel(nn.Module):
    def __init__(self, base_model_name='resnet50', num_plants=2, num_parts=2):
        super(MultiTaskPlantModel, self).__init__()
        # timm에서 기본 모델 가져오기 (출력 층을 비워두기 위해 num_classes=0 설정)
        self.base_model = timm.create_model(base_model_name, pretrained=True, num_classes=0)

        # 모델의 최종 출력 특징(Feature) 벡터 크기 알아내기
        num_features = self.base_model.num_features

        # 뇌세포 끝에 출구 2개 이어 붙이기
        self.plant_classifier = nn.Linear(num_features, num_plants)  # 식물 종류 출구
        self.part_classifier = nn.Linear(num_features, num_parts)  # 식물 부위 출구

    def forward(self, x):
        # 이미지를 보고 특징 추출
        features = self.base_model(x)

        # 추출된 특징을 바탕으로 두 가지 정답 예측
        plant_out = self.plant_classifier(features)
        part_out = self.part_classifier(features)

        return plant_out, part_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 사용 중인 장치: {device}")

    # 경로 설정
    JSON_DIR = "New_sample/labeling"
    IMAGE_DIR = "New_sample/source"

    print("데이터 파싱 중...")
    manager = PlantDataManager(base_json_dir=JSON_DIR, base_img_dir=IMAGE_DIR)
    df = manager.parse_all_image()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PlantDataset(
        manager_df=df, base_img_dir=IMAGE_DIR, target_size=(224, 224), transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False
    )

    # 클래스 개수 설정 (꼼수 방지를 위해 최솟값 2로 안전장치)
    num_plants = max(df['plant_label'].nunique(), 2)
    num_parts = max(df['part_label'].nunique(), 2)
    print(f"설정된 식물 종류 수: {num_plants} | 식물 부위 수: {num_parts}")

    # 우리가 커스텀한 멀티 타겟 모델 생성
    model = MultiTaskPlantModel(base_model_name='resnet50', num_plants=num_plants, num_parts=num_parts)
    model = model.to(device)

    # 채점관 및 옵티마이저 (두 출구를 모두 학습하므로 model.parameters() 그대로 사용)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    EPOCHS = 10
    print("\n멀티 타겟 학습을 시작합니다!")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # 통계를 따로 계산하기 위해 변수 세팅
        plant_correct, part_correct, total = 0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

        for images, labels in progress_bar:
            images = images.to(device)

            # 실제 정답지 2개 분리해서 가져오기
            target_plants = labels['plant_label'].to(device)
            target_parts = labels['part_label'].to(device)

            # 모델에게 예측시키기 (출력이 2개 나옴)
            pred_plants, pred_parts = model(images)

            # 각각 채점하기
            loss_plant = criterion(pred_plants, target_plants)
            loss_part = criterion(pred_parts, target_parts)

            # 최종 벌점(Loss)은 두 벌점의 합입니다.
            total_loss = loss_plant + loss_part

            # 역전파 학습
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 정확도 통계 계산
            running_loss += total_loss.item() * images.size(0)
            _, predicted_plant = pred_plants.max(1)
            _, predicted_part = pred_parts.max(1)

            total += target_plants.size(0)
            plant_correct += predicted_plant.eq(target_plants).sum().item()
            part_correct += predicted_part.eq(target_parts).sum().item()

            # 진행률 표시줄에 두 종류의 정확도 모두 출력
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'P_Acc': f"{100.0 * plant_correct / total:.1f}%",
                'Part_Acc': f"{100.0 * part_correct / total:.1f}%"
            })

        epoch_loss = running_loss / len(train_loader.dataset)
        print(
            f"➔ Epoch [{epoch + 1}/{EPOCHS}] 완료 - Loss: {epoch_loss:.4f} | 식물정확도: {100.0 * plant_correct / total:.2f}% | 부위정확도: {100.0 * part_correct / total:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), "multi_task_plant_model.pth")
    print("멀티 타겟 모델 저장 완료: multi_task_plant_model.pth")


if __name__ == "__main__":
    main()