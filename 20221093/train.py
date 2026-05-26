import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import timm

import os
import cv2

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

    # 1. 경로 설정 및 데이터 파싱 (기존과 동일)
    JSON_DIR = "C:/Users/PC/Downloads/20221093/New_sample/라벨링테이터"
    IMAGE_DIR = "C:/Users/PC/Downloads/20221093/New_sample/원천데이터"

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

    # -------------------------------------------------------------
    # 2. [위치 변경] 클래스 개수 설정 및 모델 생성을 먼저 합니다!
    # -------------------------------------------------------------
    num_plants = int(df['plant_label'].max() + 1)  # 에러 방지용 .max() + 1
    num_parts = int(df['part_label'].max() + 1)   # 에러 방지용 .max() + 1
    print(f"설정된 식물 종류 수: {num_plants} | 식물 부위 수: {num_parts}")

    # 모델을 만들고 바로 GPU(.to(device))로 보냅니다.
    model = MultiTaskPlantModel(base_model_name='resnet50', num_plants=num_plants, num_parts=num_parts)
    model = model.to(device)

    # -------------------------------------------------------------
    # 3. [위치 변경] 그 다음, 안전하게 DataLoader를 생성합니다.
    # -------------------------------------------------------------
    # 이제 GPU가 활성화된 상태이므로 pin_memory=True 경고가 뜨지 않습니다.
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
    )

    # 4. 채점관 및 옵티마이저 설정 (기존과 동일)
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

    print("\n[테스트 진행] 학습된 모델과 OpenCV를 이용해 개별 이미지 추론을 시작합니다...")

    # 1. 테스트할 이미지 경로 지정
    TEST_IMAGE_PATH = "New_sample/test/test_image.jpg"

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 에러: 테스트할 이미지 파일({TEST_IMAGE_PATH})이 없습니다. 경로를 확인해주세요.")
    else:
        # 2. 모델을 평가(추론) 모드로 전환
        model.eval()

        # 3. OpenCV로 이미지 로드 및 RGB 변환
        # cv2.imread는 이미지를 BGR 형태로 읽으므로, 학습 시 사용한 RGB 형태로 변환 필수!
        bgr_img = cv2.imread(TEST_IMAGE_PATH)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # 4. 이미지 크기 조절 (224, 224)
        resized_img = cv2.resize(rgb_img, (224, 224))

        # 5. 전처리 (텐서 변환 및 정규화)
        # torchvision.transforms.ToTensor()는 PIL 이미지뿐만 아니라
        # (H, W, C) 형태의 NumPy 배열(uint8)도 자동으로 [0, 1] 범위의 텐서로 변환해줍니다.
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = test_transform(resized_img)
        input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가 (1, 3, 224, 224)
        input_tensor = input_tensor.to(device)

        # 6. 예측 수행
        with torch.no_grad():
            pred_plants, pred_parts = model(input_tensor)

            # 확률값으로 변환
            prob_plants = torch.softmax(pred_plants, dim=1)
            prob_parts = torch.softmax(pred_parts, dim=1)

            val_plant, idx_plant = prob_plants.max(1)
            val_part, idx_part = prob_parts.max(1)

        # 7. 결과 출력
        print("\n" + "=" * 45)
        print(f"📷 테스트 이미지: {TEST_IMAGE_PATH} (OpenCV Load)")
        print("=" * 45)

        try:
            plant_name = manager.plant_encoder.classes_[idx_plant.item()]
            part_name = manager.part_encoder.classes_[idx_part.item()]
            print(f"🌿 예측 식물 종류: {plant_name} ({val_plant.item() * 100:.2f}%)")
            print(f"🍂 예측 식물 부위: {part_name} ({val_part.item() * 100:.2f}%)")
        except AttributeError:
            print(f"🌿 예측 식물 인덱스: {idx_plant.item()} ({val_plant.item() * 100:.2f}%)")
            print(f"🍂 예측 식물 부위 인덱스: {idx_part.item()} ({val_part.item() * 100:.2f}%)")

        print("=" * 45)


if __name__ == "__main__":
    main()

