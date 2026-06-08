import sys
import os
import json
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

# 우리가 만든 파일과 클래스 불러오기
from Utils.data_manager import PlantDataManager
from Utils.dataset import PlantDataset


# =====================================================================
# 1. 순수 torchvision.models.resnet50 기반 멀티태스크 모델 정의
# =====================================================================
class MultiTaskPlantModel(nn.Module):
    def __init__(self, num_plants=2, num_parts=2):
        super(MultiTaskPlantModel, self).__init__()
        # PyTorch 공식 표준 ResNet50 로드 (weights 옵션 적용)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 원래 레스넷50의 맨 마지막 출력층(fc) 입력 특징 벡터 크기 파악 (2048)
        num_features = self.backbone.fc.in_features

        # 원래의 단일 분류 fc 레이어는 무력화(Identity) 시킴
        self.backbone.fc = nn.Identity()

        # 뇌세포 끝에 우리가 원하는 멀티 타겟 출구 2개 새로 이어 붙이기
        self.fc_plant = nn.Linear(num_features, num_plants)  # 식물 종류 출구
        self.fc_part = nn.Linear(num_features, num_parts)    # 식물 부위 출구

    def forward(self, x):
        # 이미지를 보고 특징 추출
        features = self.backbone(x)

        # 추출된 공통 특징(feature)을 바탕으로 두 가지 정답을 각각 예측
        plant_out = self.fc_plant(features)
        part_out = self.fc_part(features)

        return plant_out, part_out


# =====================================================================
# 3. 메인 학습 및 검증 파이프라인
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 사용 중인 장치: {device}")

    # 경로 설정 및 데이터 파싱
    JSON_DIR = "D:/CDproject/data/라벨링데이터"
    IMAGE_DIR = "D:/CDproject/data/원천데이터"

    print("데이터 파싱 중...")
    manager = PlantDataManager(base_json_dir=JSON_DIR, base_img_dir=IMAGE_DIR)
    df = manager.parse_all_image()

    print(f"파싱된 데이터 개수: {len(df)}개")
    if len(df) == 0:
        print(f"❌ [경고] 데이터가 0개 파싱되었습니다! 경로를 다시 확인해주세요.")
        return

    # 🔥 [필수 보존] 징검다리 라벨을 0, 1, 2, 3 순차적 인덱스로 강제 클리닝
    df['plant_label_cat'] = df['plant_label'].astype('category')
    df['plant_label_clean'] = df['plant_label_cat'].cat.codes

    df['part_label_cat'] = df['part_label'].astype('category')
    df['part_label_clean'] = df['part_label_cat'].cat.codes

    # 🔥 [필수 보존] 추후 GUI에서 숫자를 한글 이름으로 번역하기 위한 매핑 딕셔너리 추출
    plant_mapping = dict(enumerate(df['plant_label_cat'].cat.categories))
    part_mapping = dict(enumerate(df['part_label_cat'].cat.categories))
    plant_name_mapping = dict(zip(df['plant_label_clean'], df['plant_name']))
    part_name_list = ["leaf", "stem", "flower", "root", "fruit", "group"]

    # 데이터 분리 (8:2)
    total_size = len(df)
    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)

    train_size = int(0.8 * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    # 데이터 증강 및 전처리 정의
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30, fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PlantDataset(manager_df=train_df, base_img_dir=IMAGE_DIR, transform=train_transform)
    val_dataset = PlantDataset(manager_df=val_df, base_img_dir=IMAGE_DIR, transform=val_transform)

    num_plants = int(df['plant_label_clean'].nunique())
    num_parts = int(df['part_label_clean'].nunique())
    print(f"설정된 식물 종류 수: {num_plants} | 식물 부위 수: {num_parts}")
    print(f"학습용: {len(train_dataset)}장 | 검증용: {len(val_dataset)}장")

    # 개조된 torchvision 기반의 모델 인스턴스 생성
    model = MultiTaskPlantModel(num_plants=num_plants, num_parts=num_parts)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    EPOCHS = 10
    print("\n[Pure torchvision ResNet50] 멀티 타겟 학습을 시작합니다!")

    for epoch in range(EPOCHS):
        # ------------------ TRAIN ------------------
        model.train()
        running_loss = 0.0
        plant_correct, part_correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Train")

        for images, labels in progress_bar:
            images = images.to(device)
            target_plants = labels[0].to(device)
            target_parts = labels[1].to(device)

            pred_plants, pred_parts = model(images)

            loss_plant = criterion(pred_plants, target_plants)
            loss_part = criterion(pred_parts, target_parts)
            total_loss = loss_plant + loss_part

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)
            _, predicted_plant = pred_plants.max(1)
            _, predicted_part = pred_parts.max(1)

            total += target_plants.size(0)
            plant_correct += predicted_plant.eq(target_plants).sum().item()
            part_correct += predicted_part.eq(target_parts).sum().item()

            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'P_Acc': f"{100.0 * plant_correct / total:.1f}%",
                'Part_Acc': f"{100.0 * part_correct / total:.1f}%"
            })

        epoch_loss = running_loss / len(train_loader.dataset)
        train_p_acc = 100.0 * plant_correct / total
        train_part_acc = 100.0 * part_correct / total

        # ------------------ VALIDATION ------------------
        model.eval()
        val_loss = 0.0
        v_plant_correct, v_part_correct, v_total = 0, 0, 0

        with torch.no_grad():
            for v_images, v_labels in val_loader:
                v_images = v_images.to(device)
                v_target_plants = v_labels[0].to(device)
                v_target_parts = v_labels[1].to(device)

                v_pred_plants, v_pred_parts = model(v_images)

                v_loss_plant = criterion(v_pred_plants, v_target_plants)
                v_loss_part = criterion(v_pred_parts, v_target_parts)
                v_total_loss = v_loss_plant + v_loss_part

                val_loss += v_total_loss.item() * v_images.size(0)
                _, v_predicted_plant = v_pred_plants.max(1)
                _, v_predicted_part = v_pred_parts.max(1)

                v_total += v_target_plants.size(0)
                v_plant_correct += v_predicted_plant.eq(v_target_plants).sum().item()
                v_part_correct += v_predicted_part.eq(v_target_parts).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_p_acc = 100.0 * v_plant_correct / v_total
        val_part_acc = 100.0 * v_part_correct / v_total

        scheduler.step()

        print(f"\n📢 Epoch [{epoch + 1}/{EPOCHS}] 완료")
        print(f"   [Train] Loss: {epoch_loss:.4f} | 식물정확도: {train_p_acc:.2f}% | 부위정확도: {train_part_acc:.2f}%")
        print(f"   [Val]   Loss: {epoch_val_loss:.4f} | 식물정확도: {val_p_acc:.2f}% | 부위정확도: {val_part_acc:.2f}%")
        print("-" * 70)

    # 모델 가중치(State Dict) 파일 저장
    torch.save(model.state_dict(), "multi_task_plant_model.pth")
    print("멀티 타겟 모델 저장 완료: multi_task_plant_model.pth")

    # 🔥 [필수 보존] 나중에 GUI 메인 프로그램이 읽을 가이드북 JSON 생성
    mapping_data = {
        "plant_mapping": {int(k): int(v) for k, v in plant_mapping.items()},
        "plant_name_mapping": {int(k): str(v) for k, v in plant_name_mapping.items()},
        "part_mapping": {int(k): int(v) for k, v in part_mapping.items()},
        "part_name_mapping": {int(k): str(v) for k, v in enumerate(part_name_list)}
    }

    with open("label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=4)
    print("매핑 가이드북 저장 완료: label_mapping.json")


# =====================================================================
# 4. OPENCV 개별 이미지 추론 테스트 (매핑 정보 로드 버전)
# =====================================================================
# 추후 구현 필요 시 여기에 코드를 추가하세요.


if __name__ == "__main__":
    main()