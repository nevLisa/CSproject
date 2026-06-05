import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
import json

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

    # 🚨 경로 오타 등으로 데이터가 안 읽혔을 때를 위한 방어 코드
    print(f"파싱된 데이터 개수: {len(df)}개")
    if len(df) == 0:
        print(f"❌ [경고] 데이터가 0개 파싱되었습니다! 경로를 다시 확인해주세요.")
        print(f"현재 설정된 경로: {JSON_DIR}")
        return

    # ------------------------------------------------------------------
    # 🚨 [핵심 해결책] 징검다리 라벨([23, 24, 27, 28])을 [0, 1, 2, 3]으로 강제 변환
    # ------------------------------------------------------------------
    df['plant_label_cat'] = df['plant_label'].astype('category')
    df['plant_label_clean'] = df['plant_label_cat'].cat.codes

    df['part_label_cat'] = df['part_label'].astype('category')
    df['part_label_clean'] = df['part_label_cat'].cat.codes

    # 나중에 OpenCV 추론할 때 숫자(0,1,2,3)를 원래 번호나 이름으로 되돌리기 위한 지도 보관
    plant_mapping = dict(enumerate(df['plant_label_cat'].cat.categories))
    part_mapping = dict(enumerate(df['part_label_cat'].cat.categories))

    # 텍스트 식물 이름도 복원용으로 매핑 보관
    plant_name_mapping = dict(zip(df['plant_label_clean'], df['plant_name']))
    part_name_list = ["leaf", "stem", "flower", "root", "fruit", "group"]  # PART_MAP 순서 기준

    # -------------------------------------------------------------
    # 🛠️ 분리 알고리즘: Train / Validation 8:2 안전 분리
    # -------------------------------------------------------------
    total_size = len(df)
    indices = list(range(total_size))

    import random
    random.seed(42)  # 데이터 셔플 고정 (재현성)
    random.shuffle(indices)

    train_size = int(0.8 * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    # -------------------------------------------------------------
    # 🛠️ 이미지 전처리 및 증강(Augmentation) 설정
    # -------------------------------------------------------------
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

    # 독립된 데이터셋 생성
    train_dataset = PlantDataset(manager_df=train_df, base_img_dir=IMAGE_DIR, transform=train_transform)
    val_dataset = PlantDataset(manager_df=val_df, base_img_dir=IMAGE_DIR, transform=val_transform)

    # 🚨 이제 중구난방 번호가 아닌 고유값 개수(nunique)로 정확히 노드 방을 만듭니다.
    num_plants = int(df['plant_label_clean'].nunique())
    num_parts = int(df['part_label_clean'].nunique())
    print(f"설정된 식물 종류 수: {num_plants} | 식물 부위 수: {num_parts}")
    print(f"학습용: {len(train_dataset)}장 | 검증용: {len(val_dataset)}장")

    model = MultiTaskPlantModel(base_model_name='resnet50', num_plants=num_plants, num_parts=num_parts)
    model = model.to(device)

    # Windows 멀티프로세싱 크래시 방지용 num_workers=0 고정
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    # 💡 5080의 강력한 연산 시 가중치 폭발 방지를 위해 안정적인 lr=1e-5로 세팅
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    EPOCHS = 10
    print("\n멀티 타겟 학습 및 검증을 시작합니다!")

    for epoch in range(EPOCHS):
        # ==================== [TRAIN STEP] ====================
        model.train()
        running_loss = 0.0
        plant_correct, part_correct, total = 0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Train")

        for images, labels in progress_bar:
            images = images.to(device)

            # 🚨 딕셔너리가 아닌 튜플 형태의 인덱스([0], [1])로 안전하게 꺼냅니다.
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

        # ==================== [VALIDATION STEP] ====================
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

    # 모델 가중치 저장
    torch.save(model.state_dict(), "multi_task_plant_model.pth")
    print("멀티 타겟 모델 저장 완료: multi_task_plant_model.pth")

    part_name_list = ["leaf", "stem", "flower", "root", "fruit", "group"]

    # 모델 예측 인덱스(0, 1, 2...)와 매칭할 딕셔너리로 변환


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
    print("\n[테스트 진행] 저장된 매핑 정보와 OpenCV를 이용해 개별 이미지 추론을 시작합니다...")
    TEST_IMAGE_PATH = "D:/CDproject/data/test/test.jpg"
    MAPPING_JSON_PATH = "label_mapping.json"  # 💡 저장해둔 가이드북 파일 경로

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 에러: 테스트할 이미지 파일({TEST_IMAGE_PATH})이 없습니다. 경로를 확인해주세요.")
    elif not os.path.exists(MAPPING_JSON_PATH):
        print(f"❌ 에러: 매핑 정보 파일({MAPPING_JSON_PATH})이 없습니다. 학습 단에서 먼저 저장해야 합니다.")
    else:
        # -----------------------------------------------------------------
        # [안전 조치 1] JSON 파일로부터 매핑 가이드북 불러오기
        # -----------------------------------------------------------------

        with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f:
            mappings = json.load(f)

        # 🚨 중요: JSON은 저장될 때 딕셔너리의 Key가 문자열("0", "1")로 강제 변환되므로,
        # 모델의 아웃풋인 정수(int)와 매칭하기 위해 다시 숫자로 변환해 줍니다.
        loaded_plant_mapping = {int(k): v for k, v in mappings["plant_mapping"].items()}
        loaded_plant_name_mapping = {int(k): v for k, v in mappings["plant_name_mapping"].items()}
        loaded_part_mapping = {int(k): v for k, v in mappings["part_mapping"].items()}
        loaded_part_name_mapping = {int(k): v for k, v in mappings["part_name_mapping"].items()}

        # -----------------------------------------------------------------
        # [안전 조치 2] 이미지 로드 및 전처리
        # -----------------------------------------------------------------
        model.eval()
        bgr_img = cv2.imread(TEST_IMAGE_PATH)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # PyTorch의 transforms.ToTensor()를 가장 안전하게 태우기 위해 PIL Image로 변환
        from PIL import Image
        pil_img = Image.fromarray(rgb_img)

        # 학습 때의 val_transform과 완벽히 동일하게 세팅 (크기 조정 포함)
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = test_transform(pil_img)
        input_tensor = input_tensor.unsqueeze(0)  # 차원 추가 (1, 3, 224, 224)
        input_tensor = input_tensor.to(device)

        # -----------------------------------------------------------------
        # [안전 조치 3] 모델 추론 수행
        # -----------------------------------------------------------------
        with torch.no_grad():
            pred_plants, pred_parts = model(input_tensor)
            prob_plants = torch.softmax(pred_plants, dim=1)
            prob_parts = torch.softmax(pred_parts, dim=1)

            val_plant, idx_plant = prob_plants.max(1)
            val_part, idx_part = prob_parts.max(1)

        print("\n" + "=" * 50)
        print(f"📷 테스트 이미지: {TEST_IMAGE_PATH} (OpenCV + JSON Mapping Load)")
        print("=" * 50)

        # -----------------------------------------------------------------
        # [안전 조치 4] 불러온 가이드북 정보를 바탕으로 진짜 결과 복원
        # -----------------------------------------------------------------
        try:
            target_idx = idx_plant.item()  # 모델이 뱉은 식물 압축 인덱스 (예: 1)
            target_part_idx = idx_part.item()  # 모델이 뱉은 부위 압축 인덱스 (예: 0)

            # 1. 식물 정보 역추적
            orig_plant_code = loaded_plant_mapping[target_idx]
            orig_plant_name = loaded_plant_name_mapping.get(target_idx, "Unknown")

            # 2. 부위 정보 역추적 (리스트 인덱스 에러 방지를 위해 딕셔너리 맵 구조 사용)
            orig_part_code = loaded_part_mapping[target_part_idx]
            orig_part_name = loaded_part_name_mapping.get(target_part_idx, "Unknown")

            print(f"🌿 예측 식물 종류: {orig_plant_name} (원래 코드: {orig_plant_code}) ({val_plant.item() * 100:.2f}%)")
            print(f"🍂 예측 식물 부위: {orig_part_name} (원래 코드: {orig_part_code}) ({val_part.item() * 100:.2f}%)")

        except Exception as e:
            print(f"❌ 복원 실패 (압축 번호 출력) - 에러 내용: {e}")
            print(f"🌿 예측 식물 압축 인덱스: {idx_plant.item()} ({val_plant.item() * 100:.2f}%)")
            print(f"🍂 예측 식물 부위 압축 인덱스: {idx_part.item()} ({val_part.item() * 100:.2f}%)")

        print("=" * 50)


if __name__ == "__main__":
    main()