import torch
import torch.nn as nn
import cv2
import timm
import json
from torchvision import transforms
from PIL import Image


# 1. 모델 구조 정의 (학습 때와 100% 동일)
class MultiTaskPlantModel(nn.Module):
    def __init__(self, base_model_name='resnet50', num_plants=2, num_parts=2):
        super(MultiTaskPlantModel, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=False, num_classes=0)
        num_features = self.base_model.num_features
        self.plant_classifier = nn.Linear(num_features, num_plants)
        self.part_classifier = nn.Linear(num_features, num_parts)

    def forward(self, x):
        features = self.base_model(x)
        return self.plant_classifier(features), self.part_classifier(features)


if __name__ == "__main__":
    # 📌 [경로 세팅] 내 환경에 맞게 이 3개만 확인하세요!
    MODEL_WEIGHT_PATH = "multi_task_plant_model.pth"
    MAPPING_JSON_PATH = "label_mapping.json"
    TEST_IMAGE_PATH = "D:/CDproject/data/test/test.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 매핑 가이드북(JSON) 로드 및 Key 숫자 변환
    with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    loaded_plant_mapping = {int(k): v for k, v in mappings["plant_mapping"].items()}
    loaded_plant_name_mapping = {int(k): v for k, v in mappings["plant_name_mapping"].items()}
    loaded_part_mapping = {int(k): v for k, v in mappings["part_mapping"].items()}
    loaded_part_name_mapping = {int(k): v for k, v in mappings["part_name_mapping"].items()}

    # JSON 안에 들어있는 고유 개수대로 방 크기 자동 설정
    num_plants = len(loaded_plant_mapping)
    num_parts = len(loaded_part_mapping)

    # 3. 모델 빌드 및 가중치 주입
    model = MultiTaskPlantModel(base_model_name='resnet50', num_plants=num_plants, num_parts=num_parts)

    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    # 💡 과거 꼬인 파일 포맷 유무 상관없이 알맹이만 무조건 통과시키는 방어 코드
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint,
                                                              dict) and "model_state_dict" in checkpoint else checkpoint

    model.load_state_dict(state_dict)
    model.to(device).eval()

    # 4. 이미지 로드 및 전처리
    bgr_img = cv2.imread(TEST_IMAGE_PATH)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = test_transform(pil_img).unsqueeze(0).to(device)

    # 5. 모델 추론
    with torch.no_grad():
        pred_plants, pred_parts = model(input_tensor)
        prob_plants = torch.softmax(pred_plants, dim=1)
        prob_parts = torch.softmax(pred_parts, dim=1)
        val_plant, idx_plant = prob_plants.max(1)
        val_part, idx_part = prob_parts.max(1)

    # 6. 결과 매핑 및 텍스트 최종 출력
    target_idx = idx_plant.item()
    target_part_idx = idx_part.item()

    orig_plant_code = loaded_plant_mapping.get(target_idx, "알 수 없음")
    orig_plant_name = loaded_plant_name_mapping.get(target_idx, "알 수 없음")
    orig_part_code = loaded_part_mapping.get(target_part_idx, "알 수 없음")
    orig_part_name = loaded_part_name_mapping.get(target_part_idx, "알 수 없음")

    print("\n" + "=" * 60)
    print(f"🌿 식물 예측: {orig_plant_name} (원래 코드: {orig_plant_code}) [{val_plant.item() * 100:.2f}%]")
    print(f"🍂 부위 예측: {orig_part_name} (원래 코드: {orig_part_code}) [{val_part.item() * 100:.2f}%]")
    print("=" * 60)