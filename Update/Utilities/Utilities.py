import os
import json
import torch


class PlantLabelDecoder:
    """
    모델의 출력 인덱스(0, 1, 2...)를 원래의 식물 코드 및 텍스트 이름으로
    안전하게 복원(Decoding)해주는 전용 클래스입니다.
    """

    def __init__(self, json_path="label_mapping.json"):
        self.json_path = json_path
        self.plant_mapping = {}
        self.plant_name_mapping = {}
        self.part_mapping = {}
        self.part_name_mapping = {}

        # 클래스 생성 시 자동으로 JSON 가이드북 로드
        self._load_mapping_json()

    def _load_mapping_json(self):
        """JSON 파일을 읽어와서 파이썬 정수(int) Key 구조로 정렬 및 변환합니다."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"❌ [에러] 지정된 매핑 파일이 없습니다: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)

        try:
            # 🚨 [패치 1] 인덱스 역전 현상을 막기 위해 Key(숫자) 기준으로 오름차순 강제 정렬 후 변환합니다.
            self.plant_mapping = {int(k): v for k, v in
                                  sorted(mappings["plant_mapping"].items(), key=lambda x: int(x[0]))}
            self.plant_name_mapping = {int(k): v for k, v in
                                       sorted(mappings["plant_name_mapping"].items(), key=lambda x: int(x[0]))}
            self.part_mapping = {int(k): v for k, v in
                                 sorted(mappings["part_mapping"].items(), key=lambda x: int(x[0]))}
            self.part_name_mapping = {int(k): v for k, v in
                                      sorted(mappings["part_name_mapping"].items(), key=lambda x: int(x[0]))}

            print(f"✅ 매핑 가이드북 로드 완료! (식물 종류: {len(self.plant_mapping)}개, 부위: {len(self.part_mapping)}개)")

        except KeyError as ke:
            raise KeyError(f"❌ [에러] JSON 파일 구조가 올바르지 않습니다. 필수 키 누락: {ke}")

    def decode(self, idx_plant, idx_part):
        """
        모델이 뱉은 인덱스를 받아 원래 코드와 이름으로 복원하여 딕셔너리로 반환합니다.
        """
        # 🚨 [패치 2] 입력값이 GPU에 있는 PyTorch Tensor일 경우를 대비해
        # 안전하게 .detach().cpu() 처리를 먼저 거친 후 순수 파이썬 숫자로 뽑아냅니다.
        if isinstance(idx_plant, torch.Tensor):
            target_idx = idx_plant.detach().cpu().item()
        else:
            target_idx = int(idx_plant)

        if isinstance(idx_part, torch.Tensor):
            target_part_idx = idx_part.detach().cpu().item()
        else:
            target_part_idx = int(idx_part)

        try:
            # 🚨 [패치 3] 사전에 없는 인덱스가 오면 "Unknown" 대신 에러를 발생시켜
            # 아래의 except 블록에서 예외 정보와 함께 안전하게 캐치되도록 유도합니다.
            if target_idx not in self.plant_mapping or target_part_idx not in self.part_mapping:
                raise IndexError(f"예측 인덱스(Plant:{target_idx}, Part:{target_part_idx})가 매핑 데이터 범위를 벗어났습니다.")

            # 안전하게 매핑 값 추출
            orig_plant_code = self.plant_mapping[target_idx]
            orig_plant_name = self.plant_name_mapping[target_idx]
            orig_part_code = self.part_mapping[target_part_idx]
            orig_part_name = self.part_name_mapping[target_part_idx]

            return {
                "success": True,
                "plant_code": orig_plant_code,
                "plant_name": orig_plant_name,
                "part_code": orig_part_code,
                "part_name": orig_part_name
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "plant_idx": target_idx,
                "part_idx": target_part_idx
            }

        """
        사용 예시 
        
        # 1. 추론 시작 전, 디코더 클래스 딱 한 번만 선언해두기
        decoder = PlantLabelDecoder(json_path="label_mapping.json")

        # ... (OpenCV 이미지 로드, 모델 추론 부분 생략) ...

        with torch.no_grad():
            pred_plants, pred_parts = model(input_tensor)
            prob_plants = torch.softmax(pred_plants, dim=1)
            prob_parts = torch.softmax(pred_parts, dim=1)

            val_plant, idx_plant = prob_plants.max(1)
            val_part, idx_part = prob_parts.max(1)

        # 2. 클래스 메서드를 사용해 한 줄로 착착착 복원하기 🔥
        result = decoder.decode(idx_plant, idx_part)

        print("\n" + "=" * 50)
        if result["success"]:
            print(f"🌿 예측 식물 종류: {result['plant_name']} (원래 코드: {result['plant_code']}) ({val_plant.item() * 100:.2f}%)")
            print(f"🍂 예측 식물 부위: {result['part_name']} (원래 코드: {result['part_code']}) ({val_part.item() * 100:.2f}%)")
        else:
            print(f"❌ 복원 실패: {result['error']}")
            print(f"🌿 예측 식물 압축 인덱스: {result['plant_idx']} ({val_plant.item() * 100:.2f}%)")
            print(f"🍂 예측 식물 부위 압축 인덱스: {result['part_idx']} ({val_part.item() * 100:.2f}%)")
        print("=" * 50)
        
        """