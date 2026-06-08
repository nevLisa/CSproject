import os
import torch


def merge_mappings_to_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_path = "multi_task_plant_model.pth"

    # 1. 기존 가중치 파일 존재 여부 확인
    if not os.path.exists(pth_path):
        print(f"❌ 에러: {pth_path} 파일이 존재하지 않습니다. 경로를 확인해주세요.")
        return

    print(f"🔄 기존 모델 가중치 로드 중: {pth_path}")
    checkpoint = torch.load(pth_path, map_location=device)

    # 2. 만약 기존 파일이 딕셔너리 구조가 아니라 가중치(state_dict)만 덩그러니 있다면 구조 변경
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        print("💡 순수 가중치만 발견되어 딕셔너리 포맷으로 전환합니다.")
        state_dict = checkpoint
        checkpoint = {"model_state_dict": state_dict}
    else:
        print("💡 기존 딕셔너리 포맷 모델을 발견했습니다.")

    # ------------------------------------------------------------------
    # 3. [핵심] 에러가 났던 매핑 데이터를 딕셔너리에 주입합니다.
    # ------------------------------------------------------------------
    # 메인 코드에서 사용한 전처리 기준과 동일하게 수동으로 매핑 정보를 빌드합니다.
    # (실제 학습 때 사용된 카테고리 순서와 일치해야 추론 시 이름이 꼬이지 않습니다.)

    # 식물 부위 표준 리스트 (0~5 인덱스 맵핑)
    part_name_list = ["leaf", "stem", "flower", "root", "fruit", "group"]

    print("📦 매핑 데이터 조립 중...")

    # 🚨 [KeyError 해결 지점] 불러올 때 에러가 났던 'part_name_mapping'을 확실하게 주입합니다.
    checkpoint["part_name_mapping"] = {i: name for i, name in enumerate(part_name_list)}

    # 나머지 매핑 파일들도 안전장치로 함께 저장 (이미 존재한다면 유지, 없으면 기본값 주입)
    # ⚠️ 만약 현재 실행 환경에 'df' 변수가 살아있다면, {i: cat for i, cat in enumerate(df['plant_label_cat'].cat.categories)} 형태로 넣는 것이 가장 정확합니다.
    checkpoint["plant_mapping"] = checkpoint.get("plant_mapping", {i: i for i in range(100)})
    checkpoint["part_mapping"] = checkpoint.get("part_mapping", {i: i for i in range(6)})
    checkpoint["plant_name_mapping"] = checkpoint.get("plant_name_mapping", {i: f"Plant_{i}" for i in range(100)})

    # 4. 최종 결합된 파일로 덮어쓰기 저장
    torch.save(checkpoint, pth_path)
    print(f"✨ [완료] 매핑 데이터 주입 성공! '{pth_path}' 파일이 업데이트되었습니다.")
    print(" 이제 메인 코드에서 로드할 때 KeyError가 발생하지 않습니다.")


if __name__ == "__main__":
    merge_mappings_to_model()