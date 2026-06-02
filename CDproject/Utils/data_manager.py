# 이미지와 레이블을 불러와서 이미지 , 식물종류 , 부분 만드는 코드
# 이미지와 레이블을 불러와서 이미지, 식물종류, 부위 데이터프레임을 만드는 코드
import os
import json
import pandas as pd

# [수정] 실제 AIHub 데이터셋의 한글 부위 이름에 맞게 매핑 사전 생성
PART_MAP = {
    "잎": 0,
    "줄기": 1,
    "꽃": 2,
    "뿌리": 3,
    "열매": 4,
    "전초": 5
}


class PlantDataManager:
    def __init__(self, base_json_dir, base_img_dir):
        self.base_json_dir = base_json_dir
        self.base_img_dir = base_img_dir
        self.df = pd.DataFrame()

        # 식물 한글 이름을 고유한 숫자 레이블(ID)로 변환하기 위한 딕셔너리
        self.plant_name_to_label = {}
        self.label_to_plant_name = {}

    def parse_all_image(self):
        dataset_records = []  # 데이터 임시 저장용
        plant_set = set()  # 고유 식물 이름을 모으기 위한 집합

        # 에러 검출: JSON 폴더가 존재하지 않는 경우
        if not os.path.exists(self.base_json_dir):
            print(f"[경고] JSON 디렉터리가 존재하지 않습니다: {self.base_json_dir}")
            return pd.DataFrame()

        print("JSON 파일 스캔 및 파싱 중...")

        # os.walk를 사용하여 하위 폴더까지 전부 탐색하며 JSON 파일 읽기
        for root, dirs, files in os.walk(self.base_json_dir):
            for file_name in files:
                if file_name.endswith(".json"):
                    json_file_path = os.path.join(root, file_name)

                    try:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # 1. 실제 JSON 서식에 맞춘 데이터 추출
                        img_name = data["image"]["file_name"]
                        plant_name = data["plantinfo"]["class_name"]
                        part_str = data["plantinfo"]["instance_info"]["instance_name"]

                        # 2. 부위 한글명을 정수 레이블로 변환 (없으면 -1)
                        part_label = PART_MAP.get(part_str, -1)

                        # 고유 식물 이름 집합에 추가 (나중에 숫자 ID 발급용)
                        plant_set.add(plant_name)

                        dataset_records.append({
                            "image": img_name,
                            "plant_name": plant_name,
                            "part_str": part_str,
                            "part_label": part_label,
                            "file_name": file_name  # 디버깅용 용도
                        })

                    except KeyError as ke:
                        # 갱신된 구조와 맞지 않는 레거시 파일이 섞여있을 때의 방어 코드
                        print(f"[경고] 키 누락으로 인한 패스 ({file_name}): {ke}")
                        continue
                    except Exception as e:
                        print(f"[오류] 파일 처리 실패 ({file_name}): {e}")
                        continue

        if not dataset_records:
            print("[경고] 파싱된 데이터가 하나도 없습니다. 경로를 재확인하세요.")
            return pd.DataFrame()

        # 3. 발견된 식물 이름들에 고유한 정수 ID(plant_label) 부여하기
        sorted_plants = sorted(list(plant_set))
        for idx, p_name in enumerate(sorted_plants):
            self.plant_name_to_label[p_name] = idx
            self.label_to_plant_name[idx] = p_name

        # 4. 생성된 레코드에 최종 식물 ID(plant_label) 주입
        for record in dataset_records:
            record["plant_label"] = self.plant_name_to_label[record["plant_name"]]

        # 데이터프레임 변환 및 저장
        self.df = pd.DataFrame(dataset_records)
        print(f"데이터 파싱 완료! 총 {len(self.df)}개의 이미지가 등록되었습니다.")
        print(f"발견된 식물 종류: {self.plant_name_to_label}")

        return self.df