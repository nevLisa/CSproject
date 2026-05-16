# 이미지와 레이블을 불러와서 이미지 , 식물종류 , 부분 만드는 코드

import os
import json
import pandas as pd

# 식물 부위
PART_MAP = {"leaf": 0, "stem": 1, "flower": 2, "root": 3, "fruit": 4, "group": 5}


class PlantDataManager:
    def __init__(self, base_json_dir, base_img_dir):
        self.base_json_dir = base_json_dir
        self.base_img_dir = base_img_dir
        self.df = pd.DataFrame()

    # 하위 이미지 폴더 분류 폴더 이름들이 folder_map 에 저장됨
    def make_folder_map(self):
        folder_map = {}
        if not os.path.exists(self.base_img_dir):
            return folder_map

        for folder_name in os.listdir(self.base_img_dir):
            folder_path = os.path.join(self.base_img_dir, folder_name)
            if os.path.isdir(folder_path) and "_" in folder_name:
                code = folder_name.split("_")[0]
                folder_map[code] = folder_name
        return folder_map
    #이미지이름 과 라벨을 합쳐 데이터 프레임을 만드는 코드
    def parse_all_image(self):
        dataset_records = [] #데이터 임시 저장용
        folder_map = self.make_folder_map()

        #에러 검출
        if not os.path.exists(self.base_json_dir):
            return pd.DataFrame()
        #레이블 읽어 오고 이미지 이름 불러와서 데이터 프레임에 삽입 하는 코드
        for file_name in os.listdir(self.base_json_dir):
            if file_name.endswith(".json"):
                json_file_path = os.path.join(self.base_json_dir, file_name)

                try:
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    img_name = data["imagedata"]["filename"]
                    kind_code = data["metadata"]["kind"]
                    part_str = data["metadata"]["part"]


                    if kind_code in folder_map:
                        matched_folder = folder_map[kind_code]

                        # 부분을 정수로 변환
                        part_label = PART_MAP.get(part_str, -1)

                        dataset_records.append({
                            "image": img_name,
                            "plant_label": int(kind_code),  # 식물 종류
                            "part_label": part_label,  # 부위
                            "plant_name": matched_folder.split("_")[1],
                            "folder_name": matched_folder
                        })
                except Exception as e:
                    print(f"[오류] 파일 처리 실패: {e}")

        self.df = pd.DataFrame(dataset_records)
        return self.df