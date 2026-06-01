import os
import cv2
from torch.utils.data import Dataset
import numpy as np


class PlantDataset(Dataset):
    def __init__(self, manager_df, base_img_dir, target_size=(224, 224), pad_color=(0, 0, 0), transform=None):

        #manager_df: Manager.py 에서 만든 데이터프레임
        #base_img_dir: 원본 이미지 폴더 경로

        self.df = manager_df
        self.base_img_dir = base_img_dir
        self.target_size = target_size
        self.pad_color = pad_color
        self.transform = transform

    def __len__(self):
        return len(self.df)

    #이미지 패딩
        # dataset.py 내부의 기존 함수 위에 @staticmethod를 붙이고 매개변수를 명시해 줍니다.
    @staticmethod
    def pad_and_resize(img, target_size=(224, 224), pad_color=(0, 0, 0)):
        h, w = img.shape[:2]
        tw, th = target_size  # self 대신 매개변수로 받은 target_size 사용

        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        padded_img = np.full((th, tw, 3), self.pad_color, dtype=np.uint8)

        x_offset = (tw - nw) // 2
        y_offset = (th - nh) // 2
        padded_img[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized_img

        return padded_img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 식물명과 부위명을 조합한 이미지 경로 생성
        raw_path = os.path.join(
            self.base_img_dir,
            str(row['plant_name']),
            str(row['part_str']),
            str(row['image'])
        )
        img_path = os.path.normpath(raw_path)

        # 멀티 타겟 레이블 미리 추출
        plant_label = int(row['plant_label'])
        part_label = int(row['part_label'])

        # 한글 경로 지원 방식으로 이미지 파일 로드
        try:
            if not os.path.exists(img_path):
                # 이미지가 없을 경우 경고를 띄우고, 학습이 멈추지 않도록 즉시 '더미 텐서' 반환
                # print(f"[경고] 이미지 파일 없음: {img_path}")
                import torch
                dummy_img = torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)
                return dummy_img, (plant_label, part_label)

            img_array = np.fromfile(img_path, np.uint8)
            raw_cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if raw_cv_img is None:
                import torch
                dummy_img = torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)
                return dummy_img, (plant_label, part_label)

        except Exception as e:
            import torch
            dummy_img = torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)
            return dummy_img, (plant_label, part_label)

        # ----------------------------------------------------
        # 이미지가 성공적으로 로드되었을 때만 전처리 수행
        # ----------------------------------------------------

        # 규격에 맞게 패딩 및 리사이즈 전처리 적용 (에러 방지를 위해 인자 명시)
        processed_img = PlantDataset.pad_and_resize(raw_cv_img, target_size=self.target_size, pad_color=self.pad_color)

        # BGR -> RGB 채널 변경
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        # PyTorch 텐서 변환 연산 (transform) 적용
        if self.transform:
            augmented = self.transform(image=processed_img) if hasattr(self.transform,
                                                                       'processors') else self.transform(processed_img)
            if isinstance(augmented, dict) and 'image' in augmented:
                img_tensor = augmented['image']
            else:
                img_tensor = augmented
        else:
            import torch
            img_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float() / 255.0

        return img_tensor, (plant_label, part_label)
