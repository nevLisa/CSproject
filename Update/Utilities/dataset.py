import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image


class PlantDataset(Dataset):
    def __init__(self, manager_df, base_img_dir, target_size=(224, 224), pad_color=(0, 0, 0), transform=None):
        # manager_df: 메인에서 전달받은 데이터프레임
        # base_img_dir: 원본 이미지 폴더 경로
        self.df = manager_df.reset_index(drop=True)
        self.base_img_dir = base_img_dir
        self.target_size = target_size
        self.pad_color = pad_color
        self.transform = transform

    def __len__(self):
        return len(self.df)

    # 이미지 패딩 및 리사이즈
    def pad_and_resize(self, img):
        h, w = img.shape[:2]
        tw, th = self.target_size

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

        # 원본 이미지 경로 조립 및 윈도우 경로 규격 표준화
        raw_path = os.path.join(self.base_img_dir, str(row['folder_name']), str(row['image']))
        img_path = os.path.normpath(raw_path)

        # 한글 경로 우회하여 읽기
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            img = None

        # 이미지 읽기 실패 시 예외 처리
        if img is None:
            img = np.full((self.target_size[1], self.target_size[0], 3), self.pad_color, dtype=np.uint8)
        else:
            img = self.pad_and_resize(img)

        # OpenCV(BGR)를 PyTorch/Transform이 좋아하는 RGB 포맷으로 변경
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 🚨 [중요] transforms 데이터 증강이 에러 없이 작동하도록 PIL Image 객체로 변환
        img = Image.fromarray(img)

        # 🚨 [징검다리 라벨 버그 해결] 메인에서 가공한 깨끗한 0,1,2,3 라벨 컬럼을 가져옵니다.
        plant_label = torch.tensor(int(row['plant_label_clean']), dtype=torch.long)
        part_label = torch.tensor(int(row['part_label_clean']), dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        # 🚨 [메인 코드와 동기화] 딕셔너리 대신 파이토치 표준인 튜플 형태로 안전하게 리턴
        return img, (plant_label, part_label)

