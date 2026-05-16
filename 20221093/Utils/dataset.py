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
        #데이터프레임에서 한 줄 읽기
        row = self.df.iloc[idx]

        # 원본 이미지 경로 조립 및 윈도우 경로 규격 표준화 (역슬래시 꼬임 방지) by gemini
        raw_path = os.path.join(self.base_img_dir, str(row['folder_name']), str(row['image']))
        img_path = os.path.normpath(raw_path)

        # [핵심 수정] 한글 경로 우회하여 읽기
        try:
            # 파일을 바이너리(기계어)로 먼저 읽어온 후, OpenCV로 이미지 디코딩
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            img = None

        #이미지 읽기 실패 시
        if img is None:
            img = np.full((self.target_size[1], self.target_size[0], 3), self.pad_color, dtype=np.uint8)
        else:
            #패딩 수행
            img = self.pad_and_resize(img)

        # OpenCV(BGR)를 PyTorch/Matplotlib(RGB)로 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 정답 라벨 및 데이터 증강
        plant_label = row['plant_label']
        part_label = row['part_label']

        if self.transform:
            img = self.transform(img)

        return img, {'plant_label': plant_label, 'part_label': part_label}

