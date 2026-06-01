import sys
import os
import numpy as np
import cv2
import torch
import torchvision.models as models

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# 백엔드 모듈 임포트
from Utils.data_manager import PlantDataManager, PART_MAP
from Utils.dataset import PlantDataset


class MultiTaskPlantModel(torch.nn.Module):
    def __init__(self, num_plants, num_parts):
        super(MultiTaskPlantModel, self).__init__()
        # 순정 ResNet50 모델 로드
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()  # 기존 싱글 타겟 fc 레이어 무력화

        # 멀티 타겟을 위한 독립된 전결합층(FC) 레이어 분리 탑재
        self.fc_plant = torch.nn.Linear(num_features, num_plants)
        self.fc_part = torch.nn.Linear(num_features, num_parts)

    def forward(self, x):
        features = self.backbone(x)
        plant_output = self.fc_plant(features)
        part_output = self.fc_part(features)
        # main.py 하단 추론 로직과 호환되도록 딕셔너리 형태로 반환
        return {'plant': plant_output, 'part': part_output}

class HerbClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 약초 분류 프로그램 v1.0")
        self.setGeometry(100, 100, 950, 700)

        # 1. AIHub 데이터 경로 최신화 및 파싱 데이터베이스 빌드
        self.base_json_dir = "./data/라벨링데이터"
        self.base_img_dir = "./data/원천데이터"

        self.data_manager = PlantDataManager(base_json_dir=self.base_json_dir, base_img_dir=self.base_img_dir)
        # 데이터프레임 구조 분석 및 한글 식물 클래스 ID 맵 자동 빌드 시동
        self.df = self.data_manager.parse_all_image()

        # 역방향 조회를 위한 부위 맵 생성 (ID 정수 -> 한글 이름)
        self.id_to_part = {v: k for k, v in PART_MAP.items()}

        # 2. 실시간 AI 추론 신경망(ResNet50) 인스턴스 장착
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[UI 통합] 추론 디바이스 설정 완료: {self.device}")

        # 동적 클래스 개수 할당 (학습 코드와 동일 규격 맞춤)
        num_plants = len(self.data_manager.plant_name_to_label) if hasattr(self.data_manager,
                                                                           'plant_name_to_label') else 1
        num_parts = len(PART_MAP)

        # 학습 때 사용했던 멀티 타겟 ResNet50 모델 뼈대 구조 그대로 정의
        #self.model = models.resnet50(pretrained=False)
        #num_features = self.model.fc.in_features

        # 멀티 타겟 출력을 위한 커스텀 구조 변환 (식물 분류기 & 부위 분류기 독립 탑재)
        #self.model.fc = torch.nn.ModuleDict({
        #    'plant': torch.nn.Linear(num_features, num_plants),
        #    'part': torch.nn.Linear(num_features, num_parts)
        #})
        # 2. [수정] 위에서 정의한 커스텀 멀티 타겟 모델 구조로 인스턴스 장착
        num_plants = len(self.data_manager.plant_name_to_label) if hasattr(self.data_manager,
                                                                           'plant_name_to_label') else 1
        num_parts = len(PART_MAP)

        #깔끔한 멀티 타겟 뼈대로 교체합니다.
        self.model = MultiTaskPlantModel(num_plants=num_plants, num_parts=num_parts)

        # 3. 학습된 진짜 가중치 파일(.pth) 파일 불러오기
        model_path = "multi_task_plant_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[UI 통합] 추론 디바이스 설정 완료: {self.device}")

        if os.path.exists(model_path):
            try:
                # 가중치를 안전하게 로드합니다.
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[성공] 진짜 AI 모델({model_path}) 장착 완료!")
            except Exception as e:
                print(f"[경고] 모델 가중치 로드 실패: {e}.")
        else:
            print(f"[위험] {model_path} 파일이 없습니다!")

        self.model.to(self.device)
        self.model.eval()  # 평가/추론 전용 모드 전환

        # 3. 학습된 진짜 가중치 파일(.pth) 파일 불러오기
        model_path = "multi_task_plant_model.pth"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[성공] 진짜 AI 모델({model_path}) 장착 완료!")
            except Exception as e:
                print(f"[경고] 모델 가중치 로드 실패: {e}. 더미 가중치 상태로 시뮬레이션됩니다.")
        else:
            print(f"[위험] {model_path} 파일이 프로젝트 폴더에 없습니다! 학습을 먼저 완료해 주세요.")

        self.model.to(self.device)
        self.model.eval()  # 평가/추론 전용 모드 전환

        # 화면 전환용 컨테이너 생성
        self.central_stacked_widget = QStackedWidget()
        self.setCentralWidget(self.central_stacked_widget)

        # 4가지 화면 생성 및 등록
        self.init_main_menu_ui()
        self.init_how_to_use_ui()
        self.init_upload_wait_ui()
        self.init_analysis_result_ui()

        # 첫 시작은 메인 메뉴 (Index 0)
        self.central_stacked_widget.setCurrentIndex(0)

    # 1. 메인 메뉴 화면
    def init_main_menu_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        title_lbl = QLabel("AI 약초 분류 프로그램 v1.0")
        title_lbl.setFont(QFont("Malgun Gothic", 24, QFont.Bold))
        title_lbl.setAlignment(Qt.AlignCenter)

        btn_start = QPushButton("시작하기")
        btn_howto = QPushButton("사용방법")
        btn_exit = QPushButton("종료하기")

        btn_font = QFont("Malgun Gothic", 12)
        for btn in [btn_start, btn_howto, btn_exit]:
            btn.setFont(btn_font)
            btn.setFixedSize(260, 50)

        warning_lbl = QLabel("※ AI는 전문가가 아니므로 사용시 주의가 필요합니다.")
        warning_lbl.setStyleSheet("color: red; font-weight: bold;")
        warning_lbl.setFont(QFont("Malgun Gothic", 11))
        warning_lbl.setAlignment(Qt.AlignCenter)

        btn_start.clicked.connect(lambda: self.central_stacked_widget.setCurrentIndex(2))
        btn_howto.clicked.connect(lambda: self.central_stacked_widget.setCurrentIndex(1))
        btn_exit.clicked.connect(self.ask_close_program)

        layout.addStretch(2)
        layout.addWidget(title_lbl)
        layout.addStretch(1)
        layout.addWidget(btn_start)
        layout.addWidget(btn_howto)
        layout.addWidget(btn_exit)
        layout.addStretch(3)
        layout.addWidget(warning_lbl)

        widget.setLayout(layout)
        self.central_stacked_widget.addWidget(widget)

    # 2. 사용방법 설명 화면
    def init_how_to_use_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()

        top_bar = QHBoxLayout()
        top_bar.addStretch(1)
        btn_menu = QPushButton("메뉴로 돌아가기")
        btn_start = QPushButton("시작하기")
        btn_menu.clicked.connect(lambda: self.central_stacked_widget.setCurrentIndex(0))
        btn_start.clicked.connect(lambda: self.central_stacked_widget.setCurrentIndex(2))
        top_bar.addWidget(btn_menu)
        top_bar.addWidget(btn_start)
        layout.addLayout(top_bar)

        instructions = QLabel(
            "\n[ AI 약초 분류 시스템 사용 방법 ]\n\n"
            "1. 메인 화면이나 내비게이션 바에서 '시작하기'를 클릭합니다.\n"
            "2. '이미지 업로드 하기' 버튼을 통해 판별하고자 하는 약초 사진을 선택합니다.\n"
            "3. 시스템이 자동으로 이미지를 조절(Pad & Resize)한 뒤 학습된 AI 모델로 분석을 시작합니다.\n"
            "4. 결과 화면에서 매칭된 동의보감 표준 약초 정보와 부위, 그리고 신뢰도 그래프를 확인합니다.\n\n"
            "※ 주의: AI 가이드 라인은 참고용이며 의학적 처방을 대신할 수 없습니다."
        )
        instructions.setFont(QFont("Malgun Gothic", 12))
        instructions.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        instructions.setStyleSheet("background-color: #F9F9F9; padding: 15px; border-radius: 5px;")

        layout.addWidget(instructions)
        layout.addStretch(1)
        widget.setLayout(layout)
        self.central_stacked_widget.addWidget(widget)

    # 3. 이미지 업로드 대기 화면
    def init_upload_wait_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()

        top_bar = QHBoxLayout()
        top_bar.addStretch(1)
        btn_menu = QPushButton("메뉴로 돌아가기")
        btn_menu.clicked.connect(self.ask_go_back_menu)
        top_bar.addWidget(btn_menu)
        layout.addLayout(top_bar)

        layout.addStretch(1)
        guide_lbl = QLabel("이미지를 업로드 해 주세요")
        guide_lbl.setFont(QFont("Malgun Gothic", 16, QFont.Bold))
        guide_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(guide_lbl)

        btn_upload = QPushButton("이미지 업로드 하기")
        btn_upload.setFont(QFont("Malgun Gothic", 12))
        btn_upload.setFixedSize(200, 50)
        btn_upload.clicked.connect(self.process_image_upload)

        btn_area = QHBoxLayout()
        btn_area.setAlignment(Qt.AlignCenter)
        btn_area.addWidget(btn_upload)
        layout.addLayout(btn_area)
        layout.addStretch(2)

        widget.setLayout(layout)
        self.central_stacked_widget.addWidget(widget)

    # 4. 분석 결과 화면
    def init_analysis_result_ui(self):
        widget = QWidget()
        main_layout = QVBoxLayout()

        top_bar = QHBoxLayout()
        top_bar.addStretch(1)
        btn_menu = QPushButton("메뉴로 돌아가기")
        btn_close = QPushButton("종료하기")
        btn_menu.clicked.connect(self.ask_go_back_menu)
        btn_close.clicked.connect(self.ask_close_program)
        top_bar.addWidget(btn_menu)
        top_bar.addWidget(btn_close)
        main_layout.addLayout(top_bar)

        body_layout = QHBoxLayout()

        # 좌측 이미지 배정 섹션
        left_layout = QVBoxLayout()
        self.lbl_std_photo = QLabel("AI 분석 식물 표준 사진")
        self.lbl_user_photo = QLabel("내가 업로드한 사진")

        for lbl in [self.lbl_std_photo, self.lbl_user_photo]:
            lbl.setFixedSize(300, 240)
            lbl.setStyleSheet("border: 1px solid darkgray; background-color: #EAEAEA;")
            lbl.setAlignment(Qt.AlignCenter)

        left_layout.addWidget(QLabel("<b>[AI 분석 식물 종류 사진]</b>"))
        left_layout.addWidget(self.lbl_std_photo)
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("<b>[업로드한 사진]</b>"))
        left_layout.addWidget(self.lbl_user_photo)

        # 우측 설명 및 텍스트/차트 섹션
        right_layout = QVBoxLayout()
        self.lbl_desc_text = QLabel("약초 효능 설명란")
        self.lbl_desc_text.setWordWrap(True)
        self.lbl_desc_text.setStyleSheet("border: 1px solid gray; background-color: white; padding: 10px;")
        self.lbl_desc_text.setFont(QFont("Malgun Gothic", 11))

        self.lbl_chart = QLabel("AI의 Confident 그래프 공간")
        self.lbl_chart.setFixedSize(450, 180)
        self.lbl_chart.setStyleSheet("border: 1px solid darkgray; background-color: #FAFAFA;")

        right_layout.addWidget(QLabel("<b>[약초 설명 및 효능]</b>"))
        right_layout.addWidget(self.lbl_desc_text, 1)
        right_layout.addSpacing(10)
        right_layout.addWidget(QLabel("<b>[AI의 Confident 그래프]</b>"))
        right_layout.addWidget(self.lbl_chart)

        body_layout.addLayout(left_layout)
        body_layout.addSpacing(20)
        body_layout.addLayout(right_layout)

        main_layout.addLayout(body_layout)
        widget.setLayout(main_layout)
        self.central_stacked_widget.addWidget(widget)

    # --- 실시간 기능 제어 로직 ---
    def process_image_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "약초 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # 1. 사용자가 업로드한 원본 사진 화면 출력 연동
            pix = QPixmap(file_path)
            self.lbl_user_photo.setPixmap(
                pix.scaled(self.lbl_user_photo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # 2. 한글 경로 안전 지원 방식으로 파일 읽기 및 디코딩 검증
            try:
                img_array = np.fromfile(file_path, np.uint8)
                raw_cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if raw_cv_img is None:
                    QMessageBox.warning(self, "오류", "이미지 파일을 분석 규격으로 디코딩할 수 없습니다.")
                    return

            except Exception as e:
                QMessageBox.critical(self, "오류", f"이미지를 읽는 중 치명적 오류 발생: {e}")
                return

            # 3. 인공지능 결과 화면 갱신 및 전환
            self.render_ai_predictions(raw_cv_img, file_path)
            self.central_stacked_widget.setCurrentIndex(3)

            # 3. 진짜 AI 엔진 구동 및 분석 결과 화면 표시
            self.render_ai_predictions(raw_cv_img, file_path)
            self.central_stacked_widget.setCurrentIndex(3)

    def render_ai_predictions(self, raw_cv_img, file_path):
        # [안전장치] dataset.py를 거치지 않고 main.py 내부에서 직접 레터박스(Pad & Resize) 전처리 수행
        try:
            h, w = raw_cv_img.shape[:2]
            target_w, target_h = 224, 224

            # 비율 유지 리사이즈 크기 계산
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_img = cv2.resize(raw_cv_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 검은색 바탕 가로세로 224 패딩 이미지 생성 (학습 환경과 100% 동일 규격)
            processed_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            processed_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 전처리 중 오류 발생: {e}")
            return

        # BGR -> RGB 채널 변경
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        # PyTorch 입력 데이터 규격 변환 및 정규화 (CHW 구조)
        img_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Batch 차원 추가 및 GPU 전송

        # 신경망 실시간 순전파 추론 작동
        with torch.no_grad():
            outputs = self.model(img_tensor)

            # 식물 종류 및 부위의 Softmax 확률 점수 도출
            plant_probs = torch.softmax(outputs['plant'], dim=1)[0].cpu().numpy()
            part_probs = torch.softmax(outputs['part'], dim=1)[0].cpu().numpy()

        # 최고 높은 확률값과 인덱스 번호 추출
        pred_plant_idx = np.argmax(plant_probs)
        pred_part_idx = np.argmax(part_probs)

        # 인덱스 숫자를 한글 명칭 텍스트로 치환
        plant_name_map = self.data_manager.label_to_plant_name
        predicted_plant_name = plant_name_map.get(pred_plant_idx, "알 수 없는 약초")
        predicted_part_name = self.id_to_part.get(pred_part_idx, "미확인 부위")

        # 분석 결과 데이터를 기반으로 우측 정보 창 텍스트 구성
        info_database = {
            "가는장구채": "<b>생약명:</b> 왕불유행(王不留行)<br><b>주요 효능:</b> 혈액 순환을 강력히 촉진하고 월경 불순을 통하게 하며, 종기와 유선염의 붓기를 가라앉히는 데 탁월한 효과를 발휘합니다.",
            "장구채": "<b>생약명:</b> 왕불유행 대용<br><b>주요 효능:</b> 이뇨 작용이 뛰어나 몸의 부기를 빼주고 지혈 작용과 음기를 보충하는 성질이 있어 전통 한방에서 요긴하게 사용됩니다."
        }

        desc_info = info_database.get(predicted_plant_name, "동의보감에 등록된 식물 정보를 탐색 중입니다. 안전한 생약 처방 연구 가이드라인을 참조하세요.")

        html_format = (
            f"<b>[AI 판별 식물 종류]:</b> <font color='blue'>{predicted_plant_name}</font><br>"
            f"<b>[AI 검출 판별 부위]:</b> {predicted_part_name}<br><br>"
            f"<b>[동의보감 약리 효능 가이드]:</b><br>{desc_info}"
        )
        self.lbl_desc_text.setText(html_format)

        # OpenCV 드로잉을 통한 실시간 Confident 가로 바 차트 시각화 구현
        canvas = np.full((180, 450, 3), 245, dtype=np.uint8)

        # 상위 예측 스코어 정렬 빌드
        top_indices = np.argsort(plant_probs)[::-1][:3]  # 상위 최대 3개 식물 추출

        for idx, item_idx in enumerate(top_indices):
            score = plant_probs[item_idx] * 100
            lbl_name = plant_name_map.get(item_idx, "기타")

            # 가로 바 크기 연산 및 드로잉
            bar_width = int(score * 3)
            y_offset = 25 + (idx * 45)

            # 1위 매칭 식물은 특별히 강조 색상(녹색 계열)으로 바 드로잉
            bar_color = (90, 185, 120) if idx == 0 else (180, 180, 180)
            cv2.rectangle(canvas, (110, y_offset), (110 + bar_width, y_offset + 25), bar_color, -1)

            # 한글 깨짐 방지를 위해 인덱스 이름과 스코어 텍스트 삽입
            cv2.putText(canvas, f"ID {item_idx}", (10, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, f"{score:.1f}% ({lbl_name[:5]})", (120 + bar_width, y_offset + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1, cv2.LINE_AA)

        h, w, c = canvas.shape
        q_img = QImage(canvas.data, w, h, w * c, QImage.Format_BGR888)
        self.lbl_chart.setPixmap(QPixmap.fromImage(q_img))

        # 좌측 상단 AI 분석 원본 식물 사진 연동
        resized_preview = cv2.resize(raw_cv_img, (300, 240))
        preview_h, preview_w, preview_c = resized_preview.shape
        q_preview = QImage(resized_preview.data, preview_w, preview_h, preview_w * preview_c, QImage.Format_BGR888)
        self.lbl_std_photo.setPixmap(QPixmap.fromImage(q_preview))

    # 알림 팝업 메시지 박스 핸들러들
    def ask_go_back_menu(self):
        reply = QMessageBox.question(self, '돌아가기', '정말로 분석을 취소하고 메뉴로 돌아가시겠습니까?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.central_stacked_widget.setCurrentIndex(0)

    def ask_close_program(self):
        reply = QMessageBox.question(self, '종료 확인', '정말로 프로그램을 종료하시겠습니까?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '종료 확인', '정말로 프로그램을 종료하시겠습니까?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HerbClassifierApp()
    window.show()
    sys.exit(app.exec_())