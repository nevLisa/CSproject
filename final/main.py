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
# 🔥 Utilities 파일에서 디코더 클래스 임포트
from Utils.Utilities import PlantLabelDecoder


class MultiTaskPlantModel(torch.nn.Module):
    def __init__(self, num_plants, num_parts):
        super(MultiTaskPlantModel, self).__init__()
        # ResNet50 모델 로드
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()  # 전이학습을 위한 기존 fc 레이어 무력화

        # 멀티 타겟을 위한 전결합층(FC) 레이어 분리 탑재
        self.fc_plant = torch.nn.Linear(num_features, num_plants)
        self.fc_part = torch.nn.Linear(num_features, num_parts)

    def forward(self, x):
        features = self.backbone(x)
        plant_output = self.fc_plant(features)
        part_output = self.fc_part(features)
        return {'plant': plant_output, 'part': part_output}


class HerbClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 약초 분류 프로그램 v1.0")
        self.setGeometry(100, 100, 950, 700)

        # 1. 디코더 가이드북 로드 (가장 먼저 수행하여 안전성 확보)
        try:
            self.decoder = PlantLabelDecoder(json_path="label_mapping.json")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"매핑 가이드북 로드 실패:\n{e}")
            sys.exit(1)

        # 2. 파싱 데이터베이스 빌드 (기존 유지)
        self.base_json_dir = "./data/라벨링데이터"
        self.base_img_dir = "./data/원천데이터"
        self.data_manager = PlantDataManager(base_json_dir=self.base_json_dir, base_img_dir=self.base_img_dir)
        self.df = self.data_manager.parse_all_image()

        # 3. 디바이스 및 멀티 타겟 모델 장착
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[UI 통합] 추론 디바이스 설정 완료: {self.device}")

        # 🔥 중요: 모델의 출력 차원을 label_mapping.json에 등록된 개수와 정확히 일치시킵니다.
        num_plants = len(self.decoder.plant_mapping)
        num_parts = len(self.decoder.part_mapping)

        self.model = MultiTaskPlantModel(num_plants=num_plants, num_parts=num_parts)

        # 4. 학습된 가중치 파일(.pth) 로드
        model_path = "multi_task_plant_model.pth"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[성공] 진짜 AI 모델({model_path}) 장착 완료!")
            except Exception as e:
                print(f"[경고] 모델 가중치 로드 실패: {e}.")
        else:
            print(f"[위험] {model_path} 파일이 없습니다!")

        self.model.to(self.device)
        self.model.eval()  # 평가/추론 전용 모드 전환

        # 5. 화면 전환용 컨테이너 생성 및 UI 초기화
        self.central_stacked_widget = QStackedWidget()
        self.setCentralWidget(self.central_stacked_widget)

        self.init_main_menu_ui()
        self.init_how_to_use_ui()
        self.init_upload_wait_ui()
        self.init_analysis_result_ui()

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
            pix = QPixmap(file_path)
            self.lbl_user_photo.setPixmap(
                pix.scaled(self.lbl_user_photo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            try:
                img_array = np.fromfile(file_path, np.uint8)
                raw_cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if raw_cv_img is None:
                    QMessageBox.warning(self, "오류", "이미지 파일을 분석 규격으로 디코딩할 수 없습니다.")
                    return

            except Exception as e:
                QMessageBox.critical(self, "오류", f"이미지를 읽는 중 치명적 오류 발생: {e}")
                return

            self.render_ai_predictions(raw_cv_img, file_path)
            self.central_stacked_widget.setCurrentIndex(3)

    def render_ai_predictions(self, raw_cv_img, file_path):
        try:
            h, w = raw_cv_img.shape[:2]
            target_w, target_h = 224, 224

            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_img = cv2.resize(raw_cv_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            processed_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            processed_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 전처리 중 오류 발생: {e}")
            return

        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float() / 255.0

        # 🔥 [필수 추가] 학습 환경과 완벽히 동일한 정규화(Normalization) 적용
        # 이 세 줄이 없으면 모델이 들어온 이미지를 완전히 다르게 인식하여 한 가지 정답만 찍습니다!
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            plant_probs = torch.softmax(outputs['plant'], dim=1)[0].cpu().numpy()
            part_probs = torch.softmax(outputs['part'], dim=1)[0].cpu().numpy()

        pred_plant_idx = np.argmax(plant_probs)
        pred_part_idx = np.argmax(part_probs)

        # Utilities의 PlantLabelDecoder를 이용해 인덱스 복원 수행 (텍스트 창 출력용 한글 유지)
        decoded = self.decoder.decode(pred_plant_idx, pred_part_idx)

        if decoded["success"]:
            predicted_plant_name = decoded["plant_name"]
            predicted_part_name = decoded["part_name"]
        else:
            predicted_plant_name = "알 수 없는 약초"
            predicted_part_name = "미확인 부위"
            print(f"[디코딩 실패 로그]: {decoded.get('error')}")

        # 분석 결과 데이터를 기반으로 우측 정보 창 텍스트 구성
        info_database = {
            "도라지": "<b>생약명:</b> 길경(桔梗)<br><b>주요 효능:</b> 폐기를 잘 통하게 하고 가래를 삭이며 고름을 빼주는 작용이 강해 기침, 목구멍 통증에 명약입니다.",
            "더덕": "<b>생약명:</b> 사삼(沙參)<br><b>주요 효능:</b> 음기를 보하고 폐를 부드럽게 하며, 기침을 멈추게 하고 고름을 배출하며 종기를 치료하는 데 효과가 좋습니다.",
            "참당귀": "<b>생약명:</b> 당귀(當歸)<br><b>주요 효능:</b> 대표적인 보혈 약재로, 혈액 순환을 활발히 하고 월경을 조절하며 통증을 멎게 하는 신비한 효능이 있습니다.",
            "백도라지": "<b>생약명:</b> 길경(桔梗) 주요 효능: 흰 꽃이 피는 도라지로, 폐를 맑게 하고 기관지 염증을 가라앉히며 몸의 면역력을 높여주는 효능이 있습니다.",
            "미국자리공": "<b>생약명:</b> 미상륙(美商陸) 주요 효능: 뿌리와 열매에 강한 독성이 있는 식물로, 한방에서 제한적으로 부종을 내리는 이뇨제로 쓰이나 잘못 섭취하면 구토와 마비를 유발하므로 절대 주의해야 합니다.",
            "지리강활(개당귀)": "<b>생약명:</b> 약용 불가(독초) 주요 효능: 약재로 사용되지 않는 치명적인 독초(개당귀)입니다. 참당귀와 외형이 매우 비슷해 오인하기 쉬우며, 심각한 중독을 일으키므로 절대 채취하거나 섭취하면 안 됩니다.",
            "왜당귀": "<b>생약명:</b> 일당귀(日當歸) 주요 효능: 참당귀에 비해 피를 보충하는 효능은 적지만, 혈액 순환을 활발하게 돕는 작용이 뛰어나고 향이 좋아 식용 쌈채소로도 인기가 많습니다.",
            "하수오": "<b>생약명:</b> 하수오(何首烏) 주요 효능: 간과 신장을 보하여 기혈을 북돋우고 흰머리를 검게 만들어주며, 뼈와 힘줄을 튼튼하게 하여 노화를 예방하는 신비한 효능이 있습니다.",
            "박주가리": "<b>생약명:</b> 나마근(蘿藦根) 주요 효능: 몸이 허약한 것을 보하고 정력을 강화하며, 해독 작용과 함께 산모의 젖을 잘 돌게 하는 효능이 있습니다. 다만 줄기 속 흰 진액의 독성을 주의해야 합니다.",
            "자리공": "<b>생약명:</b> 상륙(商陸) 주요 효능: 몸의 심한 부종을 내리고 대소변을 잘 통하게 하는 강력한 이뇨 효능이 있으나, 뿌리에 독성이 매우 강해 일반인이 함부로 사용해서는 안 됩니다."
        }

        desc_info = info_database.get(predicted_plant_name, "동의보감에 등록된 식물 정보를 탐색 중입니다. 안전한 생약 처방 연구 가이드라인을 참조하세요.")

        html_format = (
            f"<b>[AI 판별 식물 종류]:</b> <font color='blue'>{predicted_plant_name}</font><br>"
            f"<b>[AI 검출 판별 부위]:</b> {predicted_part_name}<br><br>"
            f"<b>[동의보감 약리 효능 가이드]:</b><br>{desc_info}"
        )
        self.lbl_desc_text.setText(html_format)

        # -------------------------------------------------------------
        # 📊 [패치 적용] 영어 이름을 사용해 OpenCV 기본 내장 폰트로 출력하기
        # -------------------------------------------------------------
        canvas = np.full((180, 450, 3), 245, dtype=np.uint8)
        top_indices = np.argsort(plant_probs)[::-1][:3]

        # 10종의 약초 인덱스에 대응하는 영문 직관적 매핑 테이블
        eng_plant_mapping = {
            0: "Doraji",
            1: "White Doraji",
            2: "Deodeok",
            3: "Pokeweed(US)",
            4: "Chamdangwi",
            5: "Oedangwi",
            6: "Gaedangwi",
            7: "Hasuo",
            8: "Bakjugari",
            9: "Jarigong"
        }

        for idx, item_idx in enumerate(top_indices):
            score = plant_probs[item_idx] * 100

            # 지정된 딕셔너리에서 영어 이름을 가져옵니다.
            lbl_name_eng = eng_plant_mapping.get(int(item_idx), f"Unknown_{item_idx}")

            bar_width = int(score * 3)
            y_offset = 25 + (idx * 45)

            bar_color = (90, 185, 120) if idx == 0 else (180, 180, 180)
            cv2.rectangle(canvas, (110, y_offset), (110 + bar_width, y_offset + 25), bar_color, -1)

            # 영어로만 출력하므로 cv2.putText가 절대 깨지지 않고 쾌속으로 출력됩니다.
            cv2.putText(canvas, f"ID {item_idx}", (10, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, f"{score:.1f}% ({lbl_name_eng})", (120 + bar_width, y_offset + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1, cv2.LINE_AA)
        # -------------------------------------------------------------

        h, w, c = canvas.shape
        q_img = QImage(canvas.data, w, h, w * c, QImage.Format_BGR888)
        self.lbl_chart.setPixmap(QPixmap.fromImage(q_img))

        resized_preview = cv2.resize(raw_cv_img, (300, 240))
        preview_h, preview_w, preview_c = resized_preview.shape
        q_preview = QImage(resized_preview.data, preview_w, preview_h, preview_w * preview_c, QImage.Format_BGR888)
        self.lbl_std_photo.setPixmap(QPixmap.fromImage(q_preview))

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