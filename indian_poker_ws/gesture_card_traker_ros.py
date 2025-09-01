# -*- coding: utf-8 -*-
"""
코/귀/팔짱 3가지만 감지 + 스코어 표시 (FaceMesh 토글/그리기 토글 지원)
- Pose: 어깨/팔꿈치/손목 (필수)
- Hands: 손끝 (코/귀 판정 필수)
- FaceMesh: 귀 위치 안정화용 (옵션, USE_FACE_MESH로 on/off)
+ YOLO 카드 인식 기능 추가
ROS2 노드 버전 (기능 변경 없음) - 입력을 /camera/camera/color/image_raw 구독으로 변경
"""

import os
import cv2, math, time, numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError

# ================== 설정 ==================
CAM_INDEX         = 6    # (보존) 의미 없음. 이제 토픽 입력 사용.
DET_CONF          = 0.35
TRK_CONF          = 0.35

USE_FACE_MESH     = False # ← True/False 한 줄로 FaceMesh 완전 on/off
DRAW_LANDMARKS    = True  # ← True/False로 시각화 점/선 그리기 on/off
SHOW_DEBUG        = True

KOREAN_FONT       = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# --- 코/귀 임계 (어깨폭 정규화 기준) ---
EAR_THR_DIST      = 0.23   # 손끝-귀 거리 임계(작을수록 엄격)
NOSE_THR_DIST     = 0.23   # 손끝-코 거리 임계
NOSE_THR_DX       = 0.10   # 손끝이 코 중앙축에서 벗어나지 않는 x 허용

# --- 히스테리시스(연속 프레임) ---
EAR_ON_FRAMES     = 4
NOSE_ON_FRAMES    = 4
ARMS_ON_FRAMES    = 3

# --- 팔짱(ㄱ자) 파라미터 ---
ELBOW_MIN, ELBOW_MAX      = 60, 140     # 팔꿈치 각도 범위(도)
FOREARM_H_MAX             = 25          # 전완 수평 허용각(도, 0=수평)
UPPER_V_MIN, UPPER_V_MAX  = 65, 125     # 상완 수직 허용범위(도, 90=수직)
CHEST_BAND_RATIO          = 0.49        # 팔꿈치 높이: 어깨선 ± (ratio * shoulder_w)
X_PAD_RATIO               = 0.20        # 팔꿈치 x: 어깨폭 좌우 여유

# --- 팔짱 빠른 전환(비대칭 히스테리시스 + 빠른 EMA) ---
ARMS_ON_THRESH   = 0.55    # 켤 때
ARMS_OFF_THRESH  = 0.35    # 끌 때
ARMS_DECAY_FAST  = 2       # OFF 구간에서 카운트 빠르게 감소
ARMS_EMA_ALPHA   = 0.50    # 팔짱 EMA 가속

# ================== 유틸 ==================
def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def clamp(x,a,b): return a if x<a else b if x>b else x
def ema(prev, x, alpha=0.25): return x if prev is None else (alpha*x+(1-alpha)*prev)
def to_xy(lm, w, h): return (lm.x*w, lm.y*h)

def draw_kor(img, text, pos, size=30, color=(255,200,0)):
    try:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = ImageFont.truetype(KOREAN_FONT, size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2); return img

def angle_deg(a, b, c):
    v1 = (a[0]-b[0], a[1]-b[1]); v2 = (c[0]-b[0], c[1]-b[1])
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6: return 180.0
    cos = max(-1.0, min(1.0, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))
    return math.degrees(math.acos(cos))

def orient_deg(vx, vy):  # 0=수평, 90=수직
    return abs(math.degrees(math.atan2(vy, vx)))

# ================== MediaPipe ==================
mp_pose  = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# FaceMesh는 플래그로 토글 초기화
if USE_FACE_MESH:
    mp_face  = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                                 min_detection_confidence=DET_CONF, min_tracking_confidence=TRK_CONF)
else:
    face_mesh = None

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=DET_CONF, min_tracking_confidence=TRK_CONF)

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=DET_CONF, min_tracking_confidence=TRK_CONF)

TIP_IDXS = [4,8,12,16,20]  # 엄지~소지 끝

class CombinedDetectionNode(Node):
    """
    기존 기능 유지 + 입력만 ROS 이미지 토픽으로 변경
    - 구독: /camera/camera/color/image_raw
    - 창 띄우기 / 키 입력 / YOLO / MediaPipe 로직 동일
    """
    def __init__(self):
        super().__init__('combined_detection_node')
        self.get_logger().info('CombinedDetectionNode 시작 (source: /camera/camera/color/image_raw)')
        self.bridge = CvBridge()

        # YOLO 모델 로드 (경로는 기존 코드 그대로 유지)
        project_folder = "src/indian_poker_ws/models"
        model_filename = "card.pt"
        base_dir = os.getcwd()
        output_dir = os.path.join(base_dir, project_folder)
        self.model_path = os.path.join(output_dir, model_filename)

        if not os.path.exists(self.model_path):
            print(f"오류: YOLO 모델 파일을 찾을 수 없습니다: {self.model_path}")
            print("현재 폴더에 'models' 폴더를 만들고 그 안에 'card.pt' 파일을 넣어주세요.")
            # 모델이 없으면 돌려도 의미가 없으니 그대로 두되, 콜백에서 early return
            self.yolo_model = None
        else:
            self.yolo_model = YOLO(self.model_path)
            print("YOLO 모델 로드 완료.")

        # 상태 변수(프레임 간 유지)
        self.ear_cnt = 0
        self.nose_cnt = 0
        self.arms_cnt = 0
        self.ear_score_ema = 0.0
        self.nose_score_ema = 0.0
        self.arms_score_ema = 0.0
        self.t_prev = time.time()
        self.fps = 0.0

        # 구독 시작
        self.sub = self.create_subscription(
            RosImage,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg: RosImage):
        global SHOW_DEBUG

        # 1) ROS Image -> OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().warn(f'CvBridge 변환 실패: {e}')
            return

        # (기존 동작 보존) 좌우반전
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 2) YOLO 카드 인식
        if self.yolo_model is not None:
            yolo_results = self.yolo_model.predict(source=frame, conf=0.6, verbose=False, device=0)
            result = yolo_results[0]
            boxes = result.boxes
            classes = result.names
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                label = f"{classes[cls_id]} {conf:.1f}%"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 3) MediaPipe 포즈/손/페이스
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_pose  = pose.process(rgb)
        res_face  = (face_mesh.process(rgb) if face_mesh is not None else None)
        res_hands = hands.process(rgb)

        # ---- 그리기(옵션) ----
        if DRAW_LANDMARKS:
            if res_pose.pose_landmarks:
                mp_draw.draw_landmarks(frame, res_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=mp_style.get_default_pose_landmarks_style())
            if face_mesh is not None and res_face and res_face.multi_face_landmarks:
                for fl in res_face.multi_face_landmarks:
                    mp_draw.draw_landmarks(frame, fl, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                           landmark_drawing_spec=None,
                                           connection_drawing_spec=mp_draw.DrawingSpec(thickness=1, circle_radius=1))
            if res_hands.multi_hand_landmarks:
                for hl in res_hands.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # 초기값
        ear_score = nose_score = arms_score = 0.0
        picked = "중립"

        if res_pose.pose_landmarks:
            plm = res_pose.pose_landmarks.landmark
            def P(i): return to_xy(plm[i], w, h)
            NOSE = mp_pose.PoseLandmark.NOSE.value
            L_EAR,R_EAR = mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value
            L_SHO,R_SHO = mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            L_ELB,R_ELB = mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value
            L_WRI,R_WRI = mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value

            nose_p = P(NOSE)
            l_sho, r_sho = P(L_SHO), P(R_SHO)
            l_elb, r_elb = P(L_ELB), P(R_ELB)
            l_wri, r_wri = P(L_WRI), P(R_WRI)

            shoulder_w = max(1e-6, dist(l_sho, r_sho))
            def nd(a,b): return dist(a,b)/shoulder_w

            if face_mesh is not None and res_face and res_face.multi_face_landmarks:
                l_ear, r_ear = P(L_EAR), P(R_EAR)
            else:
                l_ear, r_ear = P(L_EAR), P(R_EAR)

            # (1) 귀 만지기 스코어
            ear_prox = 0.0
            if res_hands.multi_hand_landmarks:
                for hl in res_hands.multi_hand_landmarks:
                    for tip_idx in TIP_IDXS:
                        tip = to_xy(hl.landmark[tip_idx], w, h)
                        d_ear = min(nd(tip, l_ear), nd(tip, r_ear))
                        s = 1.0 - clamp(d_ear / EAR_THR_DIST, 0.0, 1.0)
                        if s > ear_prox: ear_prox = s
                        if SHOW_DEBUG:
                            cv2.circle(frame, (int(tip[0]),int(tip[1])), 5, (0,0,255), -1)
            ear_score = ear_prox
            self.ear_score_ema = ema(self.ear_score_ema, ear_score, 0.25)
            self.ear_cnt = self.ear_cnt + 1 if self.ear_score_ema > 0.5 else max(0, self.ear_cnt-1)
            ear_active = self.ear_cnt >= EAR_ON_FRAMES

            # (2) 코 만지기 스코어
            nose_prox = 0.0
            if res_hands.multi_hand_landmarks:
                for hl in res_hands.multi_hand_landmarks:
                    for tip_idx in TIP_IDXS:
                        tip = to_xy(hl.landmark[tip_idx], w, h)
                        d_nose = nd(tip, nose_p)
                        dx = abs(tip[0] - nose_p[0]) / shoulder_w
                        if d_nose < NOSE_THR_DIST and dx < NOSE_THR_DX:
                            s = 1.0 - clamp(d_nose / NOSE_THR_DIST, 0.0, 1.0)
                            if s > nose_prox: nose_prox = s
            nose_score = nose_prox
            self.nose_score_ema = ema(self.nose_score_ema, nose_score, 0.25)
            self.nose_cnt = self.nose_cnt + 1 if self.nose_score_ema > 0.5 else max(0, self.nose_cnt-1)
            nose_active = self.nose_cnt >= NOSE_ON_FRAMES

            # (3) 팔짱 스코어
            x_min, x_max = min(l_sho[0], r_sho[0]), max(l_sho[0], r_sho[0])
            shoulder_y = min(l_sho[1], r_sho[1])
            band_h = CHEST_BAND_RATIO * shoulder_w
            xpad   = X_PAD_RATIO    * shoulder_w

            def side_score(sho, elb, wri):
                u = (sho[0]-elb[0], sho[1]-elb[1]); f = (wri[0]-elb[0], wri[1]-elb[1])
                ang = angle_deg(sho, elb, wri)
                elbow_ok = ELBOW_MIN <= ang <= ELBOW_MAX
                u_a = orient_deg(*u); f_a = orient_deg(*f)
                upper_v  = UPPER_V_MIN <= u_a <= UPPER_V_MAX
                fore_h   = (f_a <= FOREARM_H_MAX or f_a >= (180-FOREARM_H_MAX))
                height_ok = (shoulder_y - band_h <= elb[1] <= shoulder_y + 1.2*band_h)
                inside_x  = (x_min - xpad <= elb[0] <= x_max + xpad)
                ok_count = int(elbow_ok) + int(upper_v) + int(fore_h) + int(height_ok) + int(inside_x)
                if ok_count <= 1: return 0.0
                return [0.0, 0.25, 0.50, 0.75, 1.00][ok_count-1]

            L_sc = side_score(l_sho, l_elb, l_wri)
            R_sc = side_score(r_sho, r_elb, r_wri)
            arms_score = clamp((L_sc + R_sc)/2.0, 0.0, 1.0)
            self.arms_score_ema = ema(self.arms_score_ema, arms_score, ARMS_EMA_ALPHA)

            if L_sc <= 0.25 and R_sc <= 0.25:
                self.arms_cnt = 0
                self.arms_score_ema = min(self.arms_score_ema, 0.2)
            else:
                if self.arms_score_ema >= ARMS_ON_THRESH:
                    self.arms_cnt = min(ARMS_ON_FRAMES, self.arms_cnt + 1)
                elif self.arms_score_ema <= ARMS_OFF_THRESH:
                    self.arms_cnt = max(0, self.arms_cnt - ARMS_DECAY_FAST)
                else:
                    self.arms_cnt = max(0, self.arms_cnt - 1)
            arms_active = (self.arms_cnt >= ARMS_ON_FRAMES)

            # 우선순위
            if nose_active and (self.nose_score_ema >= self.ear_score_ema + 0.08):
                picked = "코 만지기"
            elif ear_active and (self.ear_score_ema >= self.nose_score_ema + 0.08):
                picked = "귀 만지기"
            elif arms_active:
                picked = "팔짱"
            else:
                picked = "중립"

            if DRAW_LANDMARKS:
                for p in [l_sho, r_sho, l_elb, r_elb, l_wri, r_wri, l_ear, r_ear, nose_p]:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (80,255,80), -1)

        # ---------- 스코어 보드 및 상태 표시 ----------
        board = f"Ear:{self.ear_score_ema:0.2f} Nose:{self.nose_score_ema:0.2f} Arms:{self.arms_score_ema:0.2f}"
        cv2.putText(frame, board, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,50,50), 2)
        frame = draw_kor(frame, picked, (10, 58), 32, (255,200,0))

        t_now = time.time()
        self.fps = 0.9*(1.0/max(1e-3, t_now - self.t_prev)) + 0.1*self.fps
        self.t_prev = t_now
        status = [f"FPS:{self.fps:4.1f}", f"POSE:{1 if (res_pose and res_pose.pose_landmarks) else 0}",
                  f"HANDS:{len(res_hands.multi_hand_landmarks) if (res_hands and res_hands.multi_hand_landmarks) else 0}",
                  f"FACE:{1 if (face_mesh is not None and res_face and res_face.multi_face_landmarks) else 0}"]
        cv2.putText(frame, "  ".join(status), (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,255,120), 2)

        cv2.imshow("Combined Detection (Pose + Card)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.get_logger().info('ESC 입력, 종료합니다.')
            # 창 정리 후 셧다운
            cv2.destroyAllWindows()
            rclpy.shutdown()
        if key == ord('d'):
            SHOW_DEBUG = not SHOW_DEBUG

def main():
    rclpy.init()
    node = CombinedDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
