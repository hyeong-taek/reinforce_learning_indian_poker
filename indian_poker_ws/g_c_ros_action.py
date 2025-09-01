# camera_actions_node.py
# -*- coding: utf-8 -*-
import os, re, time, threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
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
from poker_interfaces.action import DetectCardOnce, ObserveGestures

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
        self.cb_group = ReentrantCallbackGroup()
        self.declare_parameter('card_conf_thresh', 0.60)    # 감지 성공 임계
        self.declare_parameter('yolo_device', 'auto')       # 'auto' or 'cpu' or 'cuda:0'
        self.declare_parameter('debug_view', False)         # 디버그 창 켜기/끄기
        self._vis_lock = threading.Lock()
        self._vis_frame = None
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._frames_seen = 0
#################
                # __init__ 맨 아래 쪽, 구독 시작 근처 위/아래 아무데나 OK
        self._mode = None            # None | 'card' | 'gest'
        self._mode_end = 0.0

        # card 모드용 베스트 결과
        self._best_rank = 0
        self._best_conf = 0.0

        # gesture 모드용 관찰 플래그
        self._seen = {'nose': False, 'arms': False, 'ear': False}
        self.declare_parameter('gesture_score_on', 0.50)   # 행동 true 판정 임계
        # (선택) 오버레이 YOLO를 기본 끄고 필요 시만 켜고 싶다면:
        # self.declare_parameter('overlay_detect', False)
######################

        self.get_logger().info('CombinedDetectionNode 시작 (source: /camera/camera/color/image_raw)')
        self.bridge = CvBridge()
        self.srv_card = ActionServer(
            self, DetectCardOnce, '/dsr01/camera/detect_card_once',
            execute_callback=self.exec_cb_card,
            goal_callback=self.goal_cb_card,
            cancel_callback=self.cancel_cb_card,
            callback_group=self.cb_group) 
        
        self.srv_behave = ActionServer(
            self, ObserveGestures, '/dsr01/camera/observe_gestures',
            execute_callback=self.exec_cb_behave,
            goal_callback=self.goal_cb_behave,
            cancel_callback=self.cancel_cb_behave,
            callback_group=self.cb_group) 
        
        
        
    
    


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
            10,
            callback_group=self.cb_group
        )

    # 제스처 모드 시작 시 점수/카운터 초기화
    def _reset_gesture_state(self):
        self.ear_cnt = 0
        self.nose_cnt = 0
        self.arms_cnt = 0
        self.ear_score_ema = 0.0
        self.nose_score_ema = 0.0
        self.arms_score_ema = 0.0

    def _get_latest_frame(self):
        with self._frame_lock:
            if self._last_frame is None:
                return None, self._frames_seen
            return self._last_frame.copy(), self._frames_seen
    def goal_cb_behave(self, goal_request):
        # 이미 다른 모드 동작 중이면 거절
        if self._mode is not None and time.time() < self._mode_end:
            self.get_logger().warn("observe_gestures: busy")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT
    def cancel_cb_behave(self, goal_handle):
        return CancelResponse.ACCEPT
    def exec_cb_behave(self, goal_handle):
        """액션으로 제스처 관찰 트리거: goal.duration_s 동안 'gest' 모드로 동작"""
        # 1) duration_s 수신 (기본 10.0)
        try:
            duration_s = float(goal_handle.request.duration_s)
        except Exception:
            duration_s = 10.0

        # 2) 동시 실행 방지
        if self._mode is not None and time.time() < self._mode_end:
            goal_handle.abort()
            # Result 스펙: nose/arms/ear/ok/message
            return ObserveGestures.Result(nose=False, arms=False, ear=False, ok=False, message="busy")

        # 3) 모드 시작 및 상태 초기화
        self._mode = 'gest'
        t_start = time.time()
        self._mode_end = t_start + duration_s
        self._seen = {'nose': False, 'arms': False, 'ear': False}
        self._reset_gesture_state()
        thr_on = float(self.get_parameter('gesture_score_on').value)
        self.get_logger().info(f"GESTURE OBS 시작 ({duration_s:.1f}s, via action)")

        # 4) 피드백 인스턴스 (elapsed_s 필드만 존재)
        try:
            fb = ObserveGestures.Feedback()
        except Exception:
            fb = None

        # 5) 루프: duration_s 동안 계속 관찰 & 피드백
        while rclpy.ok() and time.time() < self._mode_end:
            if goal_handle.is_cancel_requested:
                self._mode = None
                goal_handle.canceled()
                return ObserveGestures.Result(nose=False, arms=False, ear=False, ok=False, message="취소됨")

            # image_callback이 EMA를 갱신 → 여기서 임계 넘었는지 스냅샷만 읽어 OR-집계
            if self.nose_score_ema >= thr_on: self._seen['nose'] = True
            if self.arms_score_ema >= thr_on: self._seen['arms'] = True
            if self.ear_score_ema  >= thr_on: self._seen['ear']  = True

            # 피드백: 경과 시간
            if fb is not None:
                try:
                    fb.elapsed_s = float(time.time() - t_start)
                    goal_handle.publish_feedback(fb)
                except Exception:
                    pass

            time.sleep(0.05)

        # 6) 종료 및 결과 반환
        self._mode = None
        nose, arms, ear = self._seen['nose'], self._seen['arms'], self._seen['ear']
        msg = f"obs_done nose={nose} arms={arms} ear={ear}"
        goal_handle.succeed()
        return ObserveGestures.Result(nose=bool(nose), arms=bool(arms), ear=bool(ear), ok=True, message=msg)



    def goal_cb_card(self, goal_request):
        # 이미 다른 모드 동작 중이면 거절
        if self._mode is not None and time.time() < self._mode_end:
            self.get_logger().warn("detect_card_once: busy")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT
    
    def cancel_cb_card(self, goal_handle):
        return CancelResponse.ACCEPT
    
    def exec_cb_card(self, goal_handle):
        """액션으로 카드 인식 트리거: goal.timeout_s 동안만 'card' 모드로 동작"""
        if self.yolo_model is None:
            goal_handle.abort()
            return DetectCardOnce.Result(rank=0, conf=0.0, ok=False, message="모델 없음")
        # NOTE: DetectCardOnce.Goal에 timeout_s가 있다고 가정. 없으면 기본 10초.
        try:
            timeout_s = float(goal_handle.request.timeout_s)
        except Exception:
            timeout_s = 10.0
        # 바쁨 방지(동시 실행 불가)
        if self._mode is not None and time.time() < self._mode_end:
            goal_handle.abort()
            return DetectCardOnce.Result(rank=0, conf=0.0, ok=False, message="busy")
        # 모드 시작
        self._mode = 'card'
        self._mode_end = time.time() + timeout_s
        self._best_rank = 0
        self._best_conf = 0.0
        self.get_logger().info(f"CARD TEST 시작 ({timeout_s:.1f}s, via action)")
        fb = None
        # 가능하면 피드백 타입 생성(패키지 정의에 따라 없으면 생략)
        # ✅ 성공 임계 (예: 0.7로 쓰려면 파라미터를 0.7로 세팅하세요)
        conf_thr = float(self.get_parameter('card_conf_thresh').value)
        try:
            fb = DetectCardOnce.Feedback()
        except Exception:
            fb = None
        # 루프: 모드 끝날 때까지 대기하며 현재 best를 피드백으로 전송
        while rclpy.ok() and time.time() < self._mode_end:
            if goal_handle.is_cancel_requested:
                self._mode = None
                goal_handle.canceled()
                return DetectCardOnce.Result(rank=0, conf=0.0, ok=False, message="취소됨")
            if fb is not None:
                try:
                    fb.conf_live = float(self._best_conf)
                    goal_handle.publish_feedback(fb)
                except Exception:
                    pass
            # ✅ 조기 성공: conf 임계 넘는 즉시 결과 반환
            if self._best_rank > 0 and self._best_conf >= conf_thr:
                ok = True
                msg = "감지 성공(조기종료)"
                self._mode = None
                goal_handle.succeed()
                return DetectCardOnce.Result(
                    rank=int(self._best_rank), conf=float(self._best_conf),
                    ok=ok, message=msg
            )
            time.sleep(0.05)
        # 종료 및 결과 판정(이미 image_callback에서 best_*가 갱신됨)
        conf_thr = float(self.get_parameter('card_conf_thresh').value)
        ok = (self._best_rank > 0 and self._best_conf >= conf_thr)
        self._mode = None
        goal_handle.succeed()
        msg = "감지 성공" if ok else "감지 실패"
        return DetectCardOnce.Result(rank=int(self._best_rank), conf=float(self._best_conf), ok=bool(ok), message=msg)


    # (재추론 금지) 만약 유지한다면 이렇게:
    def _parse_rank_from_results(self, results):
        if len(results.boxes) == 0:
            return 0, 0.0
        confs = results.boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        cls_id = int(results.boxes.cls[i])
        conf = float(confs[i])
        name = results.names.get(cls_id, "")
        m = re.search(r'(\d+)', name)
        rank = int(m.group(1)) if m else (1 if name.startswith('A') else 0)
        return (rank if 1 <= rank <= 10 else 0), conf


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
        with self._frame_lock:
            self._last_frame = frame.copy()
            self._frames_seen += 1


        h, w = frame.shape[:2]
        now = time.time()

        # ===== 모드 동작 =====
        if self._mode == 'card':
            if self.yolo_model is not None:
                dev = self.get_parameter('yolo_device').value
                yolo_results = self.yolo_model.predict(
                    source=frame,
                    conf=0.25,                 # 감지 필터 임계(박스 출력용). 0.25~0.4 권장
                    verbose=False,
                    device=None if dev == 'auto' else dev
                )[0]

                if len(yolo_results.boxes):
                    confs = yolo_results.boxes.conf.cpu().numpy()
                    i = int(np.argmax(confs))  # 가장 자신 있는 박스 사용
                    box = yolo_results.boxes[i]
                    cls_id = int(box.cls[0])
                    conf = float(confs[i])

                    # 클래스 이름에서 rank 파싱
                    name = yolo_results.names.get(cls_id, "")
                    m = re.search(r'(\d+)', name)
                    rank = int(m.group(1)) if m else (1 if name.startswith('A') else 0)

                    if 1 <= rank <= 10 and conf > self._best_conf:
                        self._best_conf, self._best_rank = conf, rank

                    # ──(선택) 모든 박스 오버레이──
                    for b in yolo_results.boxes:
                        cls = int(b.cls[0])
                        c   = float(b.conf[0]) * 100.0
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        label = f"{yolo_results.names[cls]} {c:.1f}%"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # 진행 오버레이
            remain = max(0.0, self._mode_end - now)
            cv2.putText(frame, f"[CARD TEST] {remain:0.1f}s  best: r{self._best_rank} @ {self._best_conf:0.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            if now >= self._mode_end:
                conf_thr = float(self.get_parameter('card_conf_thresh').value)
                ok = (self._best_rank > 0 and self._best_conf >= conf_thr)
                msg = "감지 성공" if ok else "감지 실패"
                self.get_logger().info(f"[CARD 10s] rank={self._best_rank}, conf={self._best_conf:.3f} -> {msg}")
                self._mode = None

        elif self._mode == 'gest':
            # MediaPipe 추론 (모드 중에만 수행)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_pose  = pose.process(rgb)
            res_face  = (face_mesh.process(rgb) if face_mesh is not None else None)
            res_hands = hands.process(rgb)

            if DRAW_LANDMARKS:
                if res_pose.pose_landmarks:
                    mp_draw.draw_landmarks(frame, res_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=mp_style.get_default_pose_landmarks_style())
                if face_mesh is not None and res_face and res_face.multi_face_landmarks:
                    for fl in res_face.multi_face_landmarks:
                        mp_draw.draw_landmarks(frame, fl, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=mp_draw.DrawingSpec(thickness=1, circle_radius=1))
                if res_hands and res_hands.multi_hand_landmarks:
                    for hl in res_hands.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            picked = "중립"
            if res_pose and res_pose.pose_landmarks:
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

                l_ear, r_ear = P(L_EAR), P(R_EAR)

                # (1) 귀 만지기
                ear_prox = 0.0
                if res_hands and res_hands.multi_hand_landmarks:
                    for hl in res_hands.multi_hand_landmarks:
                        for tip_idx in TIP_IDXS:
                            tip = to_xy(hl.landmark[tip_idx], w, h)
                            d_ear = min(nd(tip, l_ear), nd(tip, r_ear))
                            s = 1.0 - clamp(d_ear / EAR_THR_DIST, 0.0, 1.0)
                            if s > ear_prox: ear_prox = s
                            if SHOW_DEBUG:
                                cv2.circle(frame, (int(tip[0]),int(tip[1])), 5, (0,0,255), -1)
                self.ear_score_ema = ema(self.ear_score_ema, ear_prox, 0.25)
                ear_active = self.ear_score_ema > 0.5

                # (2) 코 만지기
                nose_prox = 0.0
                if res_hands and res_hands.multi_hand_landmarks:
                    for hl in res_hands.multi_hand_landmarks:
                        for tip_idx in TIP_IDXS:
                            tip = to_xy(hl.landmark[tip_idx], w, h)
                            d_nose = nd(tip, nose_p)
                            dx = abs(tip[0] - nose_p[0]) / shoulder_w
                            if d_nose < NOSE_THR_DIST and dx < NOSE_THR_DX:
                                s = 1.0 - clamp(d_nose / NOSE_THR_DIST, 0.0, 1.0)
                                if s > nose_prox: nose_prox = s
                self.nose_score_ema = ema(self.nose_score_ema, nose_prox, 0.25)
                nose_active = self.nose_score_ema > 0.5

                # (3) 팔짱
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
                arms_active = self.arms_score_ema > 0.5

                # 우선순위 및 보드
                if nose_active and (self.nose_score_ema >= self.ear_score_ema + 0.08):
                    picked = "코 만지기"
                elif ear_active and (self.ear_score_ema >= self.nose_score_ema + 0.08):
                    picked = "귀 만지기"
                elif arms_active:
                    picked = "팔짱"
                else:
                    picked = "중립"
                board = f"Ear:{self.ear_score_ema:0.2f} Nose:{self.nose_score_ema:0.2f} Arms:{self.arms_score_ema:0.2f}"
                cv2.putText(frame, board, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,50,50), 2)
                frame = draw_kor(frame, picked, (10, 58), 32, (255,200,0))

            # 진행/종료 오버레이
            remain = max(0.0, self._mode_end - now)
            flags = f"nose={self._seen['nose']} arms={self._seen['arms']} ear={self._seen['ear']}"
            cv2.putText(frame, f"[GESTURE OBS] {remain:0.1f}s  {flags}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            thr_on = float(self.get_parameter('gesture_score_on').value)
            if self.nose_score_ema >= thr_on: self._seen['nose'] = True
            if self.arms_score_ema >= thr_on: self._seen['arms'] = True
            if self.ear_score_ema  >= thr_on: self._seen['ear']  = True

            

        else:
            # Idle: 순수 영상 (액션으로만 제어)
            cv2.putText(frame, "Idle (action-controlled).",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)




        t_now = time.time()
        self.fps = 0.9*(1.0/max(1e-3, t_now - self.t_prev)) + 0.1*self.fps
        self.t_prev = t_now

        cv2.putText(frame, f"FPS:{self.fps:4.1f}", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,255,120), 2)
       
        with self._vis_lock:
            self._vis_frame = frame.copy()
        

def main():
    rclpy.init()
    node = CombinedDetectionNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # (선택) 윈도우를 미리 만들어두면 first frame 전에도 창이 뜹니다.
    cv2.namedWindow("Combined Detection (Pose + Card)", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)  # 콜백 펌핑

            # 메인 스레드에서 표시
            vis = None
            with node._vis_lock:
                if node._vis_frame is not None:
                    vis = node._vis_frame.copy()
            if vis is not None:
                cv2.imshow("Combined Detection (Pose + Card)", vis)

            # GUI 이벤트 펌핑은 반드시 메인 스레드에서!
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

