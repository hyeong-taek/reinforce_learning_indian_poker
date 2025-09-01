#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image as RosImage, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError

from poker_interfaces.action import DetectChip
from typing import Optional, Tuple


class GreenDepthGrabber(Node):
    """
    - IDLE: 컬러 raw + 안내문
    - Enter: 수동 DETECTING 10초 (녹색 검출 + 중심점 표시, 마지막 유효 xyz를 퍼블리시)
    - 액션(DetectChip): 액션 시간 동안만 'ACTION DETECTING'으로 동작, 마지막 유효 xyz를 결과로 반환
    """

    def __init__(self):
        super().__init__("green_depth_grabber")

        # 콜백 그룹 (액션/이미지 병렬)
        self.cb_img = ReentrantCallbackGroup()
        self.cb_act = ReentrantCallbackGroup()

        # 카메라 토픽
        self.color_topic = "/camera/camera/color/image_raw"
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.info_topic  = "/camera/camera/color/camera_info"

        # 구독/퍼블리셔
        self.sub_color = self.create_subscription(
            RosImage, self.color_topic, self._color_cb, 10, callback_group=self.cb_img
        )
        self.sub_depth = self.create_subscription(
            RosImage, self.depth_topic, self._depth_cb, 10, callback_group=self.cb_img
        )
        self.sub_info  = self.create_subscription(
            CameraInfo, self.info_topic, self._cinfo_cb, 10, callback_group=self.cb_img
        )
        self.pub_point = self.create_publisher(PointStamped, 'green_detect_xyz', 10)

        # CvBridge
        self.bridge = CvBridge()

        # 초록색 HSV 범위/최소면적
        self.lower_green = np.array([45, 50, 40], dtype=np.uint8)
        self.upper_green = np.array([95, 255, 255], dtype=np.uint8)
        self.min_area = 100

        # 프레임/상태
        self.last_bgr: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.intrinsics = None  # dict: fx,fy,ppx,ppy
        self.camera_frame_id = None

        self.last_center: Optional[Tuple[int,int]] = None
        self.last_xyz: Optional[Tuple[float,float,float]] = None

        # 수동 detecting (Enter)
        self.detect_active = False
        self.detect_t_end = 0.0

        # 액션 상태(비블로킹)
        self.action_in_progress = False
        self._action_mode = False
        self._action_deadline = 0.0
        self._action_last_xyz: Optional[Tuple[float,float,float]] = None

        # 시각화 프레임 버퍼 (메인스레드에서만 imshow)
        self._vis_lock = threading.Lock()
        self._vis_frame: Optional[np.ndarray] = None

        # 30Hz 주기 처리(검출 + 오버레이 생성)
        self.timer = self.create_timer(1.0/30.0, self._loop, callback_group=self.cb_img)

        # 액션 서버
        self.detect_srv = ActionServer(
            self,
            DetectChip,
            'chip_detect_once',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_act
        )

        self.get_logger().info(
            "GreenDepthGrabber started.\n"
            f"- color: {self.color_topic}\n"
            f"- depth: {self.depth_topic}\n"
            "IDLE: raw 표시 | Enter: 10s detecting | ESC: 종료"
        )

    # ───────────────────────── Camera callbacks ─────────────────────────
    def _cinfo_cb(self, msg: CameraInfo):
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        self.intrinsics = {"fx": fx, "fy": fy, "ppx": cx, "ppy": cy}
        self.camera_frame_id = msg.header.frame_id or "camera_frame"

    def _color_cb(self, msg: RosImage):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_bgr = bgr
        except CvBridgeError as e:
            self.get_logger().warning(f"CvBridge color 변환 실패: {e}")

    def _depth_cb(self, msg: RosImage):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.last_depth = depth
        except CvBridgeError as e:
            self.get_logger().warning(f"CvBridge depth 변환 실패: {e}")

    # ───────────────────────── Utilities ─────────────────────────
    @staticmethod
    def _valid_frame(frame):
        if frame is None or not hasattr(frame, "shape"):
            return False
        h, w = frame.shape[:2]
        return (w > 0 and h > 0)

    def _find_green_center(self, bgr):
        blur = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if area > best_area:
                best = c
                best_area = area

        if best is None:
            return None
        x, y, w, h = cv2.boundingRect(best)
        return (int(x + w/2), int(y + h/2))

    def _depth_to_meters(self, depth_pix):
        # 프로젝트 환경에 맞춰 필요시 조정
        if isinstance(self.last_depth, np.ndarray):
            if self.last_depth.dtype == np.uint16:
                return float(depth_pix) / 1.0  # 환경에 따라 1000.0으로 변경
            elif self.last_depth.dtype == np.float32:
                return float(depth_pix)
        return float(depth_pix) / 1.0

    def get_camera_pos(self, u, v, z_m, intrinsics):
        x = (u - intrinsics["ppx"]) * z_m / intrinsics["fx"]
        y = (v - intrinsics["ppy"]) * z_m / intrinsics["fy"]
        return (x, y, z_m)

    def _snapshot_xyz(self, center):
        if center is None:
            return False, None, "center 없음"
        if self.last_depth is None:
            return False, None, "depth 미수신"
        if self.intrinsics is None:
            return False, None, "intrinsics 미수신"

        cx, cy = center
        h, w = self.last_depth.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return False, None, f"center 범위 밖 ({cx},{cy})"

        depth_pix = self.last_depth[cy, cx]
        depth_m = self._depth_to_meters(depth_pix)
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            return False, None, f"유효하지 않은 depth: {depth_pix}"

        x, y, z = self.get_camera_pos(cx, cy, depth_m, self.intrinsics)

        # 퍼블리시
        try:
            msg = PointStamped()
            msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame_id or "camera_frame"
            self.pub_point.publish(msg)
        except Exception:
            pass

        return True, (x, y, z), "ok"

    # ───────────────────────── Main periodic loop ─────────────────────────
    def _loop(self):
        bgr = self.last_bgr
        if not self._valid_frame(bgr):
            return

        now = time.time()
        vis = bgr.copy()

        # 수동 DETECTING 모드 자동 종료 처리
        if self.detect_active and now >= self.detect_t_end:
            self.detect_active = False
            if self.last_xyz is not None:
                x, y, z = self.last_xyz
                self.get_logger().info(f"[FINAL] last xyz(m)=({x:.3f},{y:.3f},{z:.3f})")
                try:
                    msg = PointStamped()
                    msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = self.camera_frame_id or "camera_frame"
                    self.pub_point.publish(msg)
                except Exception:
                    pass
            self.last_xyz = None
            self.last_center = None
            self.get_logger().info("DETECTING 종료 → IDLE")

        # 어떤 모드에서 검출할까?
        active = self.detect_active or self._action_mode
        head = "IDLE"
        remain_txt = ""

        if active:
            center = self._find_green_center(bgr)
            self.last_center = center

            if center is not None:
                cx, cy = center
                cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)

                # 수동 detecting 중엔 last_xyz 계속 갱신
                if self.detect_active:
                    ok, xyz, _ = self._snapshot_xyz(center)
                    if ok:
                        self.last_xyz = xyz

                # 액션 모드 중엔 _action_last_xyz 갱신
                if self._action_mode:
                    ok, xyz, _ = self._snapshot_xyz(center)
                    if ok:
                        self._action_last_xyz = xyz

            # 헤더/잔여시간
            if self._action_mode:
                head = "ACTION DETECTING"
                remain = max(0.0, self._action_deadline - now)
                remain_txt = f"{remain:0.1f}s left"
            else:
                head = "DETECTING"
                remain = max(0.0, self.detect_t_end - now)
                remain_txt = f"{remain:0.1f}s left | Enter: sample xyz"

        else:
            # IDLE 안내
            cv2.putText(
                vis, "IDLE | Enter: start 10s detecting | ESC: quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA
            )

        # 상단 상태 텍스트
        if active:
            cv2.putText(
                vis, head, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )
            if remain_txt:
                cv2.putText(
                    vis, remain_txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )

        # 메인스레드에서 표시할 프레임 버퍼에 저장
        with self._vis_lock:
            self._vis_frame = vis

    # ───────────────────────── Action server ─────────────────────────
    def goal_callback(self, goal_request: DetectChip.Goal):
        if self.action_in_progress:
            self.get_logger().warning("액션 진행 중: 새 goal 거절")
            return GoalResponse.REJECT
        if goal_request.timeout_s <= 0.0:
            self.get_logger().warning("timeout_s <= 0: goal 거절")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("액션 취소 요청 수락")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """
        비블로킹 액션:
        - 여기서는 상태 플래그만 세팅하고 '마지막 유효 스냅샷'을 기다렸다가 반환.
        - 실제 프레임 처리/xyz 갱신은 _loop()에서 계속 수행.
        """
        timeout_s = max(0.1, float(goal_handle.request.timeout_s))
        self._action_deadline = time.time() + timeout_s

        # 액션 상태 on
        self.action_in_progress = True
        self._action_mode = True
        self._action_last_xyz = None
        self.detect_active = True               # 시각화/검출 on
        self.detect_t_end = self._action_deadline

        self.get_logger().info(f"[ACTION] DETECTING 시작: {timeout_s:.2f}s (non-blocking)")

        result = DetectChip.Result()
        feedback = DetectChip.Feedback()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info("[ACTION] cancel requested")
                break

            now = time.time()
            if now >= self._action_deadline:
                break

            # 피드백: 유효 xyz 존재 여부(간단)
            try:
                feedback.conf_live = 1.0 if (self._action_last_xyz is not None) else 0.0
                goal_handle.publish_feedback(feedback)
            except Exception:
                pass

            time.sleep(0.03)

        # 종료 플래그 정리
        self.action_in_progress = False
        self._action_mode = False
        self.detect_active = False
        self.last_center = None

        # 결과
        if self._action_last_xyz is not None:
            x, y, z = self._action_last_xyz
            result.points = [float(x), float(y), float(z)]
            result.ok = True
            result.message = "latest xyz from last frame"
            try: goal_handle.succeed()
            except Exception: pass
            self.get_logger().info(f"[ACTION] 반환: ({x:.3f},{y:.3f},{z:.3f})")
        else:
            result.points = []
            result.ok = False
            if goal_handle.is_cancel_requested:
                result.message = "canceled: no valid detection"
                try: goal_handle.canceled()
                except Exception: pass
            else:
                result.message = "timeout: no valid detection"
                try: goal_handle.abort()
                except Exception: pass
            self.get_logger().warning(f"[ACTION] 실패: {result.message}")

        return result


def main():
    rclpy.init()
    node = GreenDepthGrabber()

    # 멀티스레드 실행기: 액션/카메라/타이머 병렬
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    # 메인 스레드 GUI 루프
    cv2.namedWindow("green_view", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)  # 콜백 펌핑

            # 최신 시각화 프레임 표시
            vis = None
            with node._vis_lock:
                if node._vis_frame is not None:
                    vis = node._vis_frame.copy()
            if vis is not None:
                cv2.imshow("green_view", vis)

            # 키 처리(메인 스레드)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):  # Enter
                if not node.action_in_progress and not node.detect_active:
                    node.detect_active = True
                    node.detect_t_end = time.time() + 10.0
                    node.get_logger().info("DETECTING 시작 (10s) — 초록 검출/표시 활성화")
            elif key == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
