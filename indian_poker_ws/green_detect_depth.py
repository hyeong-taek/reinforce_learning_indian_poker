#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as RosImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError


from geometry_msgs.msg import PointStamped



class GreenDepthGrabber(Node):
    def __init__(self):
        super().__init__("green_depth_grabber")
        # self.image_node = realsense.ImgNode()
        # self.intrinsics = self.image_node.get_camera_intrinsic()
        self.intrinsics = None

        # 퍼블리셔: depth 값과 픽셀 좌표(+depth)
        self.color_topic = "/camera/camera/color/image_raw"
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.info_topic  = "/camera/camera/color/camera_info"

        self.sub_color = self.create_subscription(RosImage, self.color_topic, self._color_cb, 10)
        self.sub_depth = self.create_subscription(RosImage, self.depth_topic, self._depth_cb, 10)
        self.sub_info  = self.create_subscription(CameraInfo, self.info_topic,  self._cinfo_cb, 10)
        self.green_xyz_publisher = self.create_publisher(PointStamped, 'green_detect_xyz', 10)
        # CvBridge
        self.bridge = CvBridge()

        # 초록색(H,S,V) 범위 및 최소 면적
        self.lower_green = np.array([45, 50, 40], dtype=np.uint8)
        self.upper_green = np.array([95, 255, 255], dtype=np.uint8)
        self.min_area = 100

        # 마지막 프레임 캐시
        self.last_bgr = None           # np.ndarray, uint8, BGR
        self.last_depth = None         # np.ndarray, uint16 (mm가 일반적)
        self.last_center = None

        # OpenCV 윈도우
        cv2.namedWindow("green_view", cv2.WINDOW_NORMAL)
        cv2.namedWindow("green_mask", cv2.WINDOW_NORMAL)

        # 구독자: RGB / Depth (aligned-to-color)
        self.color_topic = "/camera/camera/color/image_raw"
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"

        self.sub_color = self.create_subscription(
            RosImage, self.color_topic, self._color_cb, 10
        )
        self.sub_depth = self.create_subscription(
            RosImage, self.depth_topic, self._depth_cb, 10
        )

        # 주기 타이머: 30Hz로 화면 갱신 및 키 처리
        self.timer = self.create_timer(1.0 / 30.0, self._loop)

        self.get_logger().info(
            "GreenDepthGrabber started.\n"
            f"- color: {self.color_topic}\n"
            f"- depth: {self.depth_topic}\n"
            "Press [Enter] to sample depth at detected green center."
        )

    # ─────────────────────────────────────────────────────────────
    # 콜백: color / depth
    # ─────────────────────────────────────────────────────────────
    def _cinfo_cb(self, msg: CameraInfo):
        # CameraInfo.K = [fx, 0, cx,  0, fy, cy,  0, 0, 1]
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        self.intrinsics = {"fx": fx, "fy": fy, "ppx": cx, "ppy": cy}

    def _color_cb(self, msg: RosImage):
        try:
            # OpenCV BGR8
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_bgr = bgr
        except CvBridgeError as e:
            self.get_logger().warning(f"CvBridge color 변환 실패: {e}")

    def _depth_cb(self, msg: RosImage):
        try:
            # depth는 'passthrough'로 받아 dtype/스케일 보존 (대개 uint16, mm 단위)
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.last_depth = depth
        except CvBridgeError as e:
            self.get_logger().warning(f"CvBridge depth 변환 실패: {e}")

    # ─────────────────────────────────────────────────────────────
    # 유틸: 프레임 유효성 검사
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _valid_frame(frame):
        if frame is None or not hasattr(frame, "shape"):
            return False
        h, w = frame.shape[:2]
        return (w > 0 and h > 0)

    # ─────────────────────────────────────────────────────────────
    # 초록색 중심 찾기 (가장 큰 컨투어)
    # ─────────────────────────────────────────────────────────────
    def _find_green_center(self, bgr):
        blur = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if area > best_area:
                best = c
                best_area = area

        if best is None:
            return None, mask

        x, y, w, h = cv2.boundingRect(best)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        return (cx, cy), mask

    # ─────────────────────────────────────────────────────────────
    # 메인 루프: 프레임 검사/표시 + 키 입력 처리
    # ─────────────────────────────────────────────────────────────
    def _loop(self):
        color = self.last_bgr
        depth = self.last_depth

        if not self._valid_frame(color):
            # color가 들어오기 전이면 대기
            return
        if depth is None:
            # depth 아직 없음 → RGB만 표시
            vis = color.copy()
            cv2.putText(
                vis,
                "Waiting for aligned depth...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            try:
                cv2.imshow("green_view", vis)
            except cv2.error:
                pass
            cv2.waitKey(1)
            return

        # 해상도 불일치시 경고(Aligned라면 보통 동일)
        if hasattr(depth, "shape") and color.shape[:2] != depth.shape[:2]:
            self.get_logger().warning(
                f"Size mismatch (color={color.shape[:2]}, depth={depth.shape[:2]})."
            )

        # 초록 탐지/시각화
        center, mask = self._find_green_center(color)
        vis = color.copy()
        if center is not None:
            cx, cy = center
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"GREEN center: ({cx},{cy})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                vis,
                "No GREEN detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 화면 표시
        try:
            cv2.imshow("green_view", vis)
            if self._valid_frame(mask):
                cv2.imshow("green_mask", mask)
        except cv2.error as e:
            self.get_logger().error(f"imshow failed: {e}")
            return

        # 키 입력 처리: Enter(13 또는 10) → 현재 초록 중심의 depth 읽기
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):  # Enter
            self._grab_depth_at_center(center)

        if key == 27:  # ESC → 노드 종료
            self.get_logger().info("ESC pressed. Shutting down.")
            rclpy.shutdown()

    # ─────────────────────────────────────────────────────────────
    # Enter 눌렀을 때 depth 읽기/로그/퍼블리시
    # ─────────────────────────────────────────────────────────────
    def _grab_depth_at_center(self, center):
        if center is None:
            self.get_logger().warning("No GREEN center to sample depth.")
            return
        if self.last_depth is None:
            self.get_logger().warning("Depth frame not available.")
            return
        if self.intrinsics is None:
            self.get_logger().warning("Camera intrinsics not received yet (/camera_info).")
            return

        cx, cy = center
        h, w = self.last_depth.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            self.get_logger().warning(f"Center out of range: ({cx},{cy}) not in 0..{w-1}, 0..{h-1}")
            return

        # depth 단위 처리: encoding 확인하여 m로 변환
        depth_pix = self.last_depth[cy, cx]
        depth_m = self._depth_to_meters(depth_pix)

        self.get_logger().info(f"[ENTER] depth[{cy},{cx}] = {depth_m:.3f} m")

        cam_xyz = self.get_camera_pos(cx, cy, depth_m, self.intrinsics)
        self.get_logger().info(f"camera XYZ (m): {cam_xyz}")

        point_msg = PointStamped()
        point_msg.point.x = cam_xyz[0]
        point_msg.point.y = cam_xyz[1]
        point_msg.point.z = cam_xyz[2]
        point_msg.header.frame_id = "green_detect_xyz"
        self.green_xyz_publisher.publish(point_msg)
        
        # (주의) transform_to_base는 아래 주의사항 참고
        # base_xyz = self.transform_to_base(cam_xyz)

        
    def get_camera_pos(self, u, v, z_m, intrinsics):
        # u,v: 픽셀좌표 / z_m: 미터 / fx,fy,ppx,ppy: 픽셀 단위
        x = (u - intrinsics["ppx"]) * z_m / intrinsics["fx"]
        y = (v - intrinsics["ppy"]) * z_m / intrinsics["fy"]
        return (x, y, z_m)
    def _depth_to_meters(self, depth_pix):
        """
        depth 이미지 encoding에 따라 단위를 m로 변환
        - 16UC1: mm가 일반적 → m로 변환
        - 32FC1: 이미 m 단위
        """
        # 최근에 받은 depth 메시지의 encoding을 저장해두는 방식을 권장.
        # 간단히 dtype 기반으로 추정(추측입니다): uint16 → mm, float32 → m
        if self.last_depth.dtype == np.uint16:  # 확률 높음
            return float(depth_pix) / 1
        elif self.last_depth.dtype == np.float32:
            return float(depth_pix)
        else:
            # 확실하지 않음: 안전하게 mm로 가정
            return float(depth_pix) / 1
    def transform_to_base(self, camera_coords):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate

        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        timer = time.time()

        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]


def main():
    rclpy.init()
    node = GreenDepthGrabber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
