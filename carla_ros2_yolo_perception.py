#!/usr/bin/env python
"""
Client-side bounding boxes with basic car controls, YOLO detection, and ROS2 publishing.

Controls:
    W : throttle
    S : brake
    A/D : steer
    Space : hand-brake
    ESC : quit
"""

import sys, time, weakref, random, cv2
import numpy as np
import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_a, K_d, K_w, K_s, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_m

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    import carla
except ImportError:
    print("CARLA Python API 未安装，请先安装")
    raise

from ultralytics import YOLO

# ==========================
# Config
# ==========================
VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90
MODEL_PATH = "/home/ubuntu22/Desktop/models/yolo11n.pt"

# ==========================
# Main Client
# ==========================
class BasicSynchronousClient:
    def __init__(self):
        # ROS2
        rclpy.init()
        self.ros_node = rclpy.create_node("carla_node")
        self.pub = self.ros_node.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()

        # CARLA
        self.client = None
        self.world = None
        self.car = None
        self.camera = None
        self.depth_camera = None
        self.display = None
        self.image = None
        self.depth_image = None
        self.capture = True
        self.depth_capture = True
        self.counter = 0
        self.pose = []
        self.log = False

        # YOLO
        self.model = YOLO(MODEL_PATH)
        self.classNames = self.model.names
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classNames), 3), dtype='uint8')

    # --------------------------
    # Blueprints
    # --------------------------
    def camera_blueprint(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        bp.set_attribute('fov', str(VIEW_FOV))
        return bp

    def depth_camera_blueprint(self):
        # 前视角摄像头
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        bp.set_attribute('fov', str(VIEW_FOV))
        return bp

    # --------------------------
    # Synchronous Mode
    # --------------------------
    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    # --------------------------
    # Spawn car
    # --------------------------
    def setup_car(self):
        car_bp = self.world.get_blueprint_library().filter('model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for location in spawn_points:
            try:
                self.car = self.world.spawn_actor(car_bp, location)
                break
            except RuntimeError:
                continue
        if self.car is None:
            raise RuntimeError("Failed to spawn car")

    # --------------------------
    # Setup Cameras
    # --------------------------
    def setup_camera(self):
        transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-5))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))
        self.camera.calibration = self.get_calibration()

    def setup_depth_camera(self):
        transform = carla.Transform(carla.Location(x=1.5, z=2.3), carla.Rotation(pitch=-5))
        self.depth_camera = self.world.spawn_actor(self.depth_camera_blueprint(), transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.depth_camera.listen(lambda img: weak_self().set_depth_image(weak_self, img))
        self.depth_camera.calibration = self.get_calibration()

    def get_calibration(self):
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        return calibration

    # --------------------------
    # Controls
    # --------------------------
    def control(self, car):
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True
        control = car.get_control()
        control.throttle = 0
        if keys[K_w] or keys[K_UP]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s] or keys[K_DOWN]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a] or keys[K_LEFT]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d] or keys[K_RIGHT]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]
        if keys[K_m]:
            self.log = not self.log
            if not self.log:
                np.savetxt('log/pose.txt', self.pose)
        car.apply_control(control)
        return False

    # --------------------------
    # Image callbacks
    # --------------------------
    @staticmethod
    def set_image(weak_self, img):
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    @staticmethod
    def set_depth_image(weak_self, img):
        self = weak_self()
        if self.depth_capture:
            self.depth_image = img
            self.depth_capture = False

    # --------------------------
    # Rendering + YOLO + ROS2 publish
    # --------------------------
    def depth_render(self):
        if self.depth_image is None:
            return

        img = np.array(self.depth_image.raw_data).reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))[:, :, :3]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 发布到 ROS2
        ros_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')
        self.pub.publish(ros_msg)

        # YOLO 检测
        results = self.model(img_bgr)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            color = [int(c) for c in self.colors[cls]]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_bgr, f"{self.classNames[cls]} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Front View Camera", img_bgr)

    # --------------------------
    # Logging
    # --------------------------
    def log_data(self):
        if self.log and self.depth_image:
            name = 'log/' + str(self.counter) + '.png'
            self.depth_image.save_to_disk(name)
            t = self.car.get_transform()
            self.pose.append((
                self.counter, t.location.x, t.location.y, t.location.z,
                t.rotation.roll, t.rotation.pitch, t.rotation.yaw
            ))
            self.counter += 1

    # --------------------------
    # Main Loop
    # --------------------------
    def game_loop(self):
        try:
            pygame.init()
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()
            self.setup_depth_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            clock = pygame.time.Clock()
            self.set_synchronous_mode(True)

            while True:
                self.world.tick()
                self.capture = True
                self.depth_capture = True
                clock.tick_busy_loop(30)

                self.render()
                pygame.display.flip()
                pygame.event.pump()
                self.depth_render()
                self.log_data()

                rclpy.spin_once(self.ros_node, timeout_sec=0)
                cv2.waitKey(1)
                if self.control(self.car):
                    break

        finally:
            self.cleanup()

    def render(self):
        if self.image is None:
            return
        array = np.frombuffer(self.image.raw_data, dtype=np.uint8).reshape((self.image.height, self.image.width, 4))
        array = array[:, :, :3][:, :, ::-1]
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.display.blit(surface, (0, 0))

    # --------------------------
    # Cleanup
    # --------------------------
    def cleanup(self):
        self.set_synchronous_mode(False)
        if self.camera: self.camera.destroy()
        if self.depth_camera: self.depth_camera.destroy()
        if self.car: self.car.destroy()
        self.ros_node.destroy_node()
        rclpy.shutdown()
        pygame.quit()
        cv2.destroyAllWindows()


# ==========================
# Main
# ==========================
def main():
    client = BasicSynchronousClient()
    client.game_loop()


if __name__ == "__main__":
    main()

