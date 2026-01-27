import cv2
import numpy as np
from collections import deque


def estimate_rotation(img1, img2, focal_length=800, principal_point=(320, 240)):
    # 1. 初始化 ORB 检测器
    orb = cv2.ORB.create(nfeatures=2000)
    
    # 2. 检测特征点并计算描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # 3. 特征匹配 (使用暴力匹配 BFMatcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # 提取匹配点的坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # 4. 计算本质矩阵 (Essential Matrix)
    # 假设相机内参已经校准
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=focal_length, pp=principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # 5. 从本质矩阵中恢复旋转和平移
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=focal_length, pp=principal_point)
    
    # 6. 将旋转矩阵转换为欧拉角 (弧度)
    # 对于简单的地平面运动，通常关注绕 Y 轴（OpenCV坐标系）的旋转
    yaw_angle = np.arctan2(R[0, 2], R[2, 2])
    
    return np.degrees(yaw_angle)


class TurnDetector:
    def __init__(self, window_size: int = 5, threshold_angle: float = 45):
        self.window_size = window_size
        self.threshold_angle = threshold_angle
        self.frames = deque(maxlen=2)  # 只存最近两帧用于计算
        self.angle_history = deque(maxlen=window_size) # 存储角度变化序列
        self.frame_history = deque(maxlen=window_size) # 存储最近window_size帧图像

    def process_frame(self, frame: np.ndarray):
        self.frame_history.append(frame)
        # 转换为灰度图提高速度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frames.append(gray)
        
        if len(self.frames) < 2:
            return False, 0.0

        # 计算相邻帧的旋转 (使用上文提到的 estimate_rotation 函数)
        delta_theta = estimate_rotation(self.frames[0], self.frames[1])
        self.angle_history.append(delta_theta)
        
        # 计算窗口内的累积旋转
        cumulative_turn = sum(self.angle_history)
        
        # 判定逻辑：如果窗口内累积旋转超过阈值，认为正在发生大幅度转向
        is_turning = abs(cumulative_turn) > self.threshold_angle
        print(f"Delta theta: {delta_theta:.2f}, Cumulative turn: {cumulative_turn:.2f}, Is turning: {is_turning}")

        return is_turning
    
    def reset(self):
        self.frames.clear()
        self.angle_history.clear()


if __name__ == "__main__":
    import os
    import PIL.Image as Image

    turn_detector = TurnDetector(window_size=5, threshold_angle=45)

    test_data_dir = 'assets/realworld_sample_data1'
    for frame_name in sorted(os.listdir(test_data_dir)):
        frame_path = os.path.join(test_data_dir, frame_name)
        print(f"Processing frame: {frame_path}")

        frame = Image.open(frame_path)
        frame = np.array(frame)
        is_turning, cumulative_turn = turn_detector.process_frame(frame)

        if is_turning:
            print(f"Turning detected! Over the latest {turn_detector.window_size} frames, cumulative turn angle: {cumulative_turn:.2f} degrees")