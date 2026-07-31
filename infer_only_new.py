import sys
import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# Add project path
project_root = Path('../../')
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

class Args:
    def __init__(self):
        self.device = "cuda:0"
        self.model_path = "checkpoints/InternVLA-N1-DualVLN"
        self.resize_w = 384
        self.resize_h = 384
        self.num_history = 8
        self.camera_intrinsic = np.array([
            [386.5, 0.0, 328.9, 0.0],
            [0.0, 386.5, 244.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.plan_step_gap = 4

args = Args()
print(f"Model path: {args.model_path}")
print(f"Device: {args.device}")
print(f"Image size: {args.resize_w}x{args.resize_h}")
print(f"History frames: {args.num_history}")

print("Loading model...")
agent = InternVLAN1AsyncAgent(args)

# Warm up model
print("Warming up model...")
dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
dummy_depth = np.zeros((480, 640), dtype=np.float32)
dummy_pose = np.eye(4)
agent.reset()
agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=args.camera_intrinsic)
print("Model loaded successfully!")


# Configure data directory (single scene per folder)
scene_dir = './assets/realworld_sample_data1'

# Check if instruction file exists
instruction_path = os.path.join(scene_dir, 'instruction.txt')
if not os.path.exists(instruction_path):
    print(f"Error: instruction.txt not found in {scene_dir}")
else:
    print(f"Scene directory: {scene_dir}")
    
    # Read instruction
    with open(instruction_path, 'r') as f:
        instruction = f.read().strip()
    print(f"Instruction: {instruction}")
    
    # Get all debug_raw images
    rgb_paths = sorted(glob.glob(os.path.join(scene_dir, 'debug_raw_*.jpg')))
    print(f"\nFound {len(rgb_paths)} images")
    # Show first few image names
    print("\nFirst 5 images:")
    for i, path in enumerate(rgb_paths[:5]):
        print(f"  {i+1}. {os.path.basename(path)}")

from PIL import Image, ImageDraw, ImageFont
import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def annotate_image(idx, image, llm_output, trajectory, pixel_goal, output_dir):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)

    text_content = []
    text_content.append(f"Frame    Id  : {idx}")
    text_content.append(f"Actions      : {llm_output}")

    max_width = 0
    total_height = 0
    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = 26
        max_width = max(max_width, text_width)
        total_height += text_height

    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding

    draw.rectangle(
        [box_x, box_y, box_x + box_width, box_y + box_height], fill="black"
    )

    text_color = "white"
    y_position = box_y + padding

    for line in text_content:
        draw.text(
            (box_x + padding, y_position), line, fill=text_color, font=font
        )
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = 26
        y_position += text_height
    image = np.array(image)

    # 绘制右上角无框透明轨迹图
    if trajectory is not None and len(trajectory) > 0:
        img_height, img_width = image.shape[:2]

        # 1. 设置轨迹图尺寸（保持放大效果，如 350px）
        window_size = 350
        window_margin = 10
        window_x = img_width - window_size - window_margin
        window_y = window_margin

        # 提取轨迹点
        traj_points = []
        for point in trajectory:
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                traj_points.append([float(point[0]), float(point[1])])

        if len(traj_points) > 0:
            traj_array = np.array(traj_points)
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]

            # 2. 创建完全透明背景的 Figure 和 Axes
            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=100)
            fig.patch.set_alpha(0.0)  # 图表背景全透明
            ax.patch.set_alpha(0.0)  # 坐标轴区域全透明

            # 3. 隐藏四周的黑色/白色边框线 (Spines)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # 4. 绘制轨迹与标记（高亮对比色）
            ax.plot(
                y_coords,
                x_coords,
                color="red",
                linewidth=3.5,
                label="Trajectory",
            )  # 鲜艳蓝色轨迹
            ax.plot(
                y_coords[0],
                x_coords[0],
                "go",
                markersize=9,
                label="Start",
                markeredgecolor="white",
            )
            ax.plot(
                y_coords[-1],
                x_coords[-1],
                "ro",
                markersize=9,
                label="End",
                color="blue",
                markeredgecolor="white",
            )
            ax.plot(
                0,
                0,
                "k+",
                markersize=13,
                markeredgewidth=2.5,
                label="Origin",
            )

            # 5. 设置坐标轴与字体大小
            ax.set_xlabel("Y (left +)", fontsize=11, fontweight="bold", color="white")
            ax.set_ylabel("X (up +)", fontsize=11, fontweight="bold", color="white")
            ax.invert_xaxis()
            ax.tick_params(labelsize=10, colors="white")  # 坐标轴刻度颜色为白色

            # 虚线网格，透明度调低，不遮挡背景
            ax.grid(True, linestyle="--", alpha=1, linewidth=1.2)

            ax.set_aspect("equal", adjustable="box")

            # 图例背景设为半透明白色，防止覆盖文字
            ax.legend(
                fontsize=9, loc="upper right", framealpha=0.6, facecolor="white"
            )

            plt.tight_layout(pad=0.2)

            # 6. 导出 RGBA 四通道（含 Alpha 透明度）图像
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            rgba_buffer = canvas.buffer_rgba()
            plot_rgba = np.asarray(rgba_buffer)
            plt.close(fig)

            # 7. 调整大小
            plot_rgba = cv2.resize(plot_rgba, (window_size, window_size))

            # 8. Alpha 混合叠加到原图
            plot_rgb = plot_rgba[:, :, :3]
            alpha = (plot_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]

            # 获取原图右上角 ROI 区域
            roi = image[
                window_y : window_y + window_size,
                window_x : window_x + window_size,
            ]

            # 融合 RGB 画面与透明轨迹
            blended = (plot_rgb * alpha + roi * (1.0 - alpha)).astype(np.uint8)
            image[
                window_y : window_y + window_size,
                window_x : window_x + window_size,
            ] = blended

    if pixel_goal is not None:
        cv2.circle(image, (pixel_goal[1], pixel_goal[0]), 5, (255, 0, 0), -1)

    image = Image.fromarray(image).convert("RGB")
    image.save(f"{output_dir}/rgb_{idx}_annotated.png")

    return np.array(image)


# Reset agent
agent.reset()
print(f"{'='*80}")
print(f"Processing scene: {os.path.basename(scene_dir)}")
print(f"Instruction: '{instruction}'")
print(f"Total images: {len(rgb_paths)}")
print(f"{'='*80}\n")

action_seq = []
look_down = False

save_dir = './test_data/'
os.makedirs(save_dir, exist_ok=True)
# Process each image
for i, rgb_path in enumerate(rgb_paths):
    # Check if this is a look_down image
    look_down = ('look_down' in rgb_path)
    
    # Extract image ID from filename (e.g., debug_raw_0003.jpg -> 0003)
    basename = os.path.basename(rgb_path)
    if look_down:
        # e.g., debug_raw_0010_look_down.jpg -> 0010
        image_id = basename.replace('debug_raw_', '').replace('_look_down.jpg', '')
    else:
        # e.g., debug_raw_0003.jpg -> 0003
        image_id = basename.replace('debug_raw_', '').replace('.jpg', '')
        
    # Read RGB image
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    
    # Create dummy depth image (not available in test data)
    # !Note You must full in depth to model
    depth = 10 * np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
    
    # Create dummy camera pose
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Run model or just save image
    # print(f"[{i+1}/{len(rgb_paths)}] Running model inference: {os.path.basename(rgb_path)}")
    with torch.no_grad():   
        dual_sys_output = agent.step(
            rgb, 
            depth, 
            camera_pose, 
            instruction, 
            intrinsic=args.camera_intrinsic,
            look_down=look_down
        )
    
    # Print output results
    if dual_sys_output.output_action is not None and dual_sys_output.output_action != []:
        print(f"  Output action: {dual_sys_output.output_action}")
        # action_seq.extend(s2_output.output_action)
    else:
        
        print(f"output_trajectory: {dual_sys_output.output_trajectory.tolist()}")
        if dual_sys_output.output_pixel is not None:
            print(f"output_pixel: {dual_sys_output.output_pixel}")
            annotate_image(image_id, rgb, 'traj', dual_sys_output.output_trajectory.tolist(), dual_sys_output.output_pixel, save_dir)


print(f"\nScene {os.path.basename(scene_dir)} completed!")