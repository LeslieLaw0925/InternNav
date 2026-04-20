import os
import json
import cv2
import numpy as np
import math
import time

from PIL import Image
import torch
from transformers import (
    Qwen2_5_VisionTransformerPretrainedModel,
    PretrainedConfig,
    AutoProcessor, 
    AutoTokenizer,
)

from internnav.utils.common_log_util import common_logger as log
from internnav.model.utils.misc import set_random_seed


class Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size


class VisionEncoder:

    def __init__(self, s1_type, device='cuda'):
        set_random_seed(0)
        self.device = device

        model_dir = 'checkpoints/Qwen2_5_VisionTransformer'
        if "navdp" in s1_type:  
            vit_path = os.path.join(model_dir, 'vit_from_navdp_vln.ckpt')
        elif "nextdit" in s1_type:
            vit_path = os.path.join(model_dir, 'vit_from_dual_vln.ckpt')
        else:
            raise ValueError(f"Unsupported System 1 type: {s1_type}")

        config_path = os.path.join(model_dir, 'config.json')
        config = json.load(open(config_path, 'r'))
        vision_config = config.get("vision_config", None)
        vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        self.vit_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(vision_config, 
                                                                               attn_implementation="flash_attention_2")
        self.vit_model.load_state_dict(torch.load(vit_path, map_location="cpu"), strict=True)
        self.vit_model.to(device=self.device, dtype=torch.float16).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.processor.tokenizer = tokenizer
        self.processor.tokenizer.padding_side = 'left'

    def get_patch_importance(self, image: np.ndarray, text=""):
        start_time = time.time()
        text = self.processor.apply_chat_template([text], tokenize=False, add_generation_prompt=True)
        image = Image.fromarray(image)

        # NEW: 压分辨率，加速vit推理
        w, h = image.size
        image = image.resize((w//2, h//2))

        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        image_grid_thw = inputs.get('image_grid_thw')
        h_grid, w_grid = image_grid_thw[0][1].item(), image_grid_thw[0][2].item()
        pixel_values = inputs.get('pixel_values')

        with torch.no_grad():
            merger_outputs, outputs = self.vit_model(pixel_values, image_grid_thw)

        merge_scale = math.sqrt(outputs.shape[0] / merger_outputs.shape[0])
        h_grid, w_grid = int(h_grid / merge_scale), int(w_grid / merge_scale)
        
        patch_importance = torch.norm(merger_outputs, dim=-1)
        patch_importance = patch_importance / patch_importance.sum() # 归一化到0-1
        patch_importance = patch_importance.reshape(h_grid, w_grid)

        return time.time() - start_time, patch_importance


def draw_origin_image(image: np.array, pixel=None, suffix=''):
    if pixel is not None:
        image = cv2.putText(
            image,
            f"{pixel[1]}, {pixel[0]}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        image = cv2.circle(image, (pixel[1], pixel[0]), 5, (0, 0, 255), -1)

    cv2.imwrite(f'logs/test_data/origin_image_{time.time()}{suffix}.jpg', image)  # 保存原始图像以供对比


def draw_heatmap_on_image(image, importance_map, pixel=None, episode='0', suffix=''):
    # importance_map: (h_grid, w_grid)，值在0-1之间
    h_img, w_img, _ = image.shape

    importance_map = importance_map / importance_map.max()
    importance_map = importance_map.cpu().to(dtype=torch.float32).numpy() # 转为numpy数组，方便后续处理

    # 将 importance_map 放大到图像尺寸
    heatmap = cv2.resize(importance_map, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    # 将 heatmap 转换为颜色图（使用 colormap）
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 将热力图叠加到原图上，alpha 控制透明度
    alpha = 0.5
    overlayed_image = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    if pixel is not None:
        overlayed_image = cv2.putText(
            overlayed_image,
            f"{pixel[1]}, {pixel[0]}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        overlayed_image = cv2.circle(overlayed_image, (pixel[1], pixel[0]), 5, (0, 0, 255), -1)

    cv2.imwrite(f'logs/test_data/episode_{episode}/heatmap_overlay_{time.time()}{suffix}.jpg', overlayed_image)  # 保存叠加后的图像以供对比


def generate_mask(shape, zero_ratio=0.01):
    """
    shape: tuple，例如 (224, 224) 或 (16, 16)
    zero_ratio: 置为0的比例
    """

    total = np.prod(shape)
    num_zero = int(total * zero_ratio)

    # 初始化全1
    arr = np.ones(total, dtype=np.uint8)

    # 随机选位置置0
    zero_indices = np.random.choice(total, num_zero, replace=False)
    arr[zero_indices] = 0

    # reshape回目标形状
    return arr.reshape(shape)


def numpy_compression_v2(image: np.array, patch_size=28, compression_factor=2):
    """
    image: (H, W, C) 的 numpy 数组
    patch_size: 每个 patch 的大小，例如 14
    compression_factor: 压缩因子，例如 4 表示将 patch 压缩到原来的1/4大小
    """

    H, W, _ = image.shape
    new_H, new_W = H // patch_size + 1, W // patch_size + 1

    compressed_data = []
    
    for i in range(new_H):
        for j in range(new_W):
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            y2 = min(y2, H)  # 防止越界
            x2 = min(x2, W)  # 防止越界

            patch = image[y1:y2, x1:x2, :]
            compressed_patch = cv2.resize(patch, 
                                          (patch_size//compression_factor, patch_size//compression_factor), 
                                          interpolation=cv2.INTER_AREA)
            compressed_data.append(compressed_patch)
    
    return compressed_data


def random_compression(image: np.array, importance, keep_ratio=0.1, compression_factor=4):
    H, W, _ = image.shape
    patch_size = H // importance.shape[0] # 28

    threshold = np.percentile(importance, (1 - keep_ratio) * 100)  # 根据百分位数动态确定阈值
    mask = (importance < threshold)
    mask_number = np.sum(mask) # 置1的元素数量，也就是需要压缩的patch数量

    total_num = mask.shape[0] * mask.shape[1]
    mat = np.zeros(total_num)
    idx = np.random.choice(total_num, mask_number, replace=False)
    mat[idx] = 1

    rand_mask = mat.reshape(mask.shape)

    compressed_data = []
    metadata = [] # 记录位置信息用于还原
    
    for i in range(importance.shape[0]):
        for j in range(importance.shape[1]):
            # 获取当前 patch 的像素范围
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            y2 = min(y2, H)  # 防止越界
            x2 = min(x2, W)
            patch = image[y1:y2, x1:x2, :]
            
            if rand_mask[i, j]:
                # 对低关注度 patch 进行 2x2 平均池化，体积减少 4 倍
                compressed_patch = cv2.resize(patch, 
                                              (patch_size//compression_factor, patch_size//compression_factor), 
                                              interpolation=cv2.INTER_AREA)
                compressed_data.append(compressed_patch)
                metadata.append(0) # 标记为压缩
            else:
                compressed_data.append(patch)
                metadata.append(1) # 标记为原始
    
    return compressed_data, metadata


def numpy_compression_by_patch(image, importance, keep_ratio=0.1, compression_factor=4):
    """
    image_np: (C, H, W) 的 numpy 数组
    attn_map: (h, w) 的注意力热力图，与 patch 数量对应
    threshold: 低于此阈值的区域将被压缩
    """
    H, W, _ = image.shape

    # NEW: patch importance 还原回(17, 23)，与原图对应
    importance = cv2.resize(importance, (23, 17), interpolation=cv2.INTER_LINEAR) # (17, 23)

    patch_size = H // importance.shape[0] # 28

    threshold = np.percentile(importance, (1 - keep_ratio) * 100)  # 根据百分位数动态确定阈值
    
    # 1. 标识低兴趣区域 (Low Interest Mask)
    mask = (importance < threshold)
    
    # 2. 分离数据：我们将图像切分为 Patch 列表
    # 重要区域保留原始 patch，不重要区域进行池化
    compressed_data = []
    metadata = [] # 记录位置信息用于还原
    
    # idx = 0
    for i in range(importance.shape[0]):
        for j in range(importance.shape[1]):
            # 获取当前 patch 的像素范围
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            y2 = min(y2, H)  # 防止越界
            x2 = min(x2, W)
            patch = image[y1:y2, x1:x2, :]
            
            if mask[i, j]:
                # 对低关注度 patch 进行 2x2 平均池化，体积减少 4 倍
                compressed_patch = cv2.resize(patch, 
                                              (patch_size//compression_factor, patch_size//compression_factor), 
                                              interpolation=cv2.INTER_AREA)
                compressed_data.append(compressed_patch)
                metadata.append(0) # 标记为压缩
            else:
                compressed_data.append(patch)
                metadata.append(1) # 标记为原始
    
    return compressed_data, metadata


if __name__ == "__main__":
    import os
    os.environ["TRITON_PTXAS_PATH"]="/usr/local/cuda-12.6/bin/ptxas"

    import time

    image_path = '000.jpg'
    vision_encoder = VisionEncoder()
    vision_encoder.vit_model.eval()

    image = Image.open(image_path)

    W, H = image.size
    resize_image = image.resize((384, 384))
    resize_image.save('resized_image.jpg')  # 保存缩放后的图像以供对比

    _, patch_importance = vision_encoder.get_patch_importance(np.array(image))

    start_time = time.time()
    _, patch_importance = vision_encoder.get_patch_importance(np.array(image))
    end_time = time.time()
    print(f"Original image processing time: {end_time - start_time:.4f} seconds")
    draw_heatmap_on_image(np.array(image), patch_importance, suffix='_original')

    start_time = time.time()
    _, patch_importance_resized = vision_encoder.get_patch_importance(np.array(resize_image))
    # patch_importance_resized = cv2.resize(patch_importance_resized, (patch_importance.shape[1], patch_importance.shape[0]), interpolation=cv2.INTER_LINEAR)
    print(f"Resized image processing time: {time.time() - end_time:.4f} seconds")
    draw_heatmap_on_image(np.array(image), patch_importance_resized, suffix='_resized')

       

       