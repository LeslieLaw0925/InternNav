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

    def __init__(self, device='cuda'):
        self.device = device

        vit_path = 'checkpoints/Qwen2_5_VisionTransformer/vit_from_dual_vln.ckpt'
        config_path = 'checkpoints/Qwen2_5_VisionTransformer/config.json'
        config = json.load(open(config_path, 'r'))

        model_dir = 'checkpoints/Qwen2_5_VisionTransformer'

        vision_config = config.get("vision_config", None)
        vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        self.vit_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(vision_config, 
                                                                               attn_implementation="flash_attention_2")
        self.vit_model.load_state_dict(torch.load(vit_path, map_location="cpu"), strict=True)
        self.vit_model.to(device=self.device, dtype=torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.processor.tokenizer = tokenizer
        self.processor.tokenizer.padding_side = 'left'

    def get_patch_importance(self, image: np.ndarray):
        text = self.processor.apply_chat_template([""], tokenize=False, add_generation_prompt=True)
        image = Image.fromarray(image)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        image_grid_thw = inputs.get('image_grid_thw')
        h_grid, w_grid = image_grid_thw[0][1].item(), image_grid_thw[0][2].item()
        pixel_values = inputs.get('pixel_values')

        with torch.no_grad():
            merger_outputs, outputs = self.vit_model(pixel_values, image_grid_thw)
        merger_scale = math.sqrt(outputs.shape[0] / merger_outputs.shape[0])

        h_grid, w_grid = int(h_grid / merger_scale), int(w_grid / merger_scale)
        patch_importance = torch.norm(merger_outputs, dim=-1)
        patch_importance = patch_importance / patch_importance.max() # 归一化到0-1
        patch_importance = patch_importance.reshape(h_grid, w_grid)
        patch_importance = patch_importance.cpu().to(dtype=torch.float32).numpy()

        return patch_importance


def draw_heatmap_on_image(image, importance_map):
    # importance_map: (h_grid, w_grid)，值在0-1之间
    # h_grid, w_grid = importance_map.shape
    h_img, w_img, _ = image.shape

    # 将 importance_map 放大到图像尺寸
    heatmap = cv2.resize(importance_map, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    # 将 heatmap 转换为颜色图（使用 colormap）
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 将热力图叠加到原图上，alpha 控制透明度
    alpha = 0.5
    overlayed_image = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    os.makedirs('logs/test_data', exist_ok=True)
    cv2.imwrite(f'logs/test_data/heatmap_overlay_{time.time()}.jpg', overlayed_image)  # 保存叠加后的图像以供对比


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


def numpy_compression_v2(image: np.array, patch_size=28, compression_factor=4):
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


def numpy_compression(image, importance, keep_ratio=0.01):
    """
    image_np: (C, H, W) 的 numpy 数组
    attn_map: (h, w) 的注意力热力图，与 patch 数量对应
    threshold: 低于此阈值的区域将被压缩
    """

    # cv2.imwrite('original_image.jpg', image)  # 保存原始图像以供对比
    H, W, _ = image.shape
    patch_size = 100 # H // importance.shape[0] # 28

    threshold = np.percentile(importance, (1 - keep_ratio) * 100)  # 根据百分位数动态确定阈值
    
    # 1. 标识低兴趣区域 (Low Interest Mask)
    # mask = (importance < threshold)
    # import pdb; pdb.set_trace()
    # mask = np.random
    # mask = generate_mask(importance.shape, zero_ratio=0.01)
    new_H, new_W = H // patch_size + 1, W // patch_size + 1
    # mask = generate_mask((new_H, new_W), zero_ratio=0.01)
    
    # 2. 分离数据：我们将图像切分为 Patch 列表
    # 重要区域保留原始 patch，不重要区域进行池化
    compressed_data = []
    metadata = [] # 记录位置信息用于还原
    
    idx = 0
    # for i in range(importance.shape[0]):
    #     for j in range(importance.shape[1]):
    for i in range(new_H):
        for j in range(new_W):
            # 获取当前 patch 的像素范围
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            y2 = min(y2, H)  # 防止越界
            x2 = min(x2, W)
            patch = image[y1:y2, x1:x2, :]
            
            # if mask[i, j]:
            #     # 对低关注度 patch 进行 2x2 平均池化，体积减少 4 倍
            #     compressed_patch = cv2.resize(patch, (patch_size//4, patch_size//4), 
            #                                   interpolation=cv2.INTER_AREA)
            #     compressed_data.append(compressed_patch)
            #     metadata.append(0) # 标记为压缩
            # else:
            #     compressed_data.append(patch)
            #     metadata.append(1) # 标记为原始

            compressed_patch = cv2.resize(patch, (patch_size//8, patch_size//8), 
                                              interpolation=cv2.INTER_AREA)
            compressed_data.append(compressed_patch)
    
    return compressed_data, metadata


def adaptive_compression_v2(image, patch_importance, threshold=0.1):
    """
    通过对非重要区域进行强模糊来减小文件体积，同时 100% 保留重要区域
    """
    h, w, c = image.shape
    
    # 1. 将 Patch Importance 转换为二值掩码 (0 或 1)
    # 只有重要性大于阈值的 patch 才设为 1
    threshold = np.percentile(patch_importance, (1 - threshold) * 100)  # 根据百分位数动态确定阈值
    binary_patch_mask = (patch_importance >= threshold).astype(np.float32)
    
    # 2. 将掩码放大到原图尺寸
    # 使用 cv2.INTER_NEAREST 保证 Patch 边缘清晰，不产生中间值
    mask = cv2.resize(binary_patch_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.stack([mask] * 3, axis=-1)
    
    # 3. 对全图进行强力模糊（这是压缩体积的关键）
    # 模糊程度越高，非重要区域的熵越低，压缩后的 buffer 越小
    low_quality_area = cv2.GaussianBlur(image, (51, 51), 4)
    
    # 4. 硬合成：重要区域 100% 像素保留，非重要区域 100% 模糊
    # final = 原图(重要部分) + 模糊图(非重要部分)
    final_img = (image * mask + low_quality_area * (1 - mask)).astype(np.uint8)
    
    return final_img


def compress_image_by_patch(image, patch_importance, 
                            thresholds=(0.12, 0.08), 
                            patch_size=14, 
                            quality_levels=None):
    """
    根据 Patch 重要性对图像进行局部压缩
    :param image: 输入图像 (H, W, 3)
    :param patch_importance: 重要性矩阵 (h_patches, w_patches)，值通常在 [0, 1]
    :param patch_size: ViT 的 patch 大小
    :param quality_levels: 字典，定义重要性区间对应的 JPEG 质量 (0-100)
    :return: 压缩后的图像
    """
    if quality_levels is None:
        # 默认分三档：高、中、低重要性
        quality_levels = {
            'high': 90,   # 重要性 > 0.7
            'medium': 50, # 0.3 <= 重要性 <= 0.7
            'low': 10     # 重要性 < 0.3
        }

    h, w, c = image.shape
    h_patches, w_patches = patch_importance.shape
    
    # 创建一个空的目标图像
    compressed_img = np.zeros_like(image)

    for i in range(h_patches):
        for j in range(w_patches):
            # 1. 确定当前 patch 的像素坐标范围
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size

            x2 = min(x2, w)
            y2 = min(y2, h)  # 防止越界
            
            patch = image[y1:y2, x1:x2]
            score = patch_importance[i, j]

            # 2. 根据得分确定压缩质量系数
            if score > thresholds[0]:
                q = quality_levels['high']
            elif score > thresholds[1]:
                q = quality_levels['medium']
            else:
                q = quality_levels['low']

            # 3. 对单个 Patch 执行压缩/解压模拟 (JPEG 压缩)
            # 注意：实际存储时需特殊格式，此处代码演示的是“质量损失”的效果
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            _, encimg = cv2.imencode('.jpg', patch, encode_param)
            decimg = cv2.imdecode(encimg, 1)

            # 4. 放回原位置
            compressed_img[y1:y2, x1:x2] = decimg

    return compressed_img


def verify_communication_reduction(original_img, processed_img, quality=90):
    # 将图像编码为内存缓冲区，模拟网络传输的数据流
    _, buffer_orig = cv2.imencode('.jpg', original_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    _, buffer_proc = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    size_orig = len(buffer_orig) / 1024  # KB
    size_proc = len(buffer_proc) / 1024  # KB
    reduction = (1 - size_proc / size_orig) * 100
    
    print(f"原始通信量: {size_orig:.2f} KB")
    print(f"处理后通信量: {size_proc:.2f} KB")
    print(f"通信量减少了: {reduction:.2f}%")
    
    return size_orig, size_proc


def solve(thresholds, data_sizes):
    from scipy.optimize import curve_fit

    # 线性拟合
    thresholds = thresholds.reshape(-1, 1)  # 转换为二维数组
    data_sizes = data_sizes.reshape(-1, 1)

    # 2. 定义拟合函数
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c

    model = LinearRegression()
    model.fit(thresholds, data_sizes)

    k = model.coef_[0][0]
    b = model.intercept_[0]

    print(f"拟合方程: Size = {k:.2f} * Threshold + {b:.2f}")
    print(f"相关系数 (R²): {model.score(thresholds, data_sizes):.4f}")

    return model


if __name__ == "__main__":
    import os
    import time

    image_path = '/home/smc/projects/InternNav/data/preview/vln_ce/traj_data/r2r/1LXtFkjw3qL/000087/videos/chunk-000/observation.images.rgb'
    records = []
    total_thresholds = []
    thresholds = np.arange(0.1, 1.0, 0.05)

    vision_encoder = VisionEncoder()

    for img_file in os.listdir(image_path):
        if not img_file.endswith('.jpg'):
            continue
        
        image_file_path = os.path.join(image_path, img_file)
        image = Image.open(image_file_path)

        patch_importance = vision_encoder.get_patch_importance(np.array(image))
        import pdb; pdb.set_trace()

        start_time = time.time()

        records = []
        for threshold in thresholds:
            compressed_img = adaptive_compression_v2(np.array(image), patch_importance, threshold=threshold)
            _, compressed_buffer = cv2.imencode('.jpg', compressed_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            records.append(len(compressed_buffer))
            total_thresholds.append(threshold)

        liner_model = solve(np.array(thresholds), np.array(records))
        
        end_time = time.time()
        print(f"处理 {img_file} 耗时: {end_time - start_time:.2f} 秒")

       