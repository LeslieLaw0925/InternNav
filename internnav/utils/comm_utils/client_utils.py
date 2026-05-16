import base64
import pickle

import numpy as np
import cv2


def serialize_obs(obs):
    serialized = pickle.dumps(obs)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded


def remove_from_obs(obs: dict) -> dict:
    keys = ['globalgps', 'globalrotation', 'topdown_rgb', 'topdown_depth', 'instruction_tokens'] # depth
    for key in keys:
        obs.pop(key, None)
    return obs


def numpy_compression_by_patch(image, importance, keep_ratio=0.1, compression_factor=4):
    """
    image_np: (C, H, W) 的 numpy 数组
    attn_map: (h, w) 的注意力热力图，与 patch 数量对应
    threshold: 低于此阈值的区域将被压缩
    """        
    H, W, _ = image.shape

    importance = importance.detach().cpu().numpy().astype(np.float32)
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
    
    for i in range(importance.shape[0]):
        for j in range(importance.shape[1]):
            # 获取当前 patch 的像素范围
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            y2 = min(y2, H)  # 防止越界
            x2 = min(x2, W)
            patch = image[y1:y2, x1:x2, :]
            
            if mask[i, j]:
                # 对低关注度 patch 进行平均池化
                compressed_patch = cv2.resize(patch, 
                                              (patch_size//compression_factor, patch_size//compression_factor), 
                                              interpolation=cv2.INTER_AREA)
                compressed_data.append(compressed_patch)
                metadata.append(0) # 标记为压缩
            else:
                compressed_data.append(patch)
                metadata.append(1) # 标记为原始
    
    return compressed_data, metadata


def estimate_numpy_transfer_size(image, patch_size, p_keep=0.1, dtype=np.uint8, compression_factor=4):
    """
    预估压缩后的 NumPy 图像传输大小(单位: Bytes)
    
    参数:
    image: 原始图像的 NumPy 数组
    patch_size: patch 的边长 (如 16)
    p_keep: 保留的高分辨率 patch 比例 (如 0.1)
    dtype: 数据类型，默认 uint8 (1 byte)
    """
    h, w, c = image.shape

    # 计算单个像素占用的字节数
    item_size = np.dtype(dtype).itemsize
    
    # 总 patch 数量
    p_total = (h // patch_size) * (w // patch_size)
    
    # 如果输入的是比例 (0 < p_keep < 1)
    if 0 < p_keep < 1:
        p_keep = int(p_total * p_keep)
    
    # 确保 p_keep 不超过总数
    p_keep = min(p_keep, p_total)
    
    # 1. 高分辨率部分大小
    high_res_size = p_keep * (patch_size ** 2) * c * item_size
    
    # 2. 低分辨率部分大小 (分辨率压缩 4 倍，面积压缩 16 倍)
    # 注意：如果 patch_size 是 16，低分辨率 patch 变成了 1x1 像素
    low_res_patch_size = patch_size / compression_factor
    low_res_size = (p_total - p_keep) * (low_res_patch_size ** 2) * c * item_size
    
    # 3. 还有一个 mask 来告诉接收方哪些位置是高分辨率的
    # p_total 个 bit 转换为 bytes
    mask_size = int(np.ceil(p_total / 8))
    total_size_bytes = high_res_size + low_res_size + mask_size

    return total_size_bytes


def solve_optimal_patch_ratio(image, time_constraint, bandwidth_bps, compression_factor=4, patch_size=28):
    """
    求解满足时延约束的最大 patch 保留比例
    
    参数:
    time_constraint: 约束的总时延 (s)
    image: 原始图像的 NumPy 数组
    bandwidth_bps: 当前网络带宽 (Bytes/s)
    compression_factor: 压缩倍率 (4x4, 16倍)
    """
    if bandwidth_bps is None:
        return 1.0  # 无法获取带宽信息，默认图像不做压缩
        
    if time_constraint <= 0:
        return 0.1  # 时延要求太苛刻，即使压缩到极限也无法满足，给最低的patch ratio
    
    h, w, c = image.shape
    # 掩码数组大小 (Bytes)，每个 patch 需要 1 bit 来标记是否保留高分辨率
    mask_size = int(np.ceil((h // patch_size) * (w // patch_size) / 8))
    
    # 2. 计算最大允许字节数
    size_max = time_constraint * bandwidth_bps - mask_size
    s_org = h * w * c  # 原图numpy数组的字节数大小
    
    # 3. 求解 p
    # 公式: p * (1 - 1/r^2) + 1/r^2 = size_max / s_org
    inv_r2 = 1.0 / (compression_factor**2)
    ratio_limit = size_max / s_org
    
    p_star = (ratio_limit - inv_r2) / (1 - inv_r2)
    # 4. 边界裁剪
    p_star = min(1, max(0, p_star))
    return p_star


def find_optimal_config(t_max, alpha, beta, intercept, n_range, b_range):
    '''Find optimal config for system 1 inference'''

    valid_configs = []
    
    for n in n_range:
        for b in b_range:
            predicted_t = alpha * n + beta * (n * b) + intercept
            if predicted_t <= t_max:
                # 记录配置及其实际预测时延
                valid_configs.append({
                    'infer_step': n, 
                    'traj_num': b, 
                    'latency': predicted_t,
                    'score': n * b  # 得分是 N*B
                })
    
    # 按照 score 排序，寻找最高效率的配置
    best_config = max(valid_configs, key=lambda x: x['score']) if valid_configs else None
    if best_config is None:
        best_config = {'infer_step': n_range[0], 
                        'traj_num': b_range[0]}
    return best_config