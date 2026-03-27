import torch
import torch.nn.functional as F
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time

from internnav.model.utils.misc import set_random_seed
from .visual_encoder import VisionEncoder


# 定义常量
LATENT_SHAPE = (1, 4, 3584)
LATENT_SIZE = np.prod(LATENT_SHAPE) * 4 # float32


class ComplexityAnalyzer:
    def __init__(self, device="cuda", ):
        set_random_seed(0)
        self.device = device
        self.stream = torch.cuda.Stream(device=self.device)
        self.vision_encoder = VisionEncoder(device=self.device)

        # 1. EMA 统计量，用于自动对齐量级
        self.ema_alpha = 0.1
        self.spatial_ema = None   # 空间复杂度均值
        self.temporal_ema = None  # 时序变化均值
        self.last_feat = None
        
        # 2. 灵敏度阈值 (1.5 代表比平均水平高出 50% 时触发)
        self.threshold = 1.0

    def should_trigger(self, rgb: np.array):
        """在异步流中分析环境，返回是否需要推理"""
        with torch.cuda.stream(self.stream):
            vit_feat, _ = self.vision_encoder.get_patch_importance(rgb) # (1564, 1280)

            # --- A. 空间复杂度分析 (Spatial) ---
            # 针对 1564x1280，计算 Patch 间的标准差
            raw_spatial = torch.std(vit_feat).item()
            
            # 初始化或更新 EMA
            if self.spatial_ema is None:
                self.spatial_ema = raw_spatial
            else:
                self.spatial_ema = (1 - self.ema_alpha) * self.spatial_ema + self.ema_alpha * raw_spatial
            
            # 计算空间相对得分 (当前 / 平均)
            s_norm = raw_spatial / (self.spatial_ema + 1e-6)

            # --- B. 时序变化分析 (Temporal) ---
            t_norm = 1.0 # 默认不变化
            if self.last_feat is not None:
                # 计算两帧之间的 L2 范数差异 (比全局 Cosine 更灵敏)
                diff = vit_feat - self.last_feat
                raw_temporal = torch.norm(diff, p=2).item()
                
                # 初始化或更新 EMA
                if self.temporal_ema is None:
                    self.temporal_ema = raw_temporal
                else:
                    self.temporal_ema = (1 - self.ema_alpha) * self.temporal_ema + self.ema_alpha * raw_temporal
                
                # 计算时序相对得分
                t_norm = raw_temporal / (self.temporal_ema + 1e-6)

            # --- C. 异步更新备份 ---
            if self.last_feat is None:
                self.last_feat = vit_feat.clone()
            else:
                # 使用 non_blocking 拷贝，避免阻塞主流
                self.last_feat.copy_(vit_feat, non_blocking=True)

            # --- D. 综合决策 ---
            # 采用 Max 逻辑：只要环境变复杂了，或者机器人动得比平时快，就触发请求
            combined_score = max(s_norm, t_norm)
            
            # 触发条件：得分超过阈值
            return combined_score > self.threshold, combined_score

    def reset(self):
        self.spatial_ema = None
        self.temporal_ema = None
        self.last_feat = None


def scout_process(shm_name, trigger_event):
    """端侧进程：负责分析环境并触发请求"""
    device = "cuda"
    analyzer = ComplexityAnalyzer(device=device)
    
    while True:
        # 1. 模拟获取本地 ViT 输出 (1564, 1280)
        vit_feat = torch.randn(1564, 1280, device=device)
        
        # 2. 异步分析
        triggered, score = analyzer.should_trigger(vit_feat)
        
        if triggered:
            # 触发云端推理信号
            print(f"[Scouter] High Complexity ({score:.3f})! Triggering Slow Model.")
            # 这里可以调用网络发送代码，或者通知专门的通信线程
            # network_client.send_request(vit_feat)
            
        time.sleep(0.03) # 约 30Hz 的扫描频率