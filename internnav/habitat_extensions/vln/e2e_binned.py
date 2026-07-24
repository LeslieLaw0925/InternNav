from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image


class EndToEndBinnedSwapping:
    def __init__(self, 
                 s1_type,
                 base_resolution: Tuple[int, int] = (480, 640)):
        """
        End-to-End Coarse-Grained Adaptive Baseline.
        同时针对通信（图像分辨率）与本地计算（System 1 迭代步数 S & 采样数 C）进行联动调整。
        """
        self.H_full, self.W_full = base_resolution
        self.bandwidth_history = []
        
        # 建立静态端到端多维策略表 (Binned Policy Table)
        # 根据当前带宽，同时强制指定 [通信分辨率档位] 与 [本地S1推理配置]
        if s1_type == 'nextdit_async':
            s1_config = [(10, 32), (6, 20), (2, 8)]  # (S, C) for each bandwidth bin
        elif s1_type == 'navdp_async':
            s1_config = [(20, 32), (11, 20), (2, 8)]  
        else:
            raise ValueError(f"Unsupported s1_type: {s1_type}.")
           
        self.policy_bins = [
            {
                "min_bandwidth_mbps": 2.5, 
                "scale": 1.0, "name": "High-Fidelity Mode",
                "s1_config": s1_config[0],
            },
            {
                "min_bandwidth_mbps": 1.5, 
                "scale": 0.5, "name": "Balanced Mode",
                "s1_config": s1_config[1],
            },
            {
                "min_bandwidth_mbps": 0.0,  
                "scale": 0.25, "name": "Extreme Low-Latency",
                "s1_config": s1_config[2],
            }
        ]

    def select_e2e_configuration(self, estimated_bandwidth_bps: float) -> Dict[str, Any]:
        """
        粗粒度多维联合查表：一键决定图像分辨率与 S1 算力配置
        """
        estimated_bandwidth_mbps = estimated_bandwidth_bps / 1024.0 / 1024.0
        print(f"[E2E Binned] Estimated Bandwidth: {estimated_bandwidth_mbps:.2f} Mbps.")
        selected_policy = self.policy_bins[-1] # 默认最低配置
        for policy in self.policy_bins:
            if estimated_bandwidth_mbps >= policy["min_bandwidth_mbps"]:
                selected_policy = policy
                break
                
        # 计算通信开销
        scale = selected_policy["scale"]
        target_H = int(self.H_full * scale)
        target_W = int(self.W_full * scale)
        
        return {
            "mode_name": selected_policy["name"],
            "resolution": (target_H, target_W),
            's1_config': selected_policy["s1_config"],
        }

    def execute_pipeline(self, raw_image: Image.Image, 
                         current_bandwidth_mbps: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        端到端闭环接口
        """
        if current_bandwidth_mbps is None:
            return raw_image, None
        
        config = self.select_e2e_configuration(current_bandwidth_mbps)
        
        # 执行全局下采样
        h, w = config["resolution"]
        compressed_image = raw_image.resize((w, h), Image.BILINEAR)      
        print(f"[E2E Binned] Selected Mode: {config['mode_name']}, Resolution: {h}x{w}.")      
        return compressed_image, config.get('s1_config', None)