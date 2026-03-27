import os
import json
import base64
import pickle

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
from torchvision.transforms import v2
from transformers import (
    Qwen2_5_VisionTransformerPretrainedModel,
    PretrainedConfig
)
        
image_preprocess = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True), # 替代 ToTensor()，并归一化到 [0, 1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class LightFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.mobilenet_v2(pretrained=True).features[:6]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x)


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


def init_visual_encoder(visual_encoder_path: str, 
                        device = torch.device('cuda')):
    # set_random_seed(0)
    clip_model = CLIPVisionModel.from_pretrained(visual_encoder_path)
    clip_model.to(device)
    clip_model.eval()
    clip_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path)

    vgg_feature_extractor = LightFeatureExtractor().to(device)
    vgg_feature_extractor.eval()
    return clip_model, clip_processor, vgg_feature_extractor


def init_vit_model(model_dir: str, device) -> Qwen2_5_VisionTransformerPretrainedModel:
    vit_path = os.path.join(model_dir, 'vit_from_dual_vln.ckpt')
    config_path = os.path.join(model_dir, 'config.json')
    config = json.load(open(config_path, 'r'))
    vision_config = config.get("vision_config", None)

    vision_config = Qwen2_5_VLVisionConfig(**vision_config)
    vit_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(vision_config)
    vit_model.load_state_dict(torch.load(vit_path, map_location="cpu"), strict=True)
    vit_model.to(device)

    return vit_model


def serialize_obs(obs):
    serialized = pickle.dumps(obs)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded


def remove_from_obs(obs: dict) -> dict:
    keys = ['globalgps', 'globalrotation', 'topdown_rgb', 'topdown_depth', 'instruction_tokens', 'depth']
    for key in keys:
        obs.pop(key, None)
    return obs