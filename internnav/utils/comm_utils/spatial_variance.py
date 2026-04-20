import torch
import scipy.ndimage as ndi

from internnav.utils.common_log_util import common_logger as log


def spatial_center(p):
    h, w = p.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=p.device),
        torch.arange(w, device=p.device),
        indexing='ij'
    )
    cx = (p * x).sum()
    cy = (p * y).sum()
    return cx, cy


def spatial_variance(p, cx, cy):
    h, w = p.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=p.device),
        torch.arange(w, device=p.device),
        indexing='ij'
    )
    var_x = (p * (x - cx)**2).sum()
    var_y = (p * (y - cy)**2).sum()
    return var_x + var_y


def num_hot_regions(p, threshold_ratio=1.5):
    p_np = p.cpu().numpy()
    thresh = p_np.mean() * threshold_ratio
    mask = p_np > thresh
    _, num = ndi.label(mask)
    return num


def spatial_covariance(p):
    h, w = p.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=p.device),
        torch.arange(w, device=p.device),
        indexing='ij'
    )

    cx = (p * x).sum()
    cy = (p * y).sum()

    cov_xx = (p * (x - cx)**2).sum()
    cov_yy = (p * (y - cy)**2).sum()
    cov_xy = (p * (x - cx)*(y - cy)).sum()

    return cov_xx, cov_yy, cov_xy


def analyze_spatial_distribution(patch_importance):
    eps = 1e-8

    # entropy
    entropy = -(patch_importance * (patch_importance + eps).log()).sum()

    # center
    cx, cy = spatial_center(patch_importance)
    # variance
    var = spatial_variance(patch_importance, cx, cy)

    # hot regions
    # num_regions = num_hot_regions(p)

    # covariance
    # cov_xx, cov_yy, cov_xy = spatial_covariance(p)

    complexity = 0.8 * entropy + 0.2 * var
    log.info(f"Entropy: {entropy:.4f}, Variance: {var:.4f}, Complexity: {complexity:.4f}")
    return complexity