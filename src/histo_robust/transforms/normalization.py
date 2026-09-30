import numpy as np
from typing import Dict

# Ruderman et al. (1998) matrices for Reinhard l-alpha-beta color transfer
_LMS_MAT = np.array([
  [0.3811, 0.5783, 0.0402],
  [0.1967, 0.7244, 0.0782],
  [0.0241, 0.1288, 0.8444],
], dtype=np.float64)  #https://home.cis.rit.edu/~cnspci/references/dip/color_transfer/reinhard2001.pdf

_LAB_MAT = np.array([
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
  [1.0 / np.sqrt(6.0),  1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
  [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0),  0.0],
], dtype=np.float64)

_INV_LAB_MAT = np.array([
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0),  1.0 / np.sqrt(2.0)],
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0)],
  [1.0 / np.sqrt(3.0), -2.0 / np.sqrt(6.0),  0.0],
], dtype=np.float64)

_INV_LMS_MAT = np.linalg.inv(_LMS_MAT)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
  norm_rgb = np.clip(rgb.astype(np.float64) / 255.0, 1e-4, 1.0)
  lms = norm_rgb @ _LMS_MAT.T
  log_lms = np.log10(np.clip(lms, 1e-4, None))
  return log_lms @ _LAB_MAT.T


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
  log_lms = lab @ _INV_LAB_MAT.T
  lms = 10.0 ** log_lms
  rgb = lms @ _INV_LMS_MAT.T
  return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def reinhard_fit(reference_rgb: np.ndarray) -> Dict[str, np.ndarray]:
  """Extract mean and standard deviation per channel in Reinhard LAB space."""
  lab = _rgb_to_lab(reference_rgb)
  return {
    "mean": np.mean(lab, axis=(0, 1)),
    "std": np.std(lab, axis=(0, 1)) + 1e-6,
  }


def reinhard_apply(image_rgb: np.ndarray, ref_stats: Dict[str, np.ndarray]) -> np.ndarray:
  """Apply Reinhard statistical color transfer (pure NumPy)."""
  lab = _rgb_to_lab(image_rgb)
  mean = np.mean(lab, axis=(0, 1))
  std = np.std(lab, axis=(0, 1)) + 1e-6

  norm_lab = np.zeros_like(lab)
  for c in range(3):
    norm_lab[:, :, c] = ((lab[:, :, c] - mean[c]) / std[c]) * ref_stats["std"][c] + ref_stats["mean"][c]

  return _lab_to_rgb(norm_lab)


def macenko_fit(reference_rgb: np.ndarray, od_threshold: float = 0.15) -> Dict[str, np.ndarray]:
  """Estimate H&E stain vectors and 99th percentile concentrations from reference tile."""
  od = -np.log10((reference_rgb.astype(np.float64) + 1.0) / 256.0)
  flat_od = od.reshape(-1, 3)
  mask = np.linalg.norm(flat_od, axis=1) > od_threshold
  flat_od = flat_od[mask]

  _, _, vh = np.linalg.svd(flat_od, full_matrices=False)
  proj = flat_od @ vh[:2].T
  phi = np.arctan2(proj[:, 1], proj[:, 0])

  min_phi = np.percentile(phi, 1.0)
  max_phi = np.percentile(phi, 99.0)

  v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
  v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])

  # Ensure Hematoxylin is column 0 (stronger in red absorption)
  if v1[0] < v2[0]:
    v1, v2 = v2, v1

  stain_matrix = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
  concentrations = flat_od @ np.linalg.pinv(stain_matrix).T
  q99 = np.percentile(concentrations, 99.0, axis=0)

  return {"stain_matrix": stain_matrix, "q99": q99}


def macenko_apply(image_rgb: np.ndarray, ref_params: Dict[str, np.ndarray], od_threshold: float = 0.15) -> np.ndarray:
  """Normalize an H&E tile to canonical reference stain vectors and concentrations."""
  od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
  h, w, _ = od.shape
  flat_od = od.reshape(-1, 3)
  mask = np.linalg.norm(flat_od, axis=1) > od_threshold

  if mask.sum() < 0.20 * h * w:
    return image_rgb

  valid_od = flat_od[mask]
  try:
    _, _, vh = np.linalg.svd(valid_od, full_matrices=False)
    proj = valid_od @ vh[:2].T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi = np.percentile(phi, 1.0)
    max_phi = np.percentile(phi, 99.0)

    v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])
    if v1[0] < v2[0]:
      v1, v2 = v2, v1
    tile_stain = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])

    tile_conc = flat_od @ np.linalg.pinv(tile_stain).T
    q99 = np.percentile(tile_conc[mask], 99.0, axis=0) + 1e-6

    # Match reference concentration scaling
    norm_conc = tile_conc * (ref_params["q99"] / q99)
    norm_od = norm_conc @ ref_params["stain_matrix"].T
    norm_rgb = 256.0 * (10.0 ** -norm_od) - 1.0
    return np.clip(norm_rgb.reshape(h, w, 3), 0, 255).astype(np.uint8)
  except Exception:
    return image_rgb
