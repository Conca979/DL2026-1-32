import numpy as np

# Data Augmentations (Geometric & HED Stain Jitter)
HE_STAIN_MATRIX = np.array([
  [0.650, 0.072],
  [0.704, 0.990],
  [0.286, 0.105],
], dtype=np.float64)
HE_STAIN_MATRIX /= np.linalg.norm(HE_STAIN_MATRIX, axis=0, keepdims=True)


def hed_stain_jitter(image_rgb: np.ndarray, sigma: float = 0.2, bias: float = 0.05) -> np.ndarray:
  """Apply biologically-grounded H&E stain concentration scaling and shifting."""
  od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
  h, w, _ = od.shape
  flat_od = od.reshape(-1, 3)

  c = flat_od @ np.linalg.pinv(HE_STAIN_MATRIX).T
  # Stochastic independent shift and scale per stain channel
  alpha = np.random.uniform(1.0 - sigma, 1.0 + sigma, size=2)
  beta = np.random.uniform(-bias, bias, size=2)

  c[:, 0] = np.clip(c[:, 0] * alpha[0] + beta[0], 0, None)
  c[:, 1] = np.clip(c[:, 1] * alpha[1] + beta[1], 0, None)

  od_jittered = c @ HE_STAIN_MATRIX.T
  rgb_jittered = 256.0 * (10.0 ** -od_jittered) - 1.0
  return np.clip(rgb_jittered.reshape(h, w, 3), 0, 255).astype(np.uint8)
