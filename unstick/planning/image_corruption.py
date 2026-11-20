"""Image corruptions for the visual shifts, applied to uint8 (..., H, W, C) frames."""
import numpy as np
import cv2


def apply_salt_pepper(frame, density, rng):
    h, w = frame.shape[:2]
    out = frame.copy()
    n = int(density * h * w)
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 255
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 0
    return out


def apply_blur(frame, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_dark(frame, factor):
    out = frame.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


# name -> (function, default level)
CORRUPTIONS = {
    "blur":  (apply_blur, 2.0),
    "snp1":  (apply_salt_pepper, 0.01),
    "dark":  (apply_dark, 0.5),
}


def corrupt_frames(frames, corruption_name, level=None, seed=0):
    """seed=None: fresh noise on every call."""
    if corruption_name is None or corruption_name == "none":
        return frames

    func, default_level = CORRUPTIONS[corruption_name]
    if level is None:
        level = default_level

    rng = np.random.RandomState(seed)
    orig_shape = frames.shape
    flat = frames.reshape(-1, *frames.shape[-3:])  # (N, H, W, C)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        if corruption_name.startswith("snp"):
            out[i] = func(flat[i], level, rng)
        else:
            out[i] = func(flat[i], level)
    return out.reshape(orig_shape)


def corrupt_obs_dict(obs, corruption_name, level=None, seed=0):
    """Corrupts obs["visual"] only."""
    if corruption_name is None or corruption_name == "none":
        return obs
    out = dict(obs)
    out["visual"] = corrupt_frames(obs["visual"], corruption_name, level, seed)
    return out
