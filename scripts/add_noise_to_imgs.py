import os, glob, cv2
import numpy as np
from albumentations import Compose, GaussNoise
from tqdm import tqdm

SIGMA = 40

# ─── 1) your RGB noise transform ─────────────────────────
rgb_transform = Compose([
    # var_limit takes variance → (σ², σ²)
    GaussNoise(var_limit=(SIGMA**2, SIGMA**2), p=1.0),
])

# filenames to skip noise
SKIP_RGB = 'color_1.png'
SKIP_DEPTH = 'depth_1.png'

def process_rgb(input_dir, output_dir, transform):
    # clear or make output
    if os.path.exists(output_dir):
        for f in glob.glob(os.path.join(output_dir, '*')):
            if os.path.isfile(f):
                os.remove(f)
    else:
        os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(glob.glob(os.path.join(input_dir, '*.png')), desc="RGB"):
        fn = os.path.basename(img_path)
        img = cv2.imread(img_path)                            # BGR, 8-bit

        # skip noise for the designated image
        if fn == SKIP_RGB:
            out_img = img
        else:
            out_img = transform(image=img)['image']

        cv2.imwrite(os.path.join(output_dir, fn), out_img)


# ─── 2) a simple depth‐noise function ─────────────────────
def add_depth_noise(depth, sigma=SIGMA):
    """
    depth: 2D array (uint8, uint16, or float32)
    sigma: std‐dev of the zero‐mean noise, in the same units as depth
    """
    dtype = depth.dtype
    d = depth.astype(np.float32)
    noise = np.random.randn(*d.shape) * sigma
    d_noisy = d + noise

    # clip to valid range
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        d_noisy = np.clip(d_noisy, info.min, info.max)
    else:
        d_noisy = np.clip(d_noisy, d.min(), d.max())

    return d_noisy.astype(dtype)


def process_depth(input_dir, output_dir, sigma=SIGMA):
    # clear or make output
    if os.path.exists(output_dir):
        for f in glob.glob(os.path.join(output_dir, '*')):
            if os.path.isfile(f):
                os.remove(f)
    else:
        os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(glob.glob(os.path.join(input_dir, '*.png')), desc="Depth"):
        fn = os.path.basename(img_path)
        depth = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)   # e.g. single‐channel uint16

        # skip noise for the designated image
        if fn == SKIP_DEPTH:
            out_depth = depth
        else:
            out_depth = add_depth_noise(depth, sigma=sigma)

        cv2.imwrite(os.path.join(output_dir, fn), out_depth)


if __name__ == "__main__":
    # your paths
    rgb_in   = '/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/mustard_testing_final/rgb'
    rgb_out  = '/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/mustard_testing_final_w_noise/rgb/'
    depth_in  = '/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/mustard_testing_final/depth'
    depth_out  = '/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/mustard_testing_final_w_noise/depth/'
    
    print("Starting add_noise_to_imgs.py")
    process_rgb(  rgb_in,   rgb_out,   rgb_transform)
    process_depth(depth_in, depth_out, sigma=SIGMA*2)
    print("All done!")
