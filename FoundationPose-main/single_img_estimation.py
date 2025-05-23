#!/usr/bin/env python3
"""
single_img_estimation.py
Estimate a 6-DoF pose from one RGB or RGB-D image and save it to disk.
"""
import os, argparse, logging, json, cv2, imageio
import numpy as np, trimesh
# import nvdiffrast.torch as dr
from estimater import *
from datareader import *
from scipy.spatial.transform import Rotation as R

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_intrinsics(K_file: str) -> np.ndarray:
    """Load a 3×3 intrinsics matrix from .txt or .json"""
    if K_file.endswith('.json'):
        arr = np.asarray(json.load(open(K_file)), dtype=np.float32)
    else:
        arr = np.loadtxt(K_file, dtype=np.float32)
    return torch.from_numpy(arr).to(device)

if __name__ == '__main__':
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    # I/O arguments
    parser.add_argument('--color_file', required=True, help='RGB image (png/jpg)')
    parser.add_argument('--depth_file', required=True, help='depth image (png/jpg)')
    parser.add_argument('--mask_file', required=True, help=' binary mask')
    parser.add_argument('--K_file',    required=True, help='3×3 camera intrinsics (.txt or .json)')
    parser.add_argument('--save_dir',  required=True, help='Directory to dump pose & visuals')
    # model & algorithm
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--debug', type=int, choices=[0,1,2,3], default=1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_logging_format()
    set_seed(0)

    # load color
    color_rgb = imageio.imread(args.color_file)[..., :3]
    color_bgr = color_rgb[..., ::-1].copy()
    H, W = color_rgb.shape[:2]
    # depth map png
    depth = None
    if args.depth_file:
        if args.depth_file.endswith('.npy'):
            depth = np.load(args.depth_file).astype(np.float32)
        else:
            depth_mm = imageio.imread(args.depth_file).astype(np.float32)
            depth = depth_mm / 1000.0
    else:
        raise RuntimeError("--depth_file is required for RGB-D mode.")
    # mask logic
    if args.mask_file:
        mask = imageio.imread(args.mask_file).astype(bool)
    else:
        mask = np.ones((H, W), dtype=bool)

    # load intrinsics
    K = load_intrinsics(args.K_file)
    K = K.cpu().numpy()

    # prepare model
    mesh = trimesh.load(args.mesh_file)
    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()

    est = FoundationPose(model_pts     = mesh.vertices,
                         model_normals = mesh.vertex_normals,
                         mesh          = mesh,
                         scorer        = scorer,
                         refiner       = refiner,
                         debug_dir     = args.save_dir,
                         debug         = args.debug,
                         glctx         = glctx)

    logging.info("Estimator initialised – running registration …")
        
    pose44 = est.register(K        = K,
                          rgb      = color_rgb,
                          depth    = depth,
                          ob_mask  = mask,
                          iteration= args.est_refine_iter)

    # append to CSV
    csv_path = os.path.join(args.save_dir, 'poses.csv')
    t = pose44[:3, 3]
    euler = R.from_matrix(pose44[:3, :3]).as_euler('xyz', degrees=False)
    row = np.concatenate([t, euler])
    header = 'x,y,z,roll_x,roll_y,roll_z\n'
    mode = 'a' if os.path.exists(csv_path) else 'w'
    with open(csv_path, mode) as f:
        if mode == 'w':
            f.write(header)
        f.write(','.join(f'{v:.6f}' for v in row) + '\n')
    logging.info(f"Appended pose to {csv_path}")

    # visual overlay & blocking window
    center_pose = pose44 @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=color_bgr, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1,
                        K=K, thickness=3, transparency=0, is_input_rgb=False)
    if vis.dtype != np.uint8:
        vis = (vis*255).clip(0,255).astype(np.uint8)
    vis = np.ascontiguousarray(vis)
    cv2.imshow('FoundationPose overlay', vis)
    print("[INFO] Close the window manually to exit …")
    # keep window open until user closes it
    while True:
        if cv2.getWindowProperty('FoundationPose overlay', cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(100) == 27:  # Esc to exit
            break
    cv2.destroyAllWindows()
