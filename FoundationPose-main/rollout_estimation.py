# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader_custom import *
import argparse
from scipy.spatial.transform import Rotation as R
import logging
import numpy as np
import os

# these came from your observations
Y_OFFSET = -0.8589171*2-0.0815
Z_OFFSET =  1.05920809789*2 - 0.1178
USE_DEPTH = True

def camera_to_world(pose_cam):
    T_co = np.eye(4)
    T_co[:3,:3] = pose_cam[:3,:3]
    T_co[:3, 3] = pose_cam[:3, 3]
    cam_euler = (40,0,0)
    cam_loc   = np.array([0.0, -0.9, 1.0])
    R_wc = R.from_euler('XYZ', cam_euler, degrees=True).as_matrix()
    T_wc = np.eye(4)
    T_wc[:3,:3], T_wc[:3,3] = R_wc, cam_loc
    T_wo = T_wc.dot(T_co)
    t     = T_wo[:3,3]
    t[1] = Y_OFFSET - t[1]   # flip Y around Y_CENTER
    t[2] = Z_OFFSET - t[2]   # flip Z around Z_CENTER
    q_xyzw = R.from_matrix(T_wo[:3,:3]).as_quat()  # [x,y,z,w]
    q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return t, q_wxyz

if __name__=='__main__':
  torch.cuda.empty_cache()
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--save_dir', default = "/home/frank/Desktop/full-sim-manip-pipeline/data/FoundationPose_estimations/temp_poses")
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()
  
  csv_path = os.path.join(args.save_dir, 'poses.csv')
  if os.path.exists(csv_path):
    os.remove(csv_path)

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(0, len(reader.color_files), 1):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    if USE_DEPTH:
      depth = reader.get_depth(i)
    else:
      depth = np.ones((color.shape[0], color.shape[1]), dtype=np.float32) * 2.0
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3 and USE_DEPTH:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    
    # csv_path = os.path.join(args.save_dir, 'poses.csv')
    # t = pose[:3, 3]
    # quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat()
    # quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    # # euler = R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=False)
    # row = np.concatenate([[i+1],t, quat_wxyz])
    # header = 'Rollout_step,x,y,z,qw,qx,qy,qz\n'
    # mode = 'a' if os.path.exists(csv_path) else 'w'
    # with open(csv_path, mode) as f:
    #     if mode == 'w':
    #         f.write(header)
    #     f.write(','.join(f'{v:.6f}' for v in row) + '\n')
    # logging.info(f"Appended pose to {csv_path}")
    
    # with adjustments based on blender setup to line up with mujoco

    
    # write header once
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('Rollout_step,x,y,z,qw,qx,qy,qz\n')

    # convert
    t_world, q_wxyz = camera_to_world(pose)

    # pack row
    row = [i+1,
          t_world[0], t_world[1], t_world[2],
          q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]]

    # append
    with open(csv_path, 'a') as f:
      f.write(','.join(f'{v:.6f}' for v in row) + '\n')


    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

