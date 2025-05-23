#!/usr/bin/env python3
"""
run_save_ukf.py

Loads your FoundationPose CSV and runs an Unscented Kalman Filter
using MuJoCo for the process model.  Writes out `filtered_poses.csv`
in SAVE_DIR, printing a progress bar as it goes, and overwriting the
output file on each run.
"""

import os
import sys
import numpy as np
import pandas as pd
import mujoco
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from tqdm import tqdm
from scipy.stats import chi2     

N_SIG = 0.9
ALPHA = (N_SIG**2)/6.
KAPPA = 0.
BETA = 2.

THIS_FILE = os.path.abspath(__file__)
PIPELINE_ROOT = os.path.dirname(THIS_FILE)
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)
CFDM_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT,
                                         'Complementarity-Free-Dexterous-Manipulation'))
if CFDM_ROOT not in sys.path:
    sys.path.insert(0, CFDM_ROOT)

from envs.singlefinger_env import MjSimulator
from planning.mpc_explicit import MPCExplicit
from examples.mpc.singlefinger.mustard.params import ExplicitMPCParams

# ─── USER EDIT ────────────────────────────────────────────────────────────────
EST_CSV    = '/home/frank/Desktop/full-sim-manip-pipeline/data/FoundationPose_estimations/mustard/noisy_testing_poses.csv'
ACTION_CSV = '/home/frank/Desktop/full-sim-manip-pipeline/data/CFDM_poses/true_poses/mustard/final_testing_poses.csv'
SAVE_DIR   = '/home/frank/Desktop/full-sim-manip-pipeline/data/UKF_data'
# ───────────────────────────────────────────────────────────────────────────────

def read_estimates(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.strip('"')
    data = df[['Rollout_step','x','y','z','qx','qy','qz','qw']].to_numpy()
    qs = data[:, 4:8]
    euls = -1 * R.from_quat(qs).as_euler('xyz', degrees=False)
    euls[:,0] = np.pi - euls[:,0]
    meas = np.zeros((data.shape[0], 7))
    meas[:,0]   = data[:,0]
    meas[:,1:4] = data[:,1:4]
    meas[:,4:7] = euls
    return meas

def read_actions(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.strip('"')
    return df[['Rollout_step','fx','fy','fz','ux','uy','uz']].to_numpy()

def mj_process_fn_euler(state, dt, mujoco_sim, u):
    fx, fy, fz, ux, uy, uz = u

    # 1) backup sim
    qpos0, qvel0 = mujoco_sim.data_.qpos.copy(), mujoco_sim.data_.qvel.copy()

    # 2) set object pose
    pos = state[0:3]
    quat_xyzw = R.from_euler('xyz', state[3:6], degrees=False).as_quat()
    mujoco_sim.data_.qpos[:3]  = pos
    mujoco_sim.data_.qpos[3]     = quat_xyzw[3]   # qw
    mujoco_sim.data_.qpos[4]     = quat_xyzw[0]   # qx
    mujoco_sim.data_.qpos[5]     = quat_xyzw[1]   # qy
    mujoco_sim.data_.qpos[6]     = quat_xyzw[2]   # qz
    mujoco_sim.data_.qvel[:]   = 0.0
    mujoco.mj_forward(mujoco_sim.model_, mujoco_sim.data_)

    # 3) teleport fingertip
    mujoco_sim.reset_robot_position([fx, fy, fz])

    # 4) step with command
    mujoco_sim.step(np.array([ux, uy, uz]))

    # 5) read object pose
    new_qpos = mujoco_sim.get_state()[0:7]
    pos2     = new_qpos[0:3]
    qx2,qy2,qz2,qw2 = new_qpos[4],new_qpos[5],new_qpos[6],new_qpos[3]
    e2       = R.from_quat([qx2, qy2, qz2, qw2]).as_euler('xyz', degrees=False)
    next_state = np.concatenate([pos2, e2])

    # 6) restore sim
    mujoco_sim.data_.qpos[:] = qpos0
    mujoco_sim.data_.qvel[:] = qvel0
    mujoco.mj_forward(mujoco_sim.model_, mujoco_sim.data_)

    return next_state

def meas_fn(x):
    return x

class OriScaledSigmaPoints(MerweScaledSigmaPoints):
    def __init__(self, n, alpha, beta, kappa, ori_scale=1.0):
        super().__init__(n=n, alpha=alpha, beta=beta, kappa=kappa)
        self.ori_scale = ori_scale

    def sigma_points(self, x, P):
        # make a local copy so we don't modify ukf.P
        P2 = P.copy()
        P2[3:6, 3:6] *= self.ori_scale
        return super().sigma_points(x, P2)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    meas_data   = read_estimates(EST_CSV)
    action_data = read_actions(ACTION_CSV)

    param      = ExplicitMPCParams(target_type='ground-rotation')
    mujoco_sim = MjSimulator(param)

    dim_x = 6
    
    pts = OriScaledSigmaPoints(
    n=dim_x,
    alpha=ALPHA,
    beta=BETA,
    kappa=KAPPA,
    ori_scale=1.5
)
    # pts   = MerweScaledSigmaPoints(n=dim_x, alpha=ALPHA, beta=BETA, kappa=KAPPA)
    ukf = UnscentedKalmanFilter(
        dim_x=dim_x,
        dim_z=dim_x,
        dt=1.0,
        fx=lambda x, dt, u: mj_process_fn_euler(x, dt, mujoco_sim, u),
        hx=meas_fn,
        points=pts
    )

    # initialize
    first = meas_data[0]
    ukf.x = first[1:7].copy()
    ukf.P *= 1e-2*5
    Q = np.eye(dim_x) * 1e-2
    Q[0,0] = Q[0,0]
    Q[1,1] = Q[1,1]
    Q[2,2] = Q[2,2]
    ukf.Q  = Q
    # pos_std, ori_std = 0.01, 0.017
    R = np.eye(dim_x)
    std_R_list = [0.002,0.01,0.02,1.45*np.pi/180,1.65*np.pi/180,1.5*np.pi/180]
    for i, std in enumerate(std_R_list):  
        if i > 2:
            R[i,i] = (std**2)*10
        else:
            R[i,i] = (std**2)*5
    ukf.R  = R

    # nis_thresh = chi2.ppf(0.999999, df=dim_x)

    total      = 0
    rejected   = 0
    rows       = []

    for i, m in enumerate(tqdm(meas_data, desc='Filtering UKF', unit='step', ncols=80, ascii=True)):
        z          = m[1:7]
        finger_pos = action_data[i, 1:4]
        cmd        = action_data[i, 4:7]
        u          = np.hstack((finger_pos, cmd))

        # 1) Predict
        ukf.predict(u=u)

        # 2) Backup prior state & covariance
        x_prior = ukf.x.copy()
        P_prior = ukf.P.copy()

        # 3) Attempt update
        ukf.update(z)

        # 4) Compute NIS for this update
        y   = ukf.y     # innovation
        S   = ukf.S     # innovation covariance
        nis = float(y.T @ np.linalg.inv(S) @ y)

        total += 1
        
        pred = x_prior  # [x,y,z,roll,pitch,yaw]
        diff = z - pred
        # wrap orientation difference into [-π,π]
        diff[3:6] = (diff[3:6] + np.pi) % (2*np.pi) - np.pi
        # check thresholds
        pos_bad = np.abs(np.linalg.norm(diff[0:3]) > 0.7)
        pos_bad = False
        ori_bad = np.any(np.abs(diff[3:6]) > (2*np.pi/3))
        ori_bad = False
        # if 1==0 or nis > nis_thresh:
        if ori_bad or pos_bad:
            # reject this update—restore prior
            rejected += 1
            ukf.x = x_prior
            ukf.P = P_prior

        # 5) Record 2σ bounds & NIS
        var         = np.diag(ukf.P)
        two_sigma_p = 2.0 * np.sqrt(var[0:3])
        two_sigma_e = 2.0 * np.sqrt(var[3:6])

        rows.append(np.hstack((
            [m[0]],         # Rollout_step
            ukf.x,          # x,y,z,roll,pitch,yaw
            two_sigma_p,    # 2sig_x,y,z
            two_sigma_e,    # 2sig_roll,pitch,yaw
            [nis]           # raw NIS (even if rejected)
        )))

    # 6) Write out CSV
    cols = [
        'Rollout_step','x','y','z','roll','pitch','yaw',
        '2sig_x','2sig_y','2sig_z',
        '2sig_roll','2sig_pitch','2sig_yaw',
        'NIS'
    ]
    out = pd.DataFrame(rows, columns=cols)
    out_path = os.path.join(SAVE_DIR, 'filtered_poses.csv')
    out.to_csv(out_path, mode='w', index=False, float_format='%.6f')

    # 7) Print rejection statistics
    pct = 100 * rejected / total
    print(f"\nFiltered poses + 2σ bounds + NIS written to {out_path}")
    print(f"Rejected {rejected}/{total} measurements ({pct:.1f}%) for NIS > χ²₀.₉₅")

if __name__ == '__main__':
    main()


# #!/usr/bin/env python3
# """
# run_save_ukf.py

# Loads your FoundationPose CSV and runs an Unscented Kalman Filter
# using MuJoCo for the process model.  Writes out `filtered_poses.csv`
# in SAVE_DIR, printing a progress bar as it goes, and overwriting the
# output file on each run.
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# import mujoco
# from scipy.spatial.transform import Rotation as R
# from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
# from tqdm import tqdm
# from scipy.stats import chi2     

# N_SIG = 2.6
# ALPHA = (N_SIG**2)/6.
# KAPPA = 0.
# BETA = 2.

# THIS_FILE = os.path.abspath(__file__)
# PIPELINE_ROOT = os.path.dirname(THIS_FILE)
# if PIPELINE_ROOT not in sys.path:
#     sys.path.insert(0, PIPELINE_ROOT)
# CFDM_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT,
#                                          'Complementarity-Free-Dexterous-Manipulation'))
# if CFDM_ROOT not in sys.path:
#     sys.path.insert(0, CFDM_ROOT)

# from envs.singlefinger_env import MjSimulator
# from planning.mpc_explicit import MPCExplicit
# from examples.mpc.singlefinger.mustard.params import ExplicitMPCParams

# # ─── USER EDIT ────────────────────────────────────────────────────────────────
# EST_CSV    = '/home/frank/Desktop/full-sim-manip-pipeline/data/FoundationPose_estimations/mustard/noisy_testing_poses.csv'
# ACTION_CSV = '/home/frank/Desktop/full-sim-manip-pipeline/data/CFDM_poses/true_poses/mustard/final_testing_poses.csv'
# SAVE_DIR   = '/home/frank/Desktop/full-sim-manip-pipeline/data/UKF_data'
# # ───────────────────────────────────────────────────────────────────────────────

# def read_estimates(path):
#     df = pd.read_csv(path)
#     df.columns = df.columns.str.strip().str.strip('"')
#     data = df[['Rollout_step','x','y','z','qx','qy','qz','qw']].to_numpy()
#     qs = data[:, 4:8]
#     euls = -1 * R.from_quat(qs).as_euler('xyz', degrees=False)
#     euls[:,0] = np.pi - euls[:,0]
#     meas = np.zeros((data.shape[0], 7))
#     meas[:,0]   = data[:,0]
#     meas[:,1:4] = data[:,1:4]
#     meas[:,4:7] = euls
#     return meas

# def read_actions(path):
#     df = pd.read_csv(path)
#     df.columns = df.columns.str.strip().str.strip('"')
#     return df[['Rollout_step','fx','fy','fz','ux','uy','uz']].to_numpy()

# def mj_process_fn_euler(state, dt, mujoco_sim, u):
#     fx, fy, fz, ux, uy, uz = u

#     # 1) backup sim
#     qpos0, qvel0 = mujoco_sim.data_.qpos.copy(), mujoco_sim.data_.qvel.copy()

#     # 2) set object pose
#     pos = state[0:3]
#     quat_xyzw = R.from_euler('xyz', state[3:6], degrees=False).as_quat()
#     mujoco_sim.data_.qpos[:3]  = pos
#     mujoco_sim.data_.qpos[3]     = quat_xyzw[3]   # qw
#     mujoco_sim.data_.qpos[4]     = quat_xyzw[0]   # qx
#     mujoco_sim.data_.qpos[5]     = quat_xyzw[1]   # qy
#     mujoco_sim.data_.qpos[6]     = quat_xyzw[2]   # qz
#     mujoco_sim.data_.qvel[:]   = 0.0
#     mujoco.mj_forward(mujoco_sim.model_, mujoco_sim.data_)

#     # 3) teleport fingertip
#     mujoco_sim.reset_robot_position([fx, fy, fz])

#     # 4) step with command
#     mujoco_sim.step(np.array([ux, uy, uz]))

#     # 5) read object pose
#     new_qpos = mujoco_sim.get_state()[0:7]
#     pos2     = new_qpos[0:3]
#     qx2,qy2,qz2,qw2 = new_qpos[4],new_qpos[5],new_qpos[6],new_qpos[3]
#     e2       = R.from_quat([qx2, qy2, qz2, qw2]).as_euler('xyz', degrees=False)
#     next_state = np.concatenate([pos2, e2])

#     # 6) restore sim
#     mujoco_sim.data_.qpos[:] = qpos0
#     mujoco_sim.data_.qvel[:] = qvel0
#     mujoco.mj_forward(mujoco_sim.model_, mujoco_sim.data_)

#     return next_state

# def meas_fn(x):
#     return x

# class OriScaledSigmaPoints(MerweScaledSigmaPoints):
#     def __init__(self, n, alpha, beta, kappa, ori_scale=1.0):
#         super().__init__(n=n, alpha=alpha, beta=beta, kappa=kappa)
#         self.ori_scale = ori_scale

#     def sigma_points(self, x, P):
#         # make a local copy so we don't modify ukf.P
#         P2 = P.copy()
#         P2[3:6, 3:6] *= self.ori_scale
#         return super().sigma_points(x, P2)

# def main():
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     meas_data   = read_estimates(EST_CSV)
#     action_data = read_actions(ACTION_CSV)

#     param      = ExplicitMPCParams(target_type='ground-rotation')
#     mujoco_sim = MjSimulator(param)

#     dim_x = 6
    
#     pts = OriScaledSigmaPoints(
#     n=dim_x,
#     alpha=ALPHA,
#     beta=BETA,
#     kappa=KAPPA,
#     ori_scale=15
# )
#     # pts   = MerweScaledSigmaPoints(n=dim_x, alpha=ALPHA, beta=BETA, kappa=KAPPA)
#     ukf = UnscentedKalmanFilter(
#         dim_x=dim_x,
#         dim_z=dim_x,
#         dt=1.0,
#         fx=lambda x, dt, u: mj_process_fn_euler(x, dt, mujoco_sim, u),
#         hx=meas_fn,
#         points=pts
#     )

#     # initialize
#     first = meas_data[0]
#     ukf.x = first[1:7].copy()
#     ukf.P *= 1e-3
#     Q = np.eye(dim_x) * 1e-4/10
#     Q[0,0] = Q[0,0]/2
#     Q[1,1] = Q[1,1]/2
#     Q[2,2] = Q[2,2]/2
#     ukf.Q  = Q
#     # pos_std, ori_std = 0.01, 0.017
#     R = np.eye(dim_x)
#     std_R_list = [0.002,0.01,0.007,1.45*np.pi/180,1.65*np.pi/180,1.5*np.pi/180]
#     for i, std in enumerate(std_R_list):  
#         if i > 2:
#             R[i,i] = (std**2)/1.1
#         else:
#             R[i,i] = (std**2)/1.5
#     ukf.R  = R

#     # nis_thresh = chi2.ppf(0.999999, df=dim_x)

#     total      = 0
#     rejected   = 0
#     rows       = []

#     for i, m in enumerate(tqdm(meas_data, desc='Filtering UKF', unit='step', ncols=80, ascii=True)):
#         z          = m[1:7]
#         finger_pos = action_data[i, 1:4]
#         cmd        = action_data[i, 4:7]
#         u          = np.hstack((finger_pos, cmd))

#         # 1) Predict
#         ukf.predict(u=u)

#         # 2) Backup prior state & covariance
#         x_prior = ukf.x.copy()
#         P_prior = ukf.P.copy()

#         # 3) Attempt update
#         ukf.update(z)

#         # 4) Compute NIS for this update
#         y   = ukf.y     # innovation
#         S   = ukf.S     # innovation covariance
#         nis = float(y.T @ np.linalg.inv(S) @ y)

#         total += 1
        
#         pred = x_prior  # [x,y,z,roll,pitch,yaw]
#         diff = z - pred
#         # wrap orientation difference into [-π,π]
#         diff[3:6] = (diff[3:6] + np.pi) % (2*np.pi) - np.pi
#         # check thresholds
#         pos_bad = np.abs(np.linalg.norm(diff[0:3]) > 0.7)
#         pos_bad = False
#         ori_bad = np.any(np.abs(diff[3:6]) > (2*np.pi/3))
#         ori_bad = False
#         # if 1==0 or nis > nis_thresh:
#         if ori_bad or pos_bad:
#             # reject this update—restore prior
#             rejected += 1
#             ukf.x = x_prior
#             ukf.P = P_prior

#         # 5) Record 2σ bounds & NIS
#         var         = np.diag(ukf.P)
#         two_sigma_p = 2.0 * np.sqrt(var[0:3])
#         two_sigma_e = 2.0 * np.sqrt(var[3:6])

#         rows.append(np.hstack((
#             [m[0]],         # Rollout_step
#             ukf.x,          # x,y,z,roll,pitch,yaw
#             two_sigma_p,    # 2sig_x,y,z
#             two_sigma_e,    # 2sig_roll,pitch,yaw
#             [nis]           # raw NIS (even if rejected)
#         )))

#     # 6) Write out CSV
#     cols = [
#         'Rollout_step','x','y','z','roll','pitch','yaw',
#         '2sig_x','2sig_y','2sig_z',
#         '2sig_roll','2sig_pitch','2sig_yaw',
#         'NIS'
#     ]
#     out = pd.DataFrame(rows, columns=cols)
#     out_path = os.path.join(SAVE_DIR, 'filtered_poses.csv')
#     out.to_csv(out_path, mode='w', index=False, float_format='%.6f')

#     # 7) Print rejection statistics
#     pct = 100 * rejected / total
#     print(f"\nFiltered poses + 2σ bounds + NIS written to {out_path}")
#     print(f"Rejected {rejected}/{total} measurements ({pct:.1f}%) for NIS > χ²₀.₉₅")

# if __name__ == '__main__':
#     main()