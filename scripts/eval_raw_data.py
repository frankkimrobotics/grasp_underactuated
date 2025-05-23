#!/usr/bin/env python3
# === User-editable paths ===
TRUE_CSV      = '/home/frank/Desktop/full-sim-manip-pipeline/data/CFDM_poses/true_poses/mustard/final_testing_poses.csv'
EST_CSV       = '/home/frank/Desktop/full-sim-manip-pipeline/data/FoundationPose_estimations/mustard/noisy_testing_poses.csv'
PLOT_SAVE_DIR = '/home/frank/Desktop/full-sim-manip-pipeline/data/evaluation/testing_noisy_raw_est_vs_true'

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image
import numpy as np
START_RMSE = 0

def load_csv(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = df.columns.str.strip().str.strip('"')
    return df

def plot_positions(df_true, df_est, out_path):
    steps_true = df_true['Rollout_step'].to_numpy().flatten()
    steps_est  = df_est ['Rollout_step'].to_numpy().flatten()
    plt.figure()
    for ax in ['x','y','z']:
        vals_t = df_true[ax].to_numpy().flatten()
        vals_e = df_est [ax].to_numpy().flatten()
        plt.plot(steps_true, vals_t, label=f'true {ax}')
        plt.plot(steps_est,  vals_e, label=f'est {ax}')
        rmse = np.sqrt(np.mean((vals_e[START_RMSE:] - vals_t[START_RMSE:])**2))
        print(f"RMSE {ax}: {rmse:.6f} m")
    plt.xlabel('Rollout step')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_orientations(df_true, df_est, out_path):
    qt = df_true[['qx','qy','qz','qw']].to_numpy()
    qe = df_est [['qx','qy','qz','qw']].to_numpy()
    e_t = R.from_quat(qt).as_euler('xyz', degrees=True)
    e_e = -1*R.from_quat(qe).as_euler('xyz', degrees=True)
    e_e[:,0] = 2*90 - e_e[:,0]
    steps_true = df_true['Rollout_step'].to_numpy().flatten()
    steps_est  = df_est ['Rollout_step'].to_numpy().flatten()

    plt.figure()
    for i,angle in enumerate(['roll','pitch','yaw']):
        plt.plot(steps_true, e_t[:,i],    label=f'true {angle}')
        plt.plot(steps_est,  e_e[:,i], label=f'est {angle}')
        rmse_ang = np.sqrt(np.mean((e_e[START_RMSE:,i] - e_t[START_RMSE:,i])**2))
        print(f"RMSE {angle}: {rmse_ang:.2f}°")
    plt.xlabel('Rollout step')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_quaternions(df_true, df_est, out_path):
    steps_true = df_true['Rollout_step'].to_numpy().flatten()
    steps_est  = df_est ['Rollout_step'].to_numpy().flatten()
    plt.figure(figsize=(8,6))
    for comp in ['qw','qx','qy','qz']:
        qt = df_true[comp].to_numpy().flatten()
        qe = df_est [comp].to_numpy().flatten()
        plt.plot(steps_true, qt, label=f'true {comp}')
        plt.plot(steps_est,  qe, '--', label=f'est {comp}')
    plt.xlabel('Rollout step')
    plt.ylabel('Quaternion component value')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
def plot_error_plots(df_true, df_est, out_dir, start_rmse=70):
    """
    Generates 6 plots:
      - x_error.png, y_error.png, z_error.png   (meters)
      - roll_error.png, pitch_error.png, yaw_error.png  (degrees)
    """
    os.makedirs(out_dir, exist_ok=True)
    steps = df_true['Rollout_step'].to_numpy()

    # 1) Position errors
    pos_true = df_true[['x','y','z']].to_numpy()
    pos_est  = df_est [['x','y','z']].to_numpy()
    pos_err  = pos_est - pos_true

    for i, ax in enumerate(['x','y','z']):
        plt.figure()
        plt.plot(steps, pos_err[:, i], label=f'{ax} error')
        rmse = np.sqrt(np.mean((pos_err[start_rmse:, i])**2))
        plt.title(f'{ax.upper()} Error over Time (RMSE={rmse:.3f} m)')
        plt.xlabel('Rollout Step')
        plt.ylabel(f'{ax} error (m)')
        plt.grid(True)
        outf = os.path.join(out_dir, f'error_{ax}.png')
        plt.tight_layout()
        plt.savefig(outf)
        plt.close()

    # 2) Orientation errors (roll, pitch, yaw)
    qt = df_true[['qx','qy','qz','qw']].to_numpy()
    qe = df_est [['qx','qy','qz','qw']].to_numpy()

    e_t = R.from_quat(qt).as_euler('xyz', degrees=True)
    e_e = -R.from_quat(qe).as_euler('xyz', degrees=True)
    # apply same “flip” you had for roll
    e_e[:,0] = 180 - e_e[:,0]

    # wrap to [-180,180]
    ori_err = ( (e_e - e_t) + 180 ) % 360 - 180

    for i, angle in enumerate(['roll','pitch','yaw']):
        plt.figure()
        plt.plot(steps, ori_err[:, i], label=f'{angle} error')
        rmse_ang = np.sqrt(np.mean((ori_err[start_rmse:, i])**2))
        plt.title(f'{angle.capitalize()} Error over Time (RMSE={rmse_ang:.2f}°)')
        plt.xlabel('Rollout Step')
        plt.ylabel(f'{angle} error (°)')
        plt.grid(True)
        outf = os.path.join(out_dir, f'error_{angle}.png')
        plt.tight_layout()
        plt.savefig(outf)
        plt.close()


def main():
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    df_true = load_csv(TRUE_CSV)
    df_est  = load_csv(EST_CSV)
    if len(df_true) > len(df_est):
        df_true = df_true.iloc[:len(df_est)]
    elif len(df_est) > len(df_true):
        df_est  = df_est.iloc[:len(df_true)]
    print(df_true[['x','y','z']].to_numpy().shape)
    print(df_est[['x','y','z']].to_numpy().shape)

    pos_plot  = os.path.join(PLOT_SAVE_DIR, 'position_comparison.png')
    ori_plot  = os.path.join(PLOT_SAVE_DIR, 'orientation_comparison.png')
    quat_plot = os.path.join(PLOT_SAVE_DIR, 'quaternion_comparison.png')
    
    
   # extract quaternions (must be in [qx, qy, qz, qw] order)
    qt = df_true[['qx','qy','qz','qw']].to_numpy()
    qe = df_est [['qx','qy','qz','qw']].to_numpy()

    # convert to Euler angles (degrees)
    e_t = R.from_quat(qt).as_euler('xyz', degrees=True)
    e_e = -R.from_quat(qe).as_euler('xyz', degrees=True)
    # apply same “roll flip” you used elsewhere
    e_e[:,0] = 180 - e_e[:,0]

    # wrap error into [-180,180]
    ori_err = ((e_e - e_t + 180) % 360) - 180
    mean_ori_err = ori_err.mean(axis=0)

    print(f"Systematic bias (mean error): "
          f"roll = {mean_ori_err[0]:.2f}°, "
          f"pitch = {mean_ori_err[1]:.2f}°, "
          f"yaw = {mean_ori_err[2]:.2f}°")
      # compute and print systematic (mean) bias in x,y,z
    err = df_est[['x','y','z']].to_numpy() - df_true[['x','y','z']].to_numpy()
    mean_err = err.mean(axis=0)
    print(f"Systematic bias (mean error): x = {mean_err[0]:.4f} m, y = {mean_err[1]:.4f} m, z = {mean_err[2]:.4f} m")
    
    plot_positions(df_true,  df_est,  pos_plot)
    plot_orientations(df_true, df_est,  ori_plot)
    plot_quaternions(df_true, df_est, quat_plot)
    plot_error_plots(df_true, df_est, PLOT_SAVE_DIR, start_rmse=START_RMSE)

    # pop them up
    # for p in (pos_plot, ori_plot, quat_plot):
    #     Image.open(p).show()

if __name__ == '__main__':
    main()
