#!/usr/bin/env python3
# === User-editable paths ===
TRUE_CSV      = '/home/frank/Desktop/full-sim-manip-pipeline/data/CFDM_poses/true_poses/mustard/final_tuning_poses.csv'
UKF_CSV       = '/home/frank/Desktop/full-sim-manip-pipeline/data/UKF_data/filtered_tuning_poses.csv'
PLOT_SAVE_DIR = '/home/frank/Desktop/full-sim-manip-pipeline/data/evaluation/testing_ukf_vs_true'

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.stats import chi2

START_RMSE = 200

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.strip('"')
    return df

def plot_nis(df_ukf, out_dir, window=20, m=6, alpha=0.05):
    """
    Plots raw NIS and its moving average over `window` steps,
    with two-sided 95% chi²-based bounds for NIS averaged over window*m dof.
    """
    os.makedirs(out_dir, exist_ok=True)
    steps   = df_ukf['Rollout_step'].to_numpy()
    nis_raw = df_ukf['NIS'].to_numpy()

    # moving average
    kernel = np.ones(window) / window
    nis_ma = np.convolve(nis_raw, kernel, mode='valid')
    steps_ma = steps[window-1:]  # align start of MA

    # bounds
    df = window * m
    lower = chi2.ppf(alpha/2,   df=df) / window
    upper = chi2.ppf(1-alpha/2, df=df) / window

    plt.figure()
    # plt.plot(steps,     nis_raw, ':', label='raw NIS')
    plt.plot(steps_ma, nis_ma,  '-', label=f'{window}-step NIS MA')
    plt.hlines([lower, upper],
               xmin=steps[0], xmax=steps[-1],
               linestyles='--', colors='r',
               label='95% bounds')
    plt.xlabel('Rollout Step')
    plt.ylabel('Normalized Innovation Squared')
    plt.title(f'UKF NIS with {window}-step MA & 95% Bounds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'total_nis_with_bounds.png'))
    plt.close()
    
    plt.figure()
    # plt.plot(steps[100:],     nis_raw[100:], ':', label='raw NIS')
    plt.plot(steps_ma[200-(window-1):], nis_ma[200-(window-1):],  '-', label=f'{window}-step NIS MA')
    plt.hlines([lower, upper],
               xmin=steps[0], xmax=steps[-1],
               linestyles='--', colors='r',
               label='95% bounds')
    plt.xlabel('Rollout Step')
    plt.ylabel('Normalized Innovation Squared')
    plt.title(f'UKF NIS with {window}-step MA & 95% Bounds (after {200} steps)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'nis_with_bounds_after_calibrating.png'))
    plt.close()

def plot_positions(true, ukf, out_path):
    # ← FIXED: steps as 1D numpy array
    steps = true['Rollout_step'].to_numpy()
    plt.figure()
    for ax in ['x','y','z']:
        y_true = true[ax].to_numpy()
        y_ukf  = ukf [ax].to_numpy()
        plt.plot(steps, y_true, label=f'true {ax}')
        plt.plot(steps, y_ukf, label=f'UKF {ax}')
    plt.xlabel('Rollout step')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_orientations(true, ukf, out_path):
    # true: quaternion → euler
    qt = true[['qx','qy','qz','qw']].to_numpy()
    e_t = R.from_quat(qt).as_euler('xyz', degrees=False)
    # ukf: already has roll,pitch,yaw in degrees
    e_u = ukf[['roll','pitch','yaw']].to_numpy()

    # ← FIXED: steps as numpy array
    steps = true['Rollout_step'].to_numpy()
    plt.figure()
    for i, angle in enumerate(['roll','pitch','yaw']):
        plt.plot(steps, e_t[:,i],  label=f'true {angle}')
        plt.plot(steps, e_u[:,i], label=f'UKF {angle}')
    plt.xlabel('Rollout step')
    plt.ylabel('Angle (°)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_error_plots(true, ukf, out_dir, start_rmse=START_RMSE):
    os.makedirs(out_dir, exist_ok=True)
    steps = true['Rollout_step'].to_numpy()

    # 1) Position errors + ±2σ
    for ax in ['x','y','z']:
        err   = ukf[ax].to_numpy() - true[ax].to_numpy()
        twoσ  = ukf[f'2sig_{ax}'].to_numpy()
        rmse  = np.sqrt(np.mean((err[start_rmse:])**2))
        print(f"RMSE for {ax}: {rmse}")
        plt.figure()
        plt.plot(steps, err, '-', label=f'{ax} error (RMSE={rmse:.3f} m)')
        plt.scatter(steps, err, s=5, alpha=0.6)
        plt.plot(steps, twoσ, 'r--', label='±2σ bound')
        plt.plot(steps, -twoσ, 'r--')
        plt.title(f'{ax.upper()} Error over Time')
        plt.xlabel('Rollout Step')
        plt.ylabel(f'{ax} error (m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'error_{ax}.png'))
        plt.close()

    # 2) Orientation errors + ±2σ
    qt = true[['qx','qy','qz','qw']].to_numpy()
    e_t = R.from_quat(qt).as_euler('xyz', degrees=True)
    e_u = 180*ukf[['roll','pitch','yaw']].to_numpy()/np.pi
    err = ((e_u - e_t + 180) % 360) - 180

    for i, angle in enumerate(['roll','pitch','yaw']):
        twoσ = 180*ukf[f'2sig_{angle}'].to_numpy()/np.pi
        rmse = np.sqrt(np.mean((err[start_rmse:,i])**2))
        print(f"RMSE for {angle}: {rmse}")
        plt.figure()
        plt.plot(steps, err[:,i], '-', label=f'{angle} error (RMSE={rmse:.2f}°)')
        plt.scatter(steps, err[:,i], s=5, alpha=0.6)
        plt.plot(steps, twoσ, 'r--', label='±2σ bound')
        plt.plot(steps, -twoσ, 'r--')
        plt.title(f'{angle.capitalize()} Error over Time')
        plt.xlabel('Rollout Step')
        plt.ylabel(f'{angle} error (°)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'error_{angle}.png'))
        plt.close()

def main():
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    df_true = load_csv(TRUE_CSV)
    df_ukf  = load_csv(UKF_CSV)

    # trim to same length
    N = min(len(df_true), len(df_ukf))
    df_true = df_true.iloc[:N].reset_index(drop=True)
    df_ukf  = df_ukf.iloc[:N].reset_index(drop=True)

    # comparison plots
    plot_positions(df_true, df_ukf, os.path.join(PLOT_SAVE_DIR, 'position_comparison.png'))
    plot_orientations(df_true, df_ukf, os.path.join(PLOT_SAVE_DIR, 'orientation_comparison.png'))

    # error + 2σ plots
    plot_error_plots(df_true, df_ukf, PLOT_SAVE_DIR, start_rmse=START_RMSE)
    
    plot_nis(df_ukf, PLOT_SAVE_DIR, window=100, m=6, alpha=0.05)

if __name__ == '__main__':
    main()
