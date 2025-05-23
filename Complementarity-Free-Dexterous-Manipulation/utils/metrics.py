import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
import pickle
import csv

from utils import rotations


def comp_pos_error(curr_pos, targ_pos):
    return np.linalg.norm(curr_pos - targ_pos)


def comp_quat_error(curr_quat, targ_quat):
    return 1 - np.dot(curr_quat, targ_quat) ** 2


def comp_pos_error_traj(pos_traj, targ_pos):
    return np.linalg.norm(pos_traj - targ_pos, axis=1)


def comp_quat_error_traj(quat_traj, targ_quat):
    quat_error_traj = []
    for t in range(quat_traj.shape[0]):
        quat_error_traj.append(comp_quat_error(quat_traj[t], targ_quat))
    return np.array(quat_error_traj)


def comp_angle_error_traj(quat_traj, targ_quat):
    quat_error_traj = []
    for t in range(quat_traj.shape[0]):
        quat_error_traj.append(np.abs(rotations.quat_to_rpy(quat_traj[t])[2] - rotations.quat_to_rpy(targ_quat)[2]))
    return np.array(quat_error_traj)


def save_data(data, data_name, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        save_dir = path.join(os.getcwd(), save_dir)

    if not path.exists(save_dir):
        os.makedirs(save_dir)

    saved_path = path.join(save_dir, data_name)

    with open(saved_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def save_csv_data(matrix, data_name, headers, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()  # Use current directory if not specified
    else:
        save_dir = os.path.join(os.getcwd(), save_dir)  # Append to current directory

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create directory if it doesn't exist

    saved_path = os.path.join(save_dir, data_name)

    # Ensure the filename ends with .csv
    if not saved_path.endswith(".csv"):
        saved_path += ".csv"

    # Save the matrix with headers as a CSV file
    with open(saved_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # Write headers first
        writer.writerows(matrix)  # Write each row of the matrix

    print(f"Matrix saved to {saved_path}")

def load_data(data_name, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()

    saved_path = path.join(save_dir, data_name)

    try:
        with open(saved_path, 'rb') as f:
            data = pickle.load(f)
    except:
        assert False

    return data
