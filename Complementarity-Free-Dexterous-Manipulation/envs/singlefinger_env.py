import mujoco
import mujoco.viewer

import numpy as np

from utils import rotations
#singlefinger_env

class MjSimulator():
    def __init__(self, param):
        self.param_ = param

        # init model data
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)
        self.ft_names_ = ['fingertip_0']

        self.break_out_signal_ = False
        self.dyn_paused_ = False

        self.set_goal(self.param_.target_p_, self.param_.target_q_)
        self.reset_mj_env()

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)

    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True

    def reset_mj_env(self):
        self.data_.qpos[:] = np.copy(np.concatenate((self.param_.init_obj_qpos_, self.param_.init_robot_qpos_)))
        self.data_.qvel[:] = np.copy(np.array(self.param_.n_qvel_ * [0]))
        mujoco.mj_forward(self.model_, self.data_)

    def step(self, cmd):
        finger_jpos = self.get_finger_jpos()
        target_jpos = (finger_jpos + cmd).copy()

        # run the OCS controller
        for _ in range(self.param_.frame_skip_):
            e_jpos = target_jpos - self.get_finger_jpos()
            e_jvel = self.data_.qvel[6:]
            torque = self.param_.jc_kp_ * (e_jpos) - self.param_.jc_damping_ * e_jvel
            self.data_.ctrl[:] = torque + self.data_.qfrc_bias[6:]
            mujoco.mj_step(self.model_, self.data_, nstep=1)
            self.viewer_.sync()
            # print('error_f0 = ', np.linalg.norm(target_jpos[0:3] - self.get_finger_jpos()[0:3]))
            # print('error_f1 = ', np.linalg.norm(target_jpos[3:6] - self.get_finger_jpos()[3:6]))
            # print('error_f2 = ', np.linalg.norm(target_jpos[6:] - self.get_finger_jpos()[6:]))

        mujoco.mj_forward(self.model_, self.data_)

    def get_fingertip_position(self):
        fts_pos = []
        for ft_name in self.ft_names_:
            fts_pos.append(self.data_.site(ft_name).xpos)
        return np.concatenate(fts_pos).flatten().copy()

    def get_state(self):
        mujoco.mj_forward(self.model_, self.data_)
        return self.data_.qpos.flatten().copy()

    def get_finger_jpos(self):
        mujoco.mj_forward(self.model_, self.data_)
        return self.data_.qpos.flatten()[7:].copy()

    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model_.body('goal').pos = goal_pos
        if goal_quat is not None:
            self.model_.body('goal').quat = goal_quat
        mujoco.mj_forward(self.model_, self.data_)
        pass

    def reset_robot_position(self, new_pos):
        """
        Reset the robot's position while keeping the object's state unchanged.
        
        Args:
            new_pos: New position for the robot (x, y, z)
        """
        # Get current object state
        curr_obj_qpos = self.data_.qpos[:7].copy()  # Object position and quaternion
        
        # Set new robot position
        self.data_.qpos[7:] = new_pos  # Set new robot position
        self.data_.qvel[6:] = np.zeros(3)  # Reset robot velocity
        
        # Keep object state unchanged
        self.data_.qpos[:7] = curr_obj_qpos
        
        # Forward dynamics
        mujoco.mj_forward(self.model_, self.data_)
        self.viewer_.sync()
