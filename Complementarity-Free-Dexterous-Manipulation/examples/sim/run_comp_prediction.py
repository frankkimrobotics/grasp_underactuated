import numpy as np
import time

from examples.sim.boxes_pushing.boxes_pushing_sim import MjDynViewer
from examples.sim.boxes_pushing.boxes_pushing_collision_detection import Contact
from examples.sim.boxes_pushing.boxes_pushing_explicit_model import ExplicitModel
from examples.sim.boxes_pushing.boxes_pushing_qp_model import QPModel


class Param:
    def __init__(self):
        self.model_path_ = './examples/sim/boxes_pushing/boxes_pushing_10.xml'
        self.n_obj_ = 10

        self.mu_object_ = 0.5

        # time stepping interval
        self.h_ = 0.02

        # dimensions
        self.n_v_obj_ = self.n_obj_ * 6
        self.n_qvel_ = 6 * self.n_obj_ + 1
        self.n_qpos_ = 7 * self.n_obj_ + 1
        self.n_cmd_ = 1
        self.n_mj_v_ = self.n_qvel_
        self.n_robot_qpos = 1
        self.max_ncon_ = 200

        # model hyperparameter K
        self.model_params_ = 0.1

        self.obj_mass_ = 0.001
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        # robot stiffness (equ. 5)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [500])

        # regularized object mass and inertia (equ. 5)
        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 100 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.42 * np.eye(3)

        # assemble the Q matrix (equ. 5)
        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        for i in range(self.n_obj_):
            Q[6 * i:6 * i + 6, 6 * i:6 * i + 6] = self.obj_inertia_
        Q[-1, -1] = self.robot_stiff_
        self.Q = Q


# initialize the parameter
param = Param()

# initialize the simulator (note this is not env)
sim = MjDynViewer(param)

# initialize the contact detection
contact = Contact(param)

# initialize the scene
sim.init_qpos()

# create a predictive model object (ours and QP)
explicit_model = ExplicitModel(param)
qp_model = QPModel(param)

# -------------------------------
#           Main Loop
# -------------------------------
curr_iter = 0
while True:
    # get last state
    curr_q = sim.get_qpos()

    # contact detection
    phi_results, jac_results = contact.detect_once(curr_q, sim)

    # generate command
    cmd = -0.001

    # time stepping for the proposed model, QP model, and mujoco model
    st = time.time()
    # curr_q = explicit_model.step(curr_q, cmd, phi_results, jac_results, param.model_params_)
    curr_q = qp_model.step(curr_q, cmd, phi_results, jac_results)
    # curr_q = sim.step(cmd)
    et = time.time()
    print('solving time:', time.time() - st)

    # rendering visualization
    sim.rendering(curr_q)

    curr_iter = curr_iter + 1
    if curr_iter == 800:
        while True:
            sim.rendering(curr_q)
