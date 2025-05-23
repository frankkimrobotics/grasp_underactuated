import casadi as cs
import numpy as np

from utils import rotations


class ExplicitMPCParams:
    def __init__(self, rand_seed=1, target_type=True):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters 
        # ---------------------------------------------------------------------------------------------
        # UPDATED model path to single-fingertip environment (if you have a different XML file).
        self.model_path_ = 'envs/xmls/env_singlefinger_cube.xml'
        self.object_names_ = ['obj']

        self.h_ = 0.1
        self.frame_skip_ = 10

        # ---------------------- Key dimension changes for single fingertip ----------------------
        # The object has 7 qpos (3 + 4 quat) and 6 qvel (3 linear + 3 angular).
        # A single fingertip has 3 slide joints (x, y, z).
        self.n_robot_qpos_ = 3   # Single fingertip: 3 (x, y, z)
        self.n_cmd_        = 3   # 3 controls for one fingertip
        self.n_qpos_       = 10  # 7 (object) + 3 (fingertip)
        self.n_qvel_       = 9   # 6 (object) + 3 (fingertip)
        # ---------------------------------------------------------------------------------------------

        # internal joint controller for each finger
        self.jc_kp_ = 10
        self.jc_damping_ = 0.05


        np.random.seed(100 + rand_seed)

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        # random initial pose for object
        init_height = 0.03
        init_xy_rand = 0.05 * np.random.rand(2) - 0.025
        init_angle_rand = 2 * np.pi * np.random.rand(1) - np.pi
        init_obj_quat_rand = rotations.axisangle2quat(
            np.hstack(([0, 0, 1.0], init_angle_rand))
        )
        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))

        # Single fingertip initial qpos (x,y,z)
        # You can choose whatever 3D starting point you want:
        self.init_robot_qpos_ = np.array([0.2, 0.0, 0.0])  # For example

        # random target pose for object
        if target_type == 'ground-rotation':
            target_xy_rand = 0.2 * np.random.rand(2) - 0.1
            self.target_p_ = np.hstack([target_xy_rand, init_height])
            yaw_angle = 2 * np.pi * np.random.rand(1) - np.pi
            self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))

        elif target_type == 'ground-flip':
            init_height = 0.03 + 0.02
            target_xy_rand = 0.2 * np.random.rand(2) - 0.1
            self.target_p_ = np.hstack([target_xy_rand, init_height])
            yaw_angle = 2 * np.pi * np.random.rand(1) - np.pi
            pitch_angle = np.pi * np.random.rand(1) - np.pi/2
            roll_angle = np.pi * np.random.rand(1) - np.pi/2
            self.target_q_ = rotations.rpy_to_quaternion(
                np.hstack([yaw_angle, pitch_angle, roll_angle])
            )

        elif target_type == 'in-air':
            target_height = 0.05 + 0.05 * np.random.rand(1)
            target_xy_rand = 0.2 * np.random.rand(2) - 0.1
            self.target_p_ = np.hstack([target_xy_rand, target_height])

            angle = 2 * np.pi * np.random.rand(1) - np.pi
            axis = np.array([0, 1, 1]) + np.random.randn(3) * 0.1
            self.target_q_ = rotations.axisangle2quat(np.hstack([axis, angle]))

        else:
            raise ValueError('Invalid target type')

        # ---------------------------------------------------------------------------------------------
        #      contact parameters 
        # ---------------------------------------------------------------------------------------------
        self.mu_object_ = 0.5
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 8

        # ---------------------------------------------------------------------------------------------
        #      models parameters
        # ---------------------------------------------------------------------------------------------
        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3)

        # Robot stiffness is 3x3 for single fingertip:
        self.robot_stiff_ = np.diag([100.0]*self.n_cmd_)

        # Full Q: block-diagonal with object inertia (6x6) + robot stiffness (3x3) = 9x9
        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.01
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])
        self.model_params = 1

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 500
        self.mpc_model = 'explicit'

        self.mpc_u_lb_ = -0.005
        self.mpc_u_ub_ =  0.005

        # Bounds for single fingertip (3 DoF). 
        # e.g., z >= 0.0 to ensure fingertip is above ground (if that's desired).
        fts_q_lb = np.array([-100, -100, 0.0])
        fts_q_ub = np.array([ 100,  100, 100])

        # The object part is always 7 (x, y, z, quat(4))
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fts_q_lb))  # total = 10
        self.mpc_q_ub_ = np.hstack(( 1e7 * np.ones(7), fts_q_ub))  # total = 10

        self.sol_guess_ = None

    def init_cost_fns(self):
        """
        Create and return path and final cost functions for single fingertip.
        """
        x = cs.SX.sym('x', self.n_qpos_)  # 10 states
        u = cs.SX.sym('u', self.n_cmd_)   # 3 controls

        # target (object) pose: p (3) + quat (4)
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)

        # Compute basic cost terms
        #  - position_cost: object pos vs. target
        #  - quaternion_cost: object orientation vs. target
        #  - contact_cost: fingertip vs. object position
        #  - control_cost: squared magnitude of controls
        position_cost   = cs.sumsqr(x[0:3] - target_position)           
        quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion)**2      
        contact_cost    = cs.sumsqr(x[0:3] - x[7:10])                    
        control_cost    = cs.sumsqr(u)

        # You can remove or alter any advanced multi-finger terms. 
        # For single fingertip, we omit "grasp_closure" or similar.
       
        # We still define these for param completeness (if your solver uses them):
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat])

        # Weighted sum for path cost
        #   - Example weighting: contact cost + 50*control cost
        base_cost  = contact_cost
        path_cost  = base_cost + 50.0 * control_cost

        # Weighted sum for final cost
        #   - Example weighting: 500*position + 5*orientation
        final_cost = 500.0 * position_cost + 50.0 * quaternion_cost

        # Create CasADi functions
        path_cost_fn  = cs.Function('path_cost_fn',
                                    [x, u, cost_param],
                                    [path_cost])
        # Multiply the final cost by 10 if desired (matches original style):
        final_cost_fn = cs.Function('final_cost_fn',
                                    [x, cost_param],
                                    [10.0 * final_cost])

        return path_cost_fn, final_cost_fn
