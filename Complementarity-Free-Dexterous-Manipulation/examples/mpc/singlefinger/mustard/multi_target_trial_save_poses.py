import numpy as np

from examples.mpc.singlefinger.mustard.params import ExplicitMPCParams
from planning.mpc_explicit import MPCExplicit
from envs.singlefinger_env import MjSimulator
from contact.singlefinger_collision_detection import Contact
from utils import metrics
from utils import rotations


# -------------------------------
#       loop trials
# -------------------------------
save_flag=True



trial_num = 4
success_pos_threshold = 0.02
success_quat_threshold = 0.04
consecutive_success_time_threshold = 20
max_rollout_length = 750
mpc_horizons = [8]
pos_weights = [0.05]
used_pos_weights = []
used_mpc_horizons = []

if save_flag:
    save_dir = './examples/mpc/singlefinger/mustard/save'
    prefix_data_name = 'ours_'
    save_matrix = []
    save_data = dict()
    save_row = 0
    data_name= f'mustard_multi_w_planner_singlefingerdata_:MPC:{mpc_horizons[0]},pos_weight:{pos_weights[0]}'
    headers = ['Rollout_step','Final_baseline_cost (0.5,5)','Comp_pos_error','Comp_quat_error']
    
    pose_save_dir = "/home/frank/Desktop/full-sim-manip-pipeline/data/CFDM_poses/true_poses/mustard/"
    pose_data_name = "poses"
    pose_save_matrix = []
    pose_headers = ['Rollout_step','x','y','z','qw', 'qx', 'qy', 'qz','fx','fy','fz','ux','uy','uz']
    # -------------------------------
    #        init parameters
    # -------------------------------
    
    init_height = 0.1
    init_xy_rand = 0.3 * np.random.rand(2) - 0.15
    init_angle_rand = 2 * np.pi * np.random.rand(1) - np.pi
    init_obj_quat_rand = rotations.axisangle2quat(
        np.hstack(([0, 0, 1.0], init_angle_rand))
    )
    # init_angle_rand = -0.24265831
    init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
    # init_obj_qpos_ = [ 0.97011182 -0.         -0.         -0.24265831]
rand_seeds = []
target_pose_list = []

trial_count = 0
rollout_step = 0
while trial_count < trial_num:
    for pos_weight in pos_weights:
        for mpc_horizon in mpc_horizons:

            
            change_rollout_step = rollout_step

            # rand_seed = np.random.randint(0, 101)
            # rand_seeds.append(rand_seed)
            
            # target_xy_rand = 0.25 * np.random.rand(2) - 0.12
            
            # yaw_angle = 2 * np.pi * np.random.rand(1) - np.pi
            # # Set pitch and roll to 0 for stable ground contact
            # if trial_count > -1:
            #     target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, np.pi*np.random.randint(0, 1)-np.pi/10, np.pi*np.random.randint(0, 1)+np.pi/2]))
            #     target_p_ = np.hstack([target_xy_rand, 0.035])  # Set height slightly above ground
            # else:
            #     target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))
            #     target_p_ = np.hstack([target_xy_rand, init_height])  # Set height slightly above ground
            # # target_p_ = [ 0.0689392, -0.0061667,  0.035 ]
            # # target_q_=[-0.66143117,  0.68394174,  0.17950962, -0.25001762]
            # target_pose = [target_p_,target_q_]
            # target_pose_list.append(target_pose)
            
            param = ExplicitMPCParams(init_pose = init_obj_qpos_, rand_seed=trial_count, target_type='ground-rotation',mpc_horizon=mpc_horizon,pos_weight=pos_weight)


            # -------------------------------
            #        init contact
            # -------------------------------
            contact = Contact(param)

            # -------------------------------
            #        init envs
            # -------------------------------
            env = MjSimulator(param)

            # -------------------------------
            #        init planner
            # -------------------------------
            mpc = MPCExplicit(param)

            # -------------------------------
            #        MPC rollout
            # -------------------------------
            rollout_step_trial = 0
            consecutive_success_time = 0
            stuck_threshold = 0.01  # Threshold for detecting if object is stuck
            last_obj_pos = None
            stuck_count = 0
            max_stuck_steps = 200  # Number of steps to wait before resetting

            # Get the target pose of the object
            target_pos = param.target_p_
            
            curr_q = env.get_state()
            curr_obj_pos = curr_q[0:3]  # Object position
            offset_dir = (curr_obj_pos-target_pos) 
            new_robot_pos = curr_obj_pos + 0.2*offset_dir/np.linalg.norm(offset_dir)
            new_robot_pos[2] = new_robot_pos[2] + 0.015
            env.reset_robot_position(new_robot_pos)
            stuck_count = 0
            last_obj_pos = None

            rollout_q_traj = []
            while rollout_step_trial < max_rollout_length:
                if not env.dyn_paused_:
                    # get state
                    curr_q = env.get_state()
                    rollout_q_traj.append(curr_q)

                    # Check if object is stuck 
                    curr_obj_pos = curr_q[0:3]  # Object position
                    if last_obj_pos is not None:
                        pos_diff = np.linalg.norm(curr_obj_pos - last_obj_pos)
                        if pos_diff < stuck_threshold:
                            stuck_count += 1
                        else:
                            stuck_count = 0
                    
                    last_obj_pos = curr_obj_pos

                    # Reset robot position if object is stuck
                    if stuck_count > max_stuck_steps:
                        print("Object stuck detected - resetting robot position")
                        # Get the current position of the robot and object

                        # Get the randomized direction of new_robot_pos based on the current object and robot position
                        # randomized direction
                        offset_dir = (last_obj_pos-target_pos)
                        
                        
                        # Generate new random position for robot
                        new_robot_pos = last_obj_pos + 0.2*offset_dir/np.linalg.norm(offset_dir)
                        new_robot_pos[2] = new_robot_pos[2] + 0.015
                        env.reset_robot_position(new_robot_pos)
                        stuck_count = 0
                        last_obj_pos = None
                        continue

                    # -----------------------
                    #     contact detect
                    # -----------------------
                    phi_vec, jac_mat = contact.detect_once(env)

                    # -----------------------
                    #        planning
                    # -----------------------
                    sol = mpc.plan_once(
                        param.target_p_,
                        param.target_q_,
                        curr_q,
                        phi_vec,
                        jac_mat,
                        sol_guess=param.sol_guess_)
                    param.sol_guess_ = sol['sol_guess']
                    action = sol['action']

                    # -----------------------
                    #        simulate
                    # -----------------------
                    env.step(action)
                    rollout_step = rollout_step + 1
                    rollout_step_trial = rollout_step_trial +1

                    # -----------------------
                    #        success check
                    # -----------------------
                    curr_q = env.get_state()
                    comp_pos_error = metrics.comp_pos_error(curr_q[0:3], param.target_p_)
                    comp_quat_error = metrics.comp_quat_error(curr_q[3:7], param.target_q_)
                    print(f"\n\npos error:{comp_pos_error}\nquat error:{comp_quat_error}\n\n")
                    if (metrics.comp_pos_error(curr_q[0:3], param.target_p_) < success_pos_threshold) \
                            and (metrics.comp_quat_error(curr_q[3:7], param.target_q_) < success_quat_threshold):
                        consecutive_success_time = consecutive_success_time + 1
                    else:
                        consecutive_success_time = 0
                        
                    if save_flag:
                        baseline_cost = 0.05*comp_pos_error+(1-0.05)*comp_quat_error
                        save_matrix.append([str(rollout_step), str(baseline_cost), str(comp_pos_error), str(comp_quat_error)])                
                        
                        ux, uy, uz = action  # your 3-vector control
                        pose_save_matrix.append([
                            str(rollout_step),
                            f"{curr_q[0]:.6f}", f"{curr_q[1]:.6f}", f"{curr_q[2]:.6f}",
                            f"{curr_q[3]:.6f}", f"{curr_q[4]:.6f}", f"{curr_q[5]:.6f}", f"{curr_q[6]:.6f}",
                            f"{curr_q[7]:.6f}", f"{curr_q[8]:.6f}", f"{curr_q[9]:.6f}",
                            f"{ux:.6f}",        f"{uy:.6f}",        f"{uz:.6f}",
                        ])
                        
                        save_row = save_row+1
                    
                    last_full_q = curr_q.copy()              # length–10: [obj(7), fingertip(3)]
                    last_obj_qpos = last_full_q[0:7].copy()  # take only the object qpos
                    # now update init_obj_qpos_ for the next trial
                    init_obj_qpos_ = last_obj_qpos

                    # -----------------------
                    #       early termination
                    # -----------------------
                    if consecutive_success_time > consecutive_success_time_threshold:
                        break

            # -------------------------------
            #        close viewer
            # -------------------------------
            env.viewer_.close()

            # -------------------------------
            #        save data
            # -------------------------------
            used_pos_weights.append(param.pos_weight)
            used_mpc_horizons.append(param.mpc_horizon_)
            
                # save_data.update(target_obj_pos=param.target_p_)
                # save_data.update(target_obj_quat=param.target_q_)
                # save_data.update(rollout_traj=np.array(rollout_q_traj))
                # # success index
                # if rollout_step < max_rollout_length:
                #     save_data.update(success=True)
                # else:
                #     save_data.update(success=False)
                # # save to file
                # metrics.save_data(save_data, data_name=prefix_data_name + 'trial_' + str(trial_count) + '_rollout',
                #                   save_dir=save_dir)

    trial_count = trial_count + 1
#save to csv at save_dir

print(f"rand seeds: {rand_seeds}")
print(f"target obj pose (change at rollout step {change_rollout_step})")
print(target_pose_list)

print(f"used pos weights: {used_pos_weights}")
print(f"used mpc horizons: {used_mpc_horizons}")

metrics.save_csv_data(pose_save_matrix, data_name=pose_data_name, headers=pose_headers, save_dir=pose_save_dir)

best_MPC_horizons_per_pos_weight = {}
# Now iterate through the save_matrix
for row in save_matrix:
    trial_count, rollout_step, success, final_baseline_cost, comp_pos_error, comp_quat_error, mpc_horizon, pos_weight = row
    
    # Convert values to appropriate types
    success = success == "True"  # Convert string to boolean
    final_baseline_cost = float(final_baseline_cost)
    mpc_horizon = int(mpc_horizon)
    pos_weight = float(pos_weight)

    # Only consider successful trials
    if pos_weight not in best_MPC_horizons_per_pos_weight:
        # Initialize with the first encountered value
        best_MPC_horizons_per_pos_weight[pos_weight] = (mpc_horizon, final_baseline_cost)
    else:
        # Update if a lower final_baseline_cost is found
        best_horizon, best_error = best_MPC_horizons_per_pos_weight[pos_weight]
        if final_baseline_cost < best_error:
            best_MPC_horizons_per_pos_weight[pos_weight] = (mpc_horizon, final_baseline_cost)

# Print results
print("\nBest MPC Horizons for each pos_weight (minimizing 0.5*comp_pos_error+5*comp_quat_error):")
for pos_weight, (mpc_horizon, final_baseline_cost) in sorted(best_MPC_horizons_per_pos_weight.items()):
    print(f"Position Weight: {pos_weight}, Best MPC Horizon: {mpc_horizon}, Final_cost: {final_baseline_cost}")

metrics.save_csv_data(save_matrix, data_name=data_name, headers=headers, save_dir=save_dir)


