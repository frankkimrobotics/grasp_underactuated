# import numpy as np

# from examples.mpc.singlefinger.bunny.params import ExplicitMPCParams
# from planning.mpc_explicit import MPCExplicit
# from envs.singlefinger_env import MjSimulator
# from contact.singlefinger_collision_detection import Contact
# from utils import metrics

# # -------------------------------
# #       loop trials
# # -------------------------------
# save_flag=True
# if save_flag:
#     save_dir = './examples/mpc/singlefinger/bunny/save/'
#     prefix_data_name = 'ours_'
#     save_data = dict()

# trial_num = 20
# success_pos_threshold = 0.005
# success_quat_threshold = 0.04
# consecutive_success_time_threshold = 20
# max_rollout_length = 1000

# trial_count = 0
# while trial_count < trial_num:

#     # -------------------------------
#     #        init parameters
#     # -------------------------------
#     param = ExplicitMPCParams(rand_seed=trial_count, target_type='ground-flip')

#     # -------------------------------
#     #        init contact
#     # -------------------------------
#     contact = Contact(param)

#     # -------------------------------
#     #        init envs
#     # -------------------------------
#     env = MjSimulator(param)

#     # -------------------------------
#     #        init planner
#     # -------------------------------
#     mpc = MPCExplicit(param)

#     # -------------------------------
#     #        MPC rollout
#     # -------------------------------
#     rollout_step = 0
#     consecutive_success_time = 0

#     rollout_q_traj = []
#     while rollout_step < max_rollout_length:
#         if not env.dyn_paused_:
#             # get state
#             curr_q = env.get_state()
#             rollout_q_traj.append(curr_q)

#             # -----------------------
#             #     contact detect
#             # -----------------------
#             phi_vec, jac_mat = contact.detect_once(env)

#             # -----------------------
#             #        planning
#             # -----------------------
#             sol = mpc.plan_once(
#                 param.target_p_,
#                 param.target_q_,
#                 curr_q,
#                 phi_vec,
#                 jac_mat,
#                 sol_guess=param.sol_guess_)
#             param.sol_guess_ = sol['sol_guess']
#             action = sol['action']

#             # -----------------------
#             #        simulate
#             # -----------------------
#             env.step(action)
#             rollout_step = rollout_step + 1

#             # -----------------------
#             #        success check
#             # -----------------------
#             curr_q = env.get_state()
#             if (metrics.comp_pos_error(curr_q[0:3], param.target_p_) < success_pos_threshold) \
#                     and (metrics.comp_quat_error(curr_q[3:7], param.target_q_) < success_quat_threshold):
#                 consecutive_success_time = consecutive_success_time + 1
#             else:
#                 consecutive_success_time = 0

#             # -----------------------
#             #       early termination
#             # -----------------------
#             if consecutive_success_time > consecutive_success_time_threshold:
#                 break

#     # -------------------------------
#     #        close viewer
#     # -------------------------------
#     env.viewer_.close()

#     # -------------------------------
#     #        save data
#     # -------------------------------
#     if save_flag:
#         # save
#         save_data.update(target_obj_pos=param.target_p_)
#         save_data.update(target_obj_quat=param.target_q_)
#         save_data.update(rollout_traj=np.array(rollout_q_traj))
#         # success index
#         if rollout_step < max_rollout_length:
#             save_data.update(success=True)
#         else:
#             save_data.update(success=False)
#         # save to file
#         metrics.save_data(save_data, data_name=prefix_data_name + 'trial_' + str(trial_count) + '_rollout',
#                           save_dir=save_dir)

#     trial_count = trial_count + 1
import numpy as np

from examples.mpc.singlefinger.bunny.params import ExplicitMPCParams

from planning.mpc_explicit import MPCExplicit
from planning.mpc_implicit import MPCImplicit
from envs.singlefinger_env import MjSimulator
from contact.singlefinger_collision_detection import Contact
from utils import metrics

# -------------------------------
#       loop trials
# -------------------------------
save_flag=True
if save_flag:
    save_dir = './examples/mpc/singlefinger/bunny/save/'
    prefix_data_name = 'ours_'
    save_matrix = []
    save_data = dict()
    save_row = 0
    data_name= 'bunny_singlefingerdata_varying:MPC,pos_weight,trials'
    headers = ['Trial_num','Total_rollout_steps','Success','Final_baseline_cost (0.5,5)','Comp_pos_error','Comp_quat_error','Mpc_horizon','pos_weight']


trial_num = 20
success_pos_threshold = 0.02
success_quat_threshold = 0.04
consecutive_success_time_threshold = 20
max_rollout_length = 1200
mpc_horizons = [2,4,6,8,10,12,15,20]
pos_weights = [0.8,0.1,0.01,0.005,0.001,0.0001]
used_pos_weights = []
used_mpc_horizons = []


trial_count = 0
while trial_count < trial_num:
    for pos_weight in pos_weights:
        for mpc_horizon in mpc_horizons:

            # -------------------------------
            #        init parameters
            # -------------------------------
            param = ExplicitMPCParams(rand_seed=trial_count, target_type='ground-flip')


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
            rollout_step = 0
            consecutive_success_time = 0
            stuck_threshold = 0.001  # Threshold for detecting if object is stuck
            last_obj_pos = None
            stuck_count = 0
            max_stuck_steps = 200  # Number of steps to wait before resetting

            rollout_q_traj = []
            while rollout_step < max_rollout_length:
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
                        # Generate new random position for robot
                        new_robot_pos = np.random.rand(3) * 0.1
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
            if save_flag:
                # save
                if rollout_step < max_rollout_length:
                    success=True
                else:
                    success=False
                final_baseline_cost = 500*comp_pos_error+5*comp_quat_error
                save_matrix.append([str(trial_count), str(rollout_step), str(success), str(final_baseline_cost), str(comp_pos_error), str(comp_quat_error), str(mpc_horizon), str(pos_weight)])                
                save_row = save_row+1
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
print(f"used pos weights: {used_pos_weights}")
print(f"used mpc horizons: {used_mpc_horizons}")

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


