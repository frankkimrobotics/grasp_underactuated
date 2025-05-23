import casadi as cs
import numpy as np

import utils.rotations as rot

t_base = rot.ttmat_fn([0, 0, 0])

#  finger0
f0_qpos = cs.SX.sym('f0_qpos', 3)
f0_t_upper = t_base @ rot.ttmat_fn([0, 0.04, 0.29])
f0_t_middle = f0_t_upper @ rot.rytmat_fn(f0_qpos[0]) @ rot.ttmat_fn([0, 0, 0.0])
f0_t_lower = f0_t_middle @ rot.rxtmat_fn(-f0_qpos[1]) @ rot.ttmat_fn([0, 0, -0.16])
f0_t_tip = (f0_t_lower @ rot.rxtmat_fn(f0_qpos[2]) @ rot.ttmat_fn([0.0, 0, -0.16]))
f0tip_pos_fd_fn = cs.Function('f0tp_pos_fd_fn', [f0_qpos], [f0_t_tip[0:3, -1]])
