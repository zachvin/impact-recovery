from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np

class RecoveryAviary(HoverAviary):
    def _computeReward(self):
        """
        Computes current reward value.
        """
        state = self._getDroneStateVector(0)
        # pos (3), quat (4), rpy, (3), vel (3), ang_v (3), last_clipped_action (4)

        pos     = state[0:3]
        quat    = state[3:7]
        rpy     = state[7:10]
        vel     = state[10:13]
        ang_v   = state[13:16]
        act     = state[16:20]

        TARGET_ROT = np.array([0, 0, 0])
        TARGET_POS = np.array([0, 0, 1])

        pos_reward = 2 - np.linalg.norm(TARGET_POS-pos)**4
        rot_reward = 2 - np.linalg.norm(TARGET_ROT-rpy)**4

        reward = pos_reward + rot_reward

        ret = max(0, pos_reward)
        return ret
    