from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from gym_pybullet_drones.envs.HoverAviary import HoverAviary

import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class RecoveryAviary(HoverAviary):

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        
        self.INIT_XYZS = initial_xyzs
        self.INIT_RPYS = initial_rpys
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        
    def _computeReward(self):
        """
        Computes current reward value.
        """

        state = self._getDroneStateVector(0)

        # reward for approaching target position
        pos_reward = 2 - np.abs(np.linalg.norm(self.TARGET_POS-state[0:3]))

        # punishment for being on ground
        if state[2] <= 0.015:
            pos_reward -= 1
        
        # large reward for getting to target position
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            pos_reward += 100

        pos_reward = max(pos_reward, 0)

        # reward for minimizing roll, pitch, and yaw
        ori_reward = max(2 - np.abs(np.linalg.norm(state[7:10])), 0)

        # reward for minimizing angular velocity
        rot_reward = max(2 - np.abs(np.linalg.norm(state[13:16])), 0)

        ret = 0.33*pos_reward + 0.33*ori_reward + 0.33*rot_reward
        
        return 0.67*pos_reward + 0.33*ori_reward
    
        #pos (3), quat (4), rpy, (3), vel (3), ang_v (3), last_clipped_action (4)
        #pos     = state[0:3]
        #quat    = state[3:7]
        #rpy     = state[7:10]
        #vel     = state[10:13]
        #ang_v   = state[13:16]
        #act     = state[16:20]

    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            # changed from True to False for testing
            # keep as False, True is dumb
            return False
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        elif (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            # should return True - changed for testing
            return True
        else:
            return False
        
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            ret = np.reshape(ret, (-1, 12))[0]
            return ret
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")