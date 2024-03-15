from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import BaseRLAviary

class RecoveryAviary(BaseRLAviary):
    def _computeReward(self):
        """
        Computes current reward value.
        """
        return 1

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        return False
    
    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        return False
    
    def _computeInfo(self):
        return None
    
    