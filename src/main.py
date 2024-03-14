from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import RecoveryAviary

RPM = ActionType('rpm')
KIN = ObservationType('kin')

def run():
    env = RecoveryAviary(act=RPM, obs=KIN)

    num_games = 10

    for i in range(num_games):
        score = 0

        # obs: Box() of shape (NUM_DRONES,12)
        obs, info = env.reset()
        term, trunc = False, False

        while not term and not trunc:
            # choose action
            # get reward, new observation
            # store memory and learn
            # update observation