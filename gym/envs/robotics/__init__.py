from gym.envs.robotics.fetch_env import FetchEnv
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.reach import FetchReachEnv

from gym.envs.robotics.hand.reach import HandReachEnv
from gym.envs.robotics.hand.manipulate import HandBlockEnv
from gym.envs.robotics.hand.manipulate import HandEggEnv
from gym.envs.robotics.hand.manipulate import HandPenEnv

from gym.envs.robotics.hand.manipulate_touch_sensors import HandBlockTouchSensorsEnv
from gym.envs.robotics.hand.manipulate_touch_sensors import HandEggTouchSensorsEnv
from gym.envs.robotics.hand.manipulate_touch_sensors import HandPenTouchSensorsEnv

from gym.envs.robotics.ur5.assemble import Ur5HegEnv
from gym.envs.robotics.ur5.assemble import Ur5HegCartEnv

from gym.envs.robotics.ur10.assemble import Ur10HegEnv
from gym.envs.robotics.ur10.assemble import Ur10HegSparseEnv
from gym.envs.robotics.ur10.assemble import Ur10HegRelEnv
from gym.envs.robotics.ur10.assemble import Ur10HegRelVaryEnv
from gym.envs.robotics.ur10.assemble import Ur10HegCorrectiveEnv
from gym.envs.robotics.ur10.assemble import Ur10HegCorrectiveVaryEnv
from gym.envs.robotics.ur10.assemble import Ur10HegRandForceEnv #UR10HEG-v001
from gym.envs.robotics.ur10.assemble import Ur10HegStaticEnv #UR10HEG-v000
from gym.envs.robotics.ur10.assemble import Ur10HegGenesisEnv #UR10HEG-v002
from gym.envs.robotics.ur10.assemble import Ur10HegNoisyPositionEnv #UR10HEG-v003
from gym.envs.robotics.ur10.assemble import Ur10HegNoisyForceEnv #UR10HEG-v004
from gym.envs.robotics.ur10.assemble import Ur10HegAutoRandForceEnv #UR10HEG-v005
