# impact-recovery

A machine-learning solution for righting copter drones after midair impact or loss of control. Modern autopilots are capable of hovering in slight winds and maintaining a level horizon after small disturbances, but cannot manage recovery after extreme impacts or volatile winds. As a result, this network aims to provide an autonomous method for mid-flight impact recovery.




## Maintaining neural net efficacy with different drone types
In order to train a neural net that can work on drones with a varying number of propellers in an array of orientations, the solution will be split into two networks. The first will determine the best course of action to right the drone. The second is trained to associate control inputs with drone IMU data to understand the drone's physical properties. This network will interface between the drone hardware and the first network. In this way, the software is not dependent on a specific type of drone configuration, assuming that the drone has physical control in all six degrees of freedom.




## Learning strategy
A deep reinforcement learning neural network alongside a physics simulator is ideal for this approach. The scenario exhibits a continuous observation space and a continuous action space.

### Data input/output
The training program will take in IMU data to understand the drone's orientation in 3D space.
> In non-simulated circumstances, the network will not have access to a precise, absolute location in 3D space. Therefore, the drone's absolute position will be used in the cost function to prioritize models that excel in fast recovery with minimal displacement but will not be used as a training parameter.

### Avoiding overfitting
Several steps must be taken to avoid model overfitting and optimize the network's robustness.

1. **Randomization of start parameters** - Gazebo integrates ROS topics that dictate a physical body's position and orientation. The training will randomize pitch, roll, yaw, angular velocity, translational velocity, and initial propeller speed. In this way, the final network will be capable of resisting small nudges as well as high-speed impacts.

2. **Separation of drone-specific parameters** - With a two-network approach, the software solution can adapt to any kind of copter drone, provided the network has enough on-site human-directed training. This training resembles the first few minutes of Ardupilot-controlled flight where the autopilot adjusts parameters upon observing the drone's physical properties in-flight.




## Measuring network success
The model will be evaluated on a number of factors:
1. Distance traveled from moment of impact to stable hover recovery.
2. Time elapsed from moment of impact to stable hover recovery.
3. Total angular and translational momentum in the event of ground impact.




## Testing
After the training of a neural net that maximizes its average score, the first tests will be visual assessment of the trained network's performance within Gazebo. Following several iterations of these visual assessments and provided enough time left in the semester, physical tests may be conducted using a quadcopter with an onboard companion computer running the neural network. 
