# impact-recovery

Zach Vincent - Neural Networks Semester Project

A machine-learning solution for righting copter drones after midair impact or loss of control. Modern autopilots are capable of hovering in slight winds and maintaining a level horizon after small disturbances, but cannot manage recovery after extreme impacts or volatile winds. As a result, this network aims to provide an autonomous method for mid-flight impact recovery. The solution will be trained using the Gazebo physics simulator, which can be automated and manipulated using ROS and Python.




## Maintaining neural net efficacy with different drone types
In order to train a neural net that can work on drones with a varying number of propellers in an array of orientations, the solution will be split into two networks. The first will determine the best course of action to right the drone. The second is trained to associate control inputs with drone IMU data to understand the drone's physical properties. This network will interface between the drone hardware and the first network. In this way, the software is not dependent on a specific type of drone configuration, assuming that the drone has physical control in all six degrees of freedom. However, I need to do more research to better understand if this dual-network design is feasible and, if so, how to go about it.




## Learning strategy
A deep reinforcement learning neural network alongside a physics simulator is ideal for this approach. The scenario exhibits a continuous observation space and a continuous action space, so I will need to learn which kind of reinforcement learning architecture is best suited for the situation. The neural network featured in this project will model the agent's Q-function and predict the best next action given a current state. Several different network architectures can be utilized and compared to determine the optimal structure.

### Data input/output
The training program will take in IMU data to understand the drone's orientation in 3D space.

> In non-simulated circumstances, the network will not have access to a precise, absolute location in 3D space. Therefore, the drone's absolute position will be used in the cost function to prioritize models that excel in fast recovery with minimal displacement but will not be used as a training parameter.

The network will output N discrete values, where N is the number of propellers on the drone. Depending on the feasibility of the dual-network design, the training may be limited to a quadcopter with a specific layout to decrease complexity. Each output will correspond to the power output of one motor.

### Avoiding overfitting
Several steps must be taken to avoid model overfitting and optimize the network's robustness.

1. **Randomization of start parameters** - Gazebo integrates ROS topics that dictate a physical body's position and orientation. The training will randomize pitch, roll, yaw, angular velocity, translational velocity, and initial propeller speed. In this way, the final network will be capable of resisting small nudges as well as high-speed impacts.

2. **Separation of drone-specific parameters** - With a two-network approach, the software solution can adapt to any kind of copter drone, provided the network has enough on-site human-directed training. This training resembles the first few minutes of Ardupilot-controlled flight where the autopilot adjusts parameters upon observing the drone's physical properties.




## Measuring network success
The model will be evaluated on a number of factors, each of which must be included in the Q-function:
1. Distance traveled from moment of impact to stable hover recovery.
2. Time elapsed from moment of impact to stable hover recovery.
3. Total angular and translational momentum in the event of ground impact.




## Testing
After the training of a neural net that maximizes its average score, the first tests will be visual assessment of the trained network's performance within Gazebo. Following several iterations of these visual assessments and provided enough time left in the semester, physical tests may be conducted using a quadcopter with an onboard companion computer running the neural network.


# Part Two
## Source
All of the data in this project will be generated during training time using the Gazebo physics simulator. Because there is no data to download, this part 2 submission will show a proof-of-concept for reading data from the simulation programmatically. Gazebo outputs a set of information topics that can be bridged to ROS topics using a separate ROS package. My python software then subscribes to those topics and will use the data it receives for the neural network input.


## Distinct subjects represented in data
The data included in the proof-of-concept only shows the linear acceleration along the Z axis for simplicity. However, the full dataset used for training includes:
1. X, Y, Z linear acceleration
2. Roll, pitch, yaw angular acceleration
3. Time
4. Absolute position in 3D space


## Training vs. testing
The data used for training will include all of the above topics. The only data point not used in testing is the absolute position of the drone in 3D space, since this is impractical to achieve on a real drone. Therefore, the absolute position data will only be used for the cost function to minimize overall displacement during impact recovery, prioritizing models that achieve quick stability, but the network itself will not utilize this information.

## Characterization of samples
All of the data are represented as float values. The samples come from a virtualized IMU. Sampling frequency can be controlled easily within the simulator, so this will require fine-tuning during training to ensure that training is not too resource-intensive.

## Data samples
> Because the data is scraped from the simulator, it is not feasible to reproduce the results until a Docker image is produced (this will also allow for easy development on a laptop while outsourcing the training to more powerful hardware). As a result, the files used to produce the data samples are included in this repository for proof of completion, but do not include their dependencies (i.e. ROS packages, setup files, etc.)

`ionode.py` - Python script that exists in a ROS package; subscribes to `copter_imu` ROS topic and writes data to `testdata.txt`.

`testdata.txt` - Text values of data visualized below.

`gazebo-startup` - Shell script that automates starting Gazebo, Ardupilot SITL, and the ROS topic bridge.

`topics.yaml` - Config file that contains all ROS/Gazebo topic bridging information.


![image](https://github.com/zachvin/impact-recovery/assets/43306216/9a8e792e-3dfa-4dfa-b5fb-ff06067f4bd1)

At ~4000, the drone takes off vertically 10 meters. The data becomes significantly noisier, and at ~6500, the drone rotates about the Z axis, causing a shift in the acceleration values.
