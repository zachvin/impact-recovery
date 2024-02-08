# impact-recovery

A machine-learning solution for righting copter drones after midair impact or loss of control.

## Maintaining neural net efficacy with different drone types
In order to train a neural net that can work on drones with a varying number of propellers in an array of orientations, the solution will be split into two networks. The first will determine the best course of action to right the drone. The second is trained to associate control inputs with drone IMU data to understand the drone's physical properties. This network will interface between the drone hardware and the first network. In this way, the software is not dependent on a specific type of drone configuration, assuming that the drone has physical control in all six degrees of freedom.

## Learning strategy
A deep reinforcement learning neural network alongside a physics simulator is ideal for this approach. The scenario exhibits a continuous observation space and a continuous action space.

### Data input/output
The training program will take in IMU data to understand the drone's orientation in 3D space. In non-simulated circumstances, the network will not have access to a precise, absolute location in 3D space. Therefore, the drone's absolute position will be used in the cost function to prioritize models that excel in fast recovery with minimal motion but will not be used as a training parameter.
