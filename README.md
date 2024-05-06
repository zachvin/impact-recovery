# impact-recovery

Zach Vincent - Neural Networks Semester Project

A machine-learning solution for righting copter drones after midair impact or loss of control. Modern autopilots are capable of hovering in slight winds and maintaining a level horizon after small disturbances, but cannot manage recovery after extreme impacts or volatile winds. As a result, this network aims to provide an autonomous method for mid-flight impact recovery. The solution will be trained using the Gazebo physics simulator, which can be automated and manipulated using ROS and Python.

[Setup](#setup)

[Part 1](#part-one)

[Part 2](#part-two)

[Part 3](#part-three)

[Part 4](#part-four)

[Acknowledgements](#acknowledgements)

# Setup

  > Although parts 1 and 2 reference the Gazebo simulator with Ardupilot and ROS, a more lightweight environment is used for parts 3 and 4. This environment is vastly faster than Gazebo and does not require nearly as much code overhead for programmatic interaction and observation. It is based on Pybullet and OpenAI's Gym.

Setup takes about 3 minutes.

      git clone git@github.com:zachvin/impact-recovery.git
      cd impact-recovery
      pip install -r requirements.txt
      python3 main.py -n 100 -e -c -p

The simulator will run for 100 episodes or until it receives an interrupt. Upon interrupt, the user will be prompted whether to save (in `data/`) and plot (in `plots/`) training data and whether to save the trained network. The `-e` and `-c` arguments control whether to show the GUI (Eval mode) and use the pre-trained networks (use Checkpoints), respectively. They will both be set to `False` by default. The program saves the networks repeatedly throughout training as `state_dict_[actor/critic]_quicksave` and at the end as `state_dict_[actor/critic]`, but to save the networks manually without stopping the training, send a `SIGUSR1` signal to the Python process via `kill -10 [PID]`. The `-p` flag will automatically plot and save the data, but to plot the data after training, use `util.py` in the following manner:

      python3 util.py -s [score]

Where \[score] is the integer score printed to the screen when the training data was saved.

To see the PPO-trained network run in the pendulum environment, run

      python3 ppo.py -e -c
      
Plotting isn't implemented for the pendulum environment.


# Part One

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


# Part Three

## Justification of architecture

### Layers

The network architecture was selected based on [this video](https://www.youtube.com/watch?v=T0A9voXzhng), which performed well in solving the same problem. However, the associated paper ([arXiv:1707.05110](https://arxiv.org/abs/1707.05110) Hwangbo, et al. 2017) writes that no optimization of the network architecture was done.

  - Actor network: 2 hidden layers with 64 nodes and Tanh activation function
  - Critic network: 2 hidden layers with 64 nodes and Tanh activation function
  
### Loss function

The Actor loss has two components: surrogate and entropy losses.
  
1. The surrogate loss `surr1` calculates the difference between the parameters in the network at timestep `k` and timestep `k+1`. It is multiplied by the advantage function so actions that provide a higher-than-expected score are rewarded. `surr2` is the clipped version of `surr1`. Since the advantage should be maximized, the loss is multiplied by -1 so that the optimizer produces policy gradients in the correct direction.

2. Entropy loss is noise from the probability distribution that the actor network creates. It is added to the surrogate loss in order to increase exploration given the very large observation space.

The Critic uses MSE loss between the critic's output `V` and a batch of the sum of future expected rewards (rewards-to-go). The Critic is optimized to best predict the value of a given state.

## Obstacles and performance

![model](https://github.com/zachvin/impact-recovery/assets/43306216/c05d15cb-1391-489a-931e-4a83fff807c4)

Above is an example of the model training for over 1,000 games and improving slightly before catastrophic forgetting.

The network does not perform well for this problem. Despite solving test environments such as OpenAI's swinging pendulum, the PPO algorithm provides noisy average episode reward and does not improve consistently. As a result, I took the following steps in order to diagnose the problem, but have ultimately been unsuccessful:

1. **Environment sanity check**: The environment used for this project contained some oddities (i.e. observation space changes based on action type, simulator output is returned in a batch of observations instead of a single observation, poor documentation), so it is possible that the issue was with the environment. However, manual testing shows that the simulation IO appears consistent between timesteps and between epochs. As a result, if there is an issue with the environment, I would assume it is due to an inefficient reward function.
2. **Episode truncation**: In the environment, truncation is not punished by default. As a result, the network appeared to cause episode truncation immediately to avoid negative rewards later in the simulation. An episode is truncated when the drone tilts too far or moves too far from the target location. After removing truncation (and running each simulation for a finite number of timesteps), the agent no longer converged on extremely low rewards after a few frames, but still did not produce adequate performance. However, the agent still started each episode with a sharp tilt of the quadcopter.
3. **Exploration failure**: To try to combat local minima, entropy loss was introduced to enforce greater exploration. It is difficult to assess its impact on the performance because the PPO agent still appears to converge with low-scoring policies. Entropy coefficients such as `0.001`, `0.05`, and `0.1` were tested.
4. **Start parameter randomization**: The drone's starting location was randomized to increase network generalization in the event that the poor performance was due to memorization. However, the network still doesn't improve its average reward. In many of the tests, the drone would fall from its starting location and shoot upward in an arbitrary direction upon hitting the ground. To reduce training time for debugging purposes, the start location was made static again.
5. **Reward shaping and standardization**: PPO agents require dense rewards. The original reward function provided by the environment calculated the reward as the linear distance from the target location of `(0,0,1)`. Though this is not a sparse reward function, different functions were also tested that rewarded minimizing tilt and angular velocity. Additionally, if the drone touched the ground (when `z < 0.015`), an additional point punishment was given. Prior to backpropagation, the reward is normalized according to the batch average and standard deviation. Despite several iterations and combinations of these rewards, the network still did not make significant improvements.
6. **Epsilon-greedy**: In a further attempt to encourage exploration, I implemented an epsilon-greedy policy for a short time. It did not show improvements and research suggests that it is not the optimal way to promote exploration so it was removed.

## Improvements

1. **Structure review**: The first step to improving the network's performance is a review of its structure with Prof. Czajka. Given my inexperience with neural networks and the wide array of tunable hyperparameters, it is difficult to debug.
2. **Minibatches**: Due to the large amount of training data produced by a single iteration, it is only possible to learn from a total of four epochs in a single batch with my current hardware. With minibatches, it will be possible to assemble a greater amount of data before backpropagation, and may also allow for the reduction of memorization through memory randomization.
3. **Learning rate annealing**: It is possible that the poor performance is the result of a static learning rate. My assumption is that, if a static learning rate is the issue, that would be made apparent later in training when the network is making small optimizations to its performance, whereas it currently shows no success at all. As a result, I'm skeptical of that the learning rate is the root problem.
4. **Longer testing**: It is also possible that the network is improving, but only slowly, which would in fact point to a learning rate issue as mentioned previously. With a much longer training time (several hours, as opposed to ~30mins), the graphs produced from that training may help point to a specific issue.

# Part Four

## Data

The data used in this project comes entirely from a [PyBullet environment](https://github.com/utiasDSL/gym-pybullet-drones/tree/main) built for use with quadcopters. The environment outputs 12 fields for each timestep: XYZ location, RPY orientation, XYZ linear velocity, and angular velocity. The number of samples per epoch is a controllable hyperparameter which I have decreased to 30 samples/second for training efficiency. With a high sampling rate, there was too much data to run backpropagation on several episodes at once, which led to a high degree of training instability. At each timestep, the simulator takes four continuous variables in the range \[-1, 1] that represent a proportion of the hover RPMs for each propellor. When all inputs are 0, the drone hovers; when all are -1, the propellors do not spin, and at 1 they spin at max RPM. 

## Performance

![image](https://github.com/zachvin/impact-recovery/assets/43306216/2354bfd3-1446-4d51-a0f8-5b40cc7bf226)

Unfortunately, I have not managed to achieve the desired behavior. In some cases, the drone did learn to hold its initial Z position constant, but flew erratically around the XY plane. Note that the initial Z position is about 0.1m, but the target Z position is 1m. In most cases, the network converged on an extremely poor solution; it would immediately tip itself over, causing the epoch to truncate. I am not sure why this occurs, because the initial (random) weights of the networks gave significantly higher scores, but immediately after entering the backpropagation loop, the score often dropped and ended with the tipping behavior. At this point, my best guess is that I need to optimize the size and number of minibatches in the learning loop, which is discussed in more detail below. I implemented each of the three main improvements from Part 3 and did not see improvements from them, either.

## Training v. Validation and Improvements

Since there is no explicit training/validation for reinforcement learning, I will consider simpler environments to be the "training" set and the complex quadcopter environment to be the "validation" set. Although these are not technically datasets, their relative difficulty levels are good enough analogs to training/validation. To build up to the main problem of impact recovery, I first started trying to use the PPO algorithm to solve Gym's inverted pendulum problem. Not only did the network succeed at solving the environment, but it was also very stable and converged to a near-perfect average score. Using the network and hyperparameters for the pendulum as a starting point, I moved to the first drone problem: hovering at a specific location. With confidence in the algorithm, the quadcopter environment comes down mostly to finding the proper hyperparameters.

The pendulum environment has only two continuous observation variables with one continuous input, whereas the quadcopter environment has 12 continuous observation variables with four continuous inputs. Since the observation space is extremely complex, I added an extra layer to both actor and critic networks and decreased the learning rate. In addition, I ran the training for about twice as long per test this time (~50mins) to compensate for the increased need for exploration (as planned in Part 3).

After not seeing improved performance, I then implemented learning rate annealing (as planned in Part 3). The actor network's loss almost always converged to a very small amount (<0.01) the critic network's loss would usually remain relatively high (in the thousands). The annealing may have helped, but it is difficult to tell because the stochasticity of PPO and the environment led to unstable losses. As a result, this had no tangible impact on the final scores of the agent in the simulator--it is possible that it would help later in training, assuming the agent is actually increasing its scores.

Finally, I implemented minibatch learning so that I could increase the number of episodes run between learning loops. Ideally, the randomized memory selection would introduce noise that forces the agent to generalize more effectively, and the larger number of episodes run would produce more accurate gradients. The number of epochs run between learning loops made the difference between zero improvement and convergence on an effective strategy for the pendulum environment, so I assume the issue is the same in this scenario. Despite trying several different combinations of epochs per batch, number of minibatches, and number of learning loops per batch, I could not get the network to converge on anything other than the tilting behavior.

For the future, my only remaining ideas are to continue modifying the number of minibatches. There is a lot of work to be done to parallelize the simulations and optimize for GPUs, which would help tremendously for getting larger batches and testing faster. I would also like to examine the feasibility of "seeding" the networks with example memories. If I can manually control the drone for a couple epochs to show ideal behavior, then I wonder if PPO would then bias those actions and provide a better starting point from which noise can be introduced and the state space can be incrementally explored. Overall, despite the poor performance, I think the code in this repository is a very strong start to solving this environment and has the potential to produce an effective agent. Once the drone learns to hover, learning impact recovery is only a matter of starting epochs with upside-down initial drone positions with relatively large linear and angular velocities.

# Acknowledgements
The code in `ppo.py` is adapted from [this implementation](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) by Eric Yang Yu.

The code is an implementation of [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) by Schulman, et al. 2017.

The simulator used is based on PyBullet and is found [here](https://github.com/utiasDSL/gym-pybullet-drones).
