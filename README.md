# CUDA
UCSD ECE 285 GPU Programming on CUDA

LAB 1: Reinforcement learning: Q-learning (Single Agent)
This lab requires to design an agent to interact with the environment using the reinforcement
learning algorithm in Figure 1. Specically, you need to design a single thread agent to
maximize rewards from the mine game environment using Q-learning. The agent should
interact with the given mine game environment in Figure 2.
Figure 1: Reinforcement learning.
Figure 2: 4x4 mine game environment.
 Action: (right:0, bottom:1 , left:2, top:3)
 Reward: 
ag: +1, mine: -1, otherwise: 0
 State: (x, y) position in the coordinator of (0,0) at the top-left corner
 Every episode restarts from (0,0) after the agent reaches one of mines or a 

ag.
You should not modify any given codes except CMakeLists to add your codes.
You only need to add your agent code to the lab project.
You have to use CUDA to program a single agent with

