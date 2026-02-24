# Reinforcement Learning for the Frozen Lake Environment

This project implements reinforcement learning algorithms to solve the Frozen Lake navigation problem from OpenAI Gym, a stochastic grid-world environment used to study decision-making under uncertainty.

The goal is for an agent to learn how to reach a target state while avoiding holes in a slippery environment with probabilistic transitions.

---

## Methods

Two algorithms were implemented **from scratch using NumPy**:

### Value Iteration (Model-Based)
- Computes the optimal policy using Bellman updates
- Requires knowledge of transition probabilities
- Used as a theoretical benchmark

### Q-Learning (Model-Free)
- Learns action values through interaction with the environment
- Does not require prior knowledge of dynamics
- Demonstrates learning under uncertainty

---

## Results

- Value Iteration produced optimal policies for deterministic and stochastic environments.
- Q-Learning successfully learned navigation strategies without prior knowledge.
- Learned policies closely matched theoretical optimal solutions.

This demonstrates the effectiveness of reinforcement learning in uncertain decision environments.

---

## Skills Demonstrated

- Reinforcement Learning
- Markov Decision Processes (MDP)
- Q-Learning
- Dynamic Programming
- NumPy implementation from scratch
- Stochastic environment modeling

---

## Authors

Weixun Xie et al.  
Kean University
