import logging
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import faiss
import gym
import udo_optimization

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers


import tensorflow_probability.python.distributions as tfp
from collections import deque

class Policy(object):
    def __init__(self, obssize, actsize, optimizer):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        self.model = self.create_model(obssize, actsize)
        self.optimizer = optimizer

        # 其他变量初始化
        self.explore_alpha = tf.Variable(0.1, dtype=tf.float32)

    def create_model(self, obssize, actsize):
        inputs = tf.keras.Input(shape=(obssize,))
        h1 = layers.Dense(50, activation='sigmoid')(inputs)
        outputs = layers.Dense(actsize, activation='softmax')(h1)
        return tf.keras.Model(inputs, outputs)

    def compute_prob(self, states):
        """
        Compute prob over actions given states pi(a|s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, actsize]
        """
        probabilities = self.model(states)
        return probabilities.numpy()

    def train(self, states, actions, Qs):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        with tf.GradientTape() as tape:
            probabilities = self.compute_prob(states)
            actions_one_hot = tf.one_hot(actions, self.model.output_shape[-1])
            action_probabilities = tf.reduce_sum(probabilities * actions_one_hot, axis=1)
            surrogate_loss = -tf.reduce_mean(tf.math.log(action_probabilities) * Qs)

        grads = tape.gradient(surrogate_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_expl(self, states, actions, Qs, probs_buffer, expl_alpha, weights, distance_metric):
        """
        Train the policy with exploration loss.

        states: numpy array of states
        actions: numpy array of actions taken
        Qs: numpy array of Q values (not used in this implementation)
        probs_buffer: list of numpy arrays containing policy probabilities from the buffer
        expl_alpha: exploration weight (scaling factor for exploration loss)
        weights: numpy array for weighting the exploration loss
        distance_metric: string representing the method to use for exploration loss calculation
        """
        with tf.GradientTape() as tape:
            # Compute current policy probabilities
            probabilities = self.compute_prob(states)

            # Calculate exploration loss
            exploration_loss = self.compute_exploration_loss(probabilities, actions, Qs, probs_buffer, weights, distance_metric)

            # Calculate total loss (negative of exploration loss)
            total_loss = -expl_alpha * exploration_loss

        # Compute gradients and apply updates
        grads = tape.gradient(total_loss, self.model.trainable_variables)

        if grads is not None and any(g is not None for g in grads):
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        else:
            print("No gradients provided for any variable.")
    

    def compute_exploration_loss(self, probabilities, actions, advantages, probs_buffer, weights, distance_metric):
        """
        Compute exploration loss based on the specified distance metric.
        
        states: numpy array of size [numsamples, obssize]
        actions: numpy array of size [numsamples]
        advantages: numpy array of computed advantages
        probs_buffer: list of policy probabilities from different buffers
        weights: numpy array of weights for exploration loss
        distance_metric: string representing method to use for exploration
        """
        # Compute current policy probabilities
        X = tfp.Categorical(probs=probabilities)

        if distance_metric == "KL":
            losses = []
            for prob in probs_buffer:
                Y = tfp.Categorical(probs=prob)
                distance = tfp.kl_divergence(X, Y)
                losses.append(distance)
            exploration_loss = tf.reduce_mean(tf.reduce_sum(tf.stack(losses), axis=0) * weights)


        elif distance_metric == "JS":
            # Calculate JS divergence
            m = 0.5 * (probs_buffer + probabilities)
            M = tfp.Categorical(probs=m)
            Y = [tfp.Categorical(probs=prob) for prob in probs_buffer]
            JS_losses = [0.5 * (tfp.kl_divergence(X, M) + tfp.kl_divergence(Y_i, M)) for Y_i in Y]

            # Convert JS_losses to a tensor and ensure proper shape
            JS_losses_tensor = tf.stack(JS_losses)  # Shape [num_buffers, num_samples]

            # Ensure weights are reshaped for compatibility and converted to float
            weights_reshaped = tf.convert_to_tensor(weights, dtype=tf.float32)
            weights_reshaped = tf.reshape(weights_reshaped, [len(weights), 1])  # Shape [num_samples, 1] for broadcasting

            # Calculate exploration loss
            exploration_loss = tf.reduce_mean(tf.reduce_sum(JS_losses_tensor * weights_reshaped, axis=0))



        else:
            raise ValueError("Unknown distance metric: {}".format(distance_metric))

        return exploration_loss


# define value function as a class
class ValueFunction(object):
    def __init__(self, obssize, optimizer):
        """
        obssize: size of states
        """
        self.model = self.create_model(obssize)
        self.optimizer = optimizer

    def create_model(self, obssize):
        inputs = tf.keras.Input(shape=(obssize,))
        h1 = layers.Dense(50, activation='sigmoid')(inputs)
        outputs = layers.Dense(1)(h1)  # Output layer for value prediction
        return tf.keras.Model(inputs, outputs)

    def compute_values(self, states):
        """
        Compute value function for given states
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples]
        """
        return self.model(states).numpy()

    def train(self, states, targets):
        """
        states: numpy array
        targets: numpy array
        """
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            error = predictions - targets
            loss = tf.reduce_mean(tf.square(error))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


def discounted_rewards(r, gamma):
    """ 
    take 1D float array of rewards and compute discounted reward 
    returns a list where the first element is the complete discounted reward for the whole trajectory (already summed),
    the second element is the complete discounted reward for the trajectory starting at t=1 and so on...
    """
    
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)



# Implement replay buffer
class ReplayBuffer(object):
    
    def __init__(self, maxlength):
        """
        maxlength: max number of tuples to store in the buffer
        if there are more tuples than maxlength, pop out the oldest tuples
        """
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength
    
    def append(self, experience):
        """
        this function implements appending new experience tuple
        experience: a tuple of the form (s,a,r,s^\prime)
        """
        self.buffer.append(experience)
        self.number += 1
        
    def pop(self):
        """
        pop out the oldest tuples if self.number > self.maxlength
        """
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1
    
    def sample(self, batchsize):
        """
        this function samples 'batchsize' experience tuples
        batchsize: size of the minibatch to be sampled
        return: a list of tuples of form (s,a,r,s^\prime)
        """
        batch = random.sample(self.buffer, batchsize)
            
        return batch
    
    

def build_target_update(from_model, to_model):
    """
    from_model: source model
    to_model: target model
    """
    ops = []
    for from_var, to_var in zip(from_model.model.trainable_variables, to_model.model.trainable_variables):
        ops.append(to_var.assign(from_var))
    return ops

    
def run_a2c_exploration_agent(driver, queries, candidate_indices, tuning_config):
    """Run a2c_exploration agent for universal optimization"""
    
    episodes = tuning_config['duration']
    adaptive_scaling_strategy = 'proactive' # Options are distance, proactive, reactive. TODO proactive and reactive
    distance_metric = "JS" # Options are: Linf, L1, L2, KL (for Kullback–Leibler divergence), JS (for Jensen–Shannon divergence)
    distribution_buffer = "Unif" # Options are: Unif, exp_high_recent, exp_high_old, reward_high


    # tune/add hyper-parameters numtrajs too high ? code is very slow
    # parameter initializations
    
    alpha = 1e-3  # learning rate for PG
    beta = 1e-3  # learning rate for baseline
    delta = 0.05 # exploration weight in loss function. 0.05 is already quite optimized
    numtrajs = 1  # num of trajecories to collect at each iteration 
    
    # envname = "CartPole-v1"  # environment name
    
    gamma = .99  # discount
    # episodes = 400 # total num of iterations
    maxlength = 5
    start = False
    exploration_alpha = 0.1
    
    # adaptive_scaling_strategy = 'proactive' # Options are distance, proactive, reactive. TODO proactive and reactive
    # distance_metric = "JS" # Options are: Linf, L1, L2, KL (for Kullback–Leibler divergence), JS (for Jensen–Shannon divergence)
    # distribution_buffer = "Unif" # Options are: Unif, exp_high_recent, exp_high_old, reward_high
    
    # initialize environment
    env = gym.make("udo_optimization-v0", driver=driver, queries=queries, candidate_indices=candidate_indices,
                   config=tuning_config)
    obssize = env.observation_space.shape[0]
    actsize = env.nA_index


    # Optimizers
    optimizer_p = tf.keras.optimizers.Adam(alpha)
    optimizer_v = tf.keras.optimizers.Adam(beta)

    # Initialize networks
    actor = Policy(obssize, actsize, optimizer_p)  # Policy initialization
    baseline = ValueFunction(obssize, optimizer_v)  # Baseline initialization
    
    buffers = [Policy(obssize, actsize, optimizer_p) for _ in range(5)]  # Create multiple buffers


    buffer_count = 1

    #logging info
    exploration_loss_log = np.array([])
    exploration_alpha_log = np.array([])
    avg_reward_per_policy = []
    reward_buffer_1, reward_buffer_2, reward_buffer_3, reward_buffer_4, reward_buffer_5 = 0,0,0,0,0

    
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    episodes_list = []
    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(log_dir='results/'+(datetime.now()+ timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S_")+"a2c_exploration")


    # main iteration
    ite=0
    while ite <episodes:    

        # trajs records for batch update
        OBS = []  # observations
        ACTS = []  # actions
        ADS = []  # advantages (to update policy)
        VAL = []  # value functions (to update baseline)
        episodic_reward = 0

        for num in range(numtrajs):
            # record for each episode
            obss = []  # observations
            acts = []   # actions
            rews = []  # instant rewards

            obs = env.reset()
            done = False

            while not done:

                prob = actor.compute_prob(np.expand_dims(obs,0))
                action = np.random.choice(actsize, p=prob.flatten(), size=1)
                newobs, reward, done, _ = env.step(action[0])
                episodic_reward += reward
                logging.info(f"Episode * {ite} * Reward is ==> {reward}")

                ite+=1

                # record
                obss.append(obs)
                acts.append(action[0])
                rews.append(reward)
                #print(reward)

                # update
                obs = newobs
            
            ep_reward_list.append(episodic_reward)
            writer.add_scalar('Reward', episodic_reward, ite)
            episodes_list.append(ite)
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            logging.info(f"Episode * {ite} * Avg Reward is ==> {avg_reward}")
            avg_reward_list.append(avg_reward)
            writer.add_scalar('Avg_Reward', avg_reward, ite)
            logging.info(f"episode: {ite}")

            # compute returns from instant rewards for one whole trajectory
            returns = discounted_rewards(rews, gamma)
            avg_reward_per_policy += [sum(returns)]

            # record for batch update
            VAL += returns # NOTE that the list of returns just gets extended. 
                           # There is no separate entry created for each trajectory
            OBS += obss
            ACTS += acts         

        # update baseline
        VAL = np.array(VAL)# represents an array where the discounted reward lists are concatenated to each other.
        # the size of VAL should be [numtrajs * len_of_traj_i, 1] where len_of_traj_i is variable depending on the length 
        # of each trajectory.
        OBS = np.array(OBS)# represents an array where the list of states for all trajectories are concatenated.
        # the size of OBS should be [numtrajs * len_of_traj_i, obssize]
        ACTS = np.array(ACTS)

        baseline.train(OBS, VAL)  # update only one step

        # update policy
        BAS = baseline.compute_values(OBS)  # compute baseline for variance reduction
        ADS = VAL - np.squeeze(BAS,1) # computes advantages. An array of (targets)-(estimated from our network) for each
                                      # state

        # Update buffers
        for i, buffer in enumerate(buffers):
            update_op = build_target_update(actor, buffer)
            # 执行变量更新
            for from_var, to_var in zip(actor.model.trainable_variables, buffer.model.trainable_variables):
                to_var.assign(from_var)


        if distribution_buffer == "Unif":
            weights = np.array([1,1,1,1,1])
        elif distribution_buffer == "exp_high_recent": # recent experience has a bigger historic order value
            weights = np.roll(np.array([1,2,3,4,5]), buffer_count)
        elif distribution_buffer == "exp_high_older": # older experience has a bigger historic order value                      
            weights = np.roll(np.array([5,4,3,2,1]), buffer_count)
        elif distribution_buffer == "reward_high": # experience with high reward has a bigger historic order value                      
            weights = np.array([reward_buffer_1, reward_buffer_2, reward_buffer_3, reward_buffer_4, reward_buffer_5])
        avg_reward_per_policy = []

        # Compute exploration loss
        probs_buffer = [buffer.compute_prob(OBS) for buffer in buffers]


        # Compute current probabilities for use in exploration loss
        probabilities = actor.compute_prob(OBS)

        # Compute exploration loss
        exploration_loss = actor.compute_exploration_loss(probabilities, ACTS, ADS, probs_buffer, weights, distance_metric)

        # Adaptive scaling strategy
        if adaptive_scaling_strategy == 'distance':
            if np.mean(exploration_loss) < delta:
                exploration_alpha *= 1.01
            else:
                exploration_alpha *= 0.99
        elif adaptive_scaling_strategy == 'proactive':
            if len(ep_reward_list) >= 5:
                exploration_alpha = -(2 * (np.min(ep_reward_list[-5:]) - np.min(ep_reward_list)) / 
                                      (np.max(ep_reward_list) - np.min(ep_reward_list)) - 1)
        elif adaptive_scaling_strategy == 'reactive':
            exploration_alpha = 1 - (np.min(ep_reward_list[-5:]) - np.min(ep_reward_list)) / (
                               np.max(ep_reward_list) - np.min(ep_reward_list))

        # Train actor with exploration loss
        actor.train_expl(OBS, ACTS, ADS, probs_buffer, exploration_alpha, weights, distance_metric) # update only one step

    # 关闭SummaryWriter
    writer.close()