import logging
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import faiss
import gym
import udo_optimization
from udo.agent.EDER import EDER
from udo.agent.PrioritizedReplayBuffer import PrioritizedReplayBuffer

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def run_ddpg_agent(driver, queries, candidate_indices, tuning_config):
    """Run DDPG agent for universal optimization"""
    env = gym.make("udo_optimization-v0", driver=driver, queries=queries, candidate_indices=candidate_indices,
                   config=tuning_config)

    num_states = env.observation_space.shape[0]
    num_actions = env.nA_index
    logging.debug(f"Size of State Space ->  {num_states}")
    logging.debug(f"Size of Action Space ->  {num_actions}")

    upper_bound = 1
    lower_bound = -1

    logging.debug(f"Max Value of Action ->  {upper_bound}")
    logging.debug(f"Min Value of Action ->  {lower_bound}")

    # the following code is from Keras RL example, https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py
    class OUActionNoise:
        def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
            self.theta = theta
            self.mean = mean
            self.std_dev = std_deviation
            self.dt = dt
            self.x_initial = x_initial
            self.reset()

        def __call__(self):
            # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
            x = (
                    self.x_prev
                    + self.theta * (self.mean - self.x_prev) * self.dt
                    + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
            )
            # Store x into x_prev
            # Makes next noise dependent on current one
            self.x_prev = x
            return x

        def reset(self):
            if self.x_initial is not None:
                self.x_prev = self.x_initial
            else:
                self.x_prev = np.zeros_like(self.mean)

    class Buffer:
        def __init__(self, buffer_capacity=100000, batch_size=64,MaximumDiversity="-1"):
            # Number of "experiences" to store at max
            self.buffer_capacity = buffer_capacity
            # Num of tuples to train on.
            self.batch_size = batch_size
            self.MaximumDiversity=MaximumDiversity

            self.EDERReplayBuffer = EDER(obs_dim=num_states, act_dim=num_actions, size=buffer_capacity)
            self.PrioritizedReplayBuffer = PrioritizedReplayBuffer(state_size=num_states, action_size=num_actions, buffer_size=buffer_capacity)

            # Its tells us num of times record() was called.
            self.buffer_counter = 0

            # Instead of list of tuples as the exp.replay concept go
            # We use different np.arrays for each tuple element
            self.state_buffer = np.zeros((self.buffer_capacity, num_states))
            self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

            
            self.hnsw_index7 = faiss.IndexHNSWFlat(num_states + num_states+num_actions + 1 +1, 32, faiss.METRIC_INNER_PRODUCT)
            self.hnsw_index8 = faiss.IndexHNSWFlat(num_states +num_actions + 1 +1, 32, faiss.METRIC_INNER_PRODUCT)

        # Takes (s,a,r,s') obervation tuple as input
        def record(self, obs_tuple):
            # Set index to zero if buffer_capacity is exceeded,
            # replacing old records
            index = self.buffer_counter % self.buffer_capacity

            self.state_buffer[index] = obs_tuple[0]
            self.action_buffer[index] = obs_tuple[1]
            self.reward_buffer[index] = obs_tuple[2]
            self.next_state_buffer[index] = obs_tuple[3]

            self.buffer_counter += 1

            if self.MaximumDiversity =='eder':
                self.EDERReplayBuffer.store( obs_tuple[0], obs_tuple[1], obs_tuple[2], obs_tuple[3],obs_tuple[4])
            elif self.MaximumDiversity=='per':
                self.PrioritizedReplayBuffer.add((obs_tuple[0], obs_tuple[1], obs_tuple[2], obs_tuple[3], int(obs_tuple[4]))) 

            elif self.MaximumDiversity=="7":
                newBuffer = np.concatenate((self.state_buffer[index],self.next_state_buffer[index],self.action_buffer[index],self.reward_buffer[index]))
                #print('new buffer:',newBuffer.shape,newBuffer)
                newBuffer=newBuffer.reshape(1,-1)
                #print('new buffer:',newBuffer.shape,newBuffer)
                newBuffer /= np.linalg.norm(newBuffer, axis=1, keepdims=True)
                #print('new buffer:',newBuffer.shape,newBuffer)
                t_prime = np.concatenate((-2 * newBuffer[0], [np.linalg.norm(newBuffer)**2]))
                t_prime = t_prime.reshape(1, -1)
                #print('t_prime:',t_prime.shape,t_prime)
                #exit(1)
                self.hnsw_index7.add(t_prime)
                #print('hnsw index points number:',self.hnsw_index7.ntotal)

                #if self.hnsw_index7.ntotal %1000==0:
                print('hnsw index points number:',self.hnsw_index7.ntotal)
            elif self.MaximumDiversity=="8":
                newBuffer = np.concatenate((self.next_state_buffer[index]-self.state_buffer[index],self.action_buffer[index],self.reward_buffer[index]))
                newBuffer=newBuffer.reshape(1,-1)
                newBuffer /= np.linalg.norm(newBuffer, axis=1, keepdims=True)
                t_prime = np.concatenate((-2 * newBuffer[0], [np.linalg.norm(newBuffer)**2]))
                t_prime = t_prime.reshape(1, -1)
                self.hnsw_index8.add(t_prime)
                print('hnsw index points number:',self.hnsw_index8.ntotal)



        # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
        # TensorFlow to build a static graph out of the logic and computations in our function.
        # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
        @tf.function
        def update(
                self, state_batch, action_batch, reward_batch, next_state_batch,
        ):
            # Training and updating Actor & Critic networks.
            # See Pseudo Code.
            with tf.GradientTape() as tape:
                target_actions = target_actor(next_state_batch, training=True)
                y = reward_batch + gamma * target_critic(
                    [next_state_batch, target_actions], training=True
                )
                critic_value = critic_model([state_batch, action_batch], training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
            critic_optimizer.apply_gradients(
                zip(critic_grad, critic_model.trainable_variables)
            )

            with tf.GradientTape() as tape:
                actions = actor_model(state_batch, training=True)
                critic_value = critic_model([state_batch, actions], training=True)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(
                zip(actor_grad, actor_model.trainable_variables)
            )
        def sample(self):
            #batch_size=min(self.buffer_counter, self.buffer_capacity)

            #print('sample function: self.MaximumDiversity ',self.MaximumDiversity)


            if self.MaximumDiversity =='eder':
                batch = self.EDERReplayBuffer.sample(self.batch_size)
                return batch
            
            elif self.MaximumDiversity=='per':
                batch, weights, tree_idxs = self.PrioritizedReplayBuffer.sample(self.batch_size)
                return batch

            elif self.MaximumDiversity=="-1":
                # Get sampling range
                record_range = min(self.buffer_counter, self.buffer_capacity)
                # Randomly sample indices
                batch_indices = np.random.choice(record_range, self.batch_size)
                return batch_indices
            elif self.MaximumDiversity == "7":
                
                randome_number=0
                ID_indx=[]

                buffer_size=self.hnsw_index7.ntotal
                if self.batch_size>=buffer_size:
                    return np.random.choice(buffer_size, self.batch_size)
                idxs = np.random.randint(0, buffer_size, size=1)
                randome_number=randome_number+1
                #print(idxs)
                ID_indx.append(idxs[0])

                o, a, r, o2= self.state_buffer[idxs[0]], self.action_buffer[idxs[0]], self.reward_buffer[idxs[0]], self.next_state_buffer[idxs[0]]
                v_0=np.concatenate((o,o2, a,r))
                q = np.concatenate((v_0, [1]))
                q = q.reshape(1, -1)
                D, I = self.hnsw_index7.search(q, int(self.batch_size))
                
                flag_append=0
                for item in I[0]:
                    if item not in ID_indx :
                        ID_indx.append(item)
                        flag_append=1
                        break  # 找到第一个未出现的元素后退出循环
                if flag_append==0:
                    randome_number=randome_number+1
                    
                    # 生成所有可能的索引
                    all_indices = np.arange(buffer_size)
                    
                    # 排除 ID_indx 中的索引
                    available_indices = np.setdiff1d(all_indices, ID_indx)
                    #available_indices = np.setdiff1d(all_indices, self.used_queue)
                    
                    # 随机选择一个索引
                    random_index = np.random.choice(available_indices)
                    ID_indx.append(random_index)
                    

                for i in range( self.batch_size-2):
                    o, a, r, o2= self.state_buffer[idxs[0]], self.action_buffer[idxs[0]], self.reward_buffer[idxs[0]], self.next_state_buffer[idxs[0]]
                    v_temp=np.concatenate((o,o2, a, r))
                    # 计算对应元素相加
                    v_0 = v_0 + v_temp
                    # 构造新向量 q'
                    q_prime = np.concatenate((v_0, [i+2]))  # 将总和和常数2组合
                    q_prime = q_prime.reshape(1, -1)
                
                    D, I = self.hnsw_index7.search(q_prime, int(self.batch_size))
                    
                    # 遍历列表 B，添加第一次未在 A 中出现的元素
                    flag_append=0
                    for item in I[0]:
                        if item not in ID_indx :
                            ID_indx.append(item)
                            flag_append=1
                            break  # 找到第一个未出现的元素后退出循环
                    if flag_append==0:
                        randome_number=randome_number+1

                        # 生成所有可能的索引
                        all_indices = np.arange(buffer_size)
                        
                        # 排除 ID_indx 中的索引
                        available_indices = np.setdiff1d(all_indices, ID_indx)
                        #available_indices = np.setdiff1d(all_indices, self.used_queue)
                        
                        # 随机选择一个索引
                        random_index = np.random.choice(available_indices)
                        ID_indx.append(random_index)

                print('randome_number',randome_number)
                if len(ID_indx) != self.batch_size:
                    print('!!!!! len(ID_indx) != self.batch_size')
                print(ID_indx)
                return ID_indx

            elif self.MaximumDiversity == "8":
                randome_number=0
                ID_indx=[]

                buffer_size=self.hnsw_index8.ntotal
                if self.batch_size>=buffer_size:
                    return np.random.choice(buffer_size, self.batch_size)
            
                idxs = np.random.randint(0, buffer_size, size=1)
                randome_number=randome_number+1
                #print(idxs)
                ID_indx.append(idxs[0])

                o, a, r, o2= self.state_buffer[idxs[0]], self.action_buffer[idxs[0]], self.reward_buffer[idxs[0]], self.next_state_buffer[idxs[0]]
                v_0=np.concatenate((o2-o, a,r))
                q = np.concatenate((v_0, [1]))
                q = q.reshape(1, -1)
                D, I = self.hnsw_index8.search(q, int(self.batch_size))
                
                flag_append=0
                for item in I[0]:
                    if item not in ID_indx :
                        ID_indx.append(item)
                        flag_append=1
                        break  # 找到第一个未出现的元素后退出循环
                if flag_append==0:
                    randome_number=randome_number+1
                    while True:
                        idxs = np.random.randint(0, buffer_size, size=1)
                        if idxs[0] not in ID_indx :
                            ID_indx.append(idxs[0])
                            break

                for i in range( self.batch_size-2):
                    o, a, r, o2= self.state_buffer[idxs[0]], self.action_buffer[idxs[0]], self.reward_buffer[idxs[0]], self.next_state_buffer[idxs[0]]
                    v_temp=np.concatenate((o2-o, a, r))
                    # 计算对应元素相加
                    v_0 = v_0 + v_temp
                    # 构造新向量 q'
                    q_prime = np.concatenate((v_0, [i+2]))  # 将总和和常数2组合
                    q_prime = q_prime.reshape(1, -1)
                
                    D, I = self.hnsw_index8.search(q_prime, int(self.batch_size))
                    
                    # 遍历列表 B，添加第一次未在 A 中出现的元素
                    flag_append=0
                    for item in I[0]:
                        if item not in ID_indx :
                            ID_indx.append(item)
                            flag_append=1
                            break  # 找到第一个未出现的元素后退出循环
                    if flag_append==0:
                        randome_number=randome_number+1
                        
                        # 生成所有可能的索引
                        all_indices = np.arange(buffer_size)
                        
                        # 排除 ID_indx 中的索引
                        available_indices = np.setdiff1d(all_indices, ID_indx)
                        #available_indices = np.setdiff1d(all_indices, self.used_queue)
                        
                        # 随机选择一个索引
                        random_index = np.random.choice(available_indices)
                        ID_indx.append(random_index)

                print('randome_number',randome_number)
                if len(ID_indx) != self.batch_size:
                    print('!!!!! len(ID_indx) != self.batch_size')

                print(ID_indx)
                return (ID_indx)

        # We compute the loss and update parameters
        def learn(self):
            # Get sampling range
            #record_range = min(self.buffer_counter, self.buffer_capacity)
            # Randomly sample indices
            #batch_indices = np.random.choice(record_range, self.batch_size)
            batch_indices=self.sample()

            #print('learn function:',batch_indices)

            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
            reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

            self.update(state_batch, action_batch, reward_batch, next_state_batch)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_actor():
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic():
        # State as input
        state_input = layers.Input(shape=(num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(state, noise_object, previous_actions):
        sampled_actions = tf.squeeze(actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise
        logging.debug(f"sampled_actions: {sampled_actions}")

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

        action_weights = [np.squeeze(legal_action)]
        print(action_weights)
        max_action_weight = -1000
        nr_action = len(action_weights[0])
        start_action = random.randrange(0, nr_action)
        action = 0
        logging.debug(f"start_action {start_action}")
        # pick up the action which has the highest weight
        for action_idx in range(0, nr_action):
            current_action = (start_action + action_idx) % nr_action
            if (current_action not in previous_actions) and (action_weights[0][current_action] > max_action_weight):
                max_action_weight = action_weights[0][current_action]
                action = current_action

        # action = np.random.choice(np.flatnonzero(action_weights == max(action_weights)))
        # action = np.argmax(action_weights)
        return action

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005

    MaximumDiversity=tuning_config['MaximumDiversity']

    buffer = Buffer(1000000, 32,MaximumDiversity)

    # set the horizon
    env.horizon = tuning_config['horizon']
    duration_in_seconds = tuning_config['duration'] * 3600
    duration_in_episodes=tuning_config['duration']

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    episodes = []

    start_time = time.time()
    current_time = time.time()

    ep = 0

    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(log_dir='results/'+(datetime.now()+ timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S_")+"MaximumDiversity_"+str(MaximumDiversity))



    #while (current_time - start_time) < duration_in_seconds:
    while ep<duration_in_episodes:
        # start a new episode
        prev_state = env.reset()
        episodic_reward = 0
        previous_actions = []
        while True:
            # Uncomment this to see the Actor in action
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            # select action according to policy
            action = policy(tf_prev_state, ou_noise, previous_actions)
            previous_actions.append(action)
            # receive state and reward from environment.
            state, reward, done, info = env.step(action)
            print('ep:',ep,' reward:',reward)

            buffer.record((prev_state, action, reward, state,done))
            episodic_reward += reward

            ep += 1

            if ep>100:
                buffer.learn()
                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)

            # End this episode when `done` is True
            if done:
                current_time = time.time()
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)
        writer.add_scalar('Reward', episodic_reward, ep)
        episodes.append(ep)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        logging.info(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")
        avg_reward_list.append(avg_reward)
        writer.add_scalar('Avg_Reward', avg_reward, ep)
        logging.info(f"episode: {ep}")
        logging.info(f"evaluate duration: {(current_time - start_time)} s")
        env.print_state_summary(env.best_state)
    # 关闭SummaryWriter
    writer.close()
    
    # 保存到 CSV 文件
    #data = {'Episode': episodes, 'Avg Reward': avg_reward_list, 'Episodic Reward': ep_reward_list}
    #df = pd.DataFrame(data)
    #now = (datetime.now()+ timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S_")
    #df.to_csv('/mnt/share/lixu/UDO-master/'+now+'avg_rewards.csv', index=False)  # 不保存索引
