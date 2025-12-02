# Adapted REDQSAC agent to use diffusion samples.

import numpy as np
from redq.algos.core import ReplayBuffer
from redq.algos.redq_sac import REDQSACAgent
from torch import Tensor

import faiss
from collections import deque

def combine_two_tensors(tensor1, tensor2):
    return Tensor(np.concatenate([tensor1, tensor2], axis=0))


class REDQRLPDAgent(REDQSACAgent):

    def __init__(self,vector_selection=0 ,index_type=0,normalization=0, diffusion_random=0,diffusion_buffer_size=int(1e6), diffusion_sample_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=diffusion_buffer_size)
        self.diffusion_sample_ratio = diffusion_sample_ratio
        self.vector_selection=vector_selection
        self.index_type=index_type
        self.normalization=normalization
        self.diffusion_random=diffusion_random
        self.used_queue=deque()
        self.used_queue_max_length=2000
        self.diffusion_buffer_used_queue=deque()
        self.diffusion_buffer_used_queue_max_length=1000
        if vector_selection == 1:  # o2-o1,a,r,d,1
            if index_type == 1:
                self.diversity_index=faiss.IndexHNSWFlat(self.obs_dim + self.act_dim + 1 +1+1, 32, faiss.METRIC_INNER_PRODUCT)
                self.diversity_diffusion_index=faiss.IndexHNSWFlat(self.obs_dim + self.act_dim + 1 +1+1, 32, faiss.METRIC_INNER_PRODUCT)
            elif index_type ==2:
                self.diversity_index=faiss.IndexFlatIP(self.obs_dim + self.act_dim + 3)
                self.diversity_diffusion_index=faiss.IndexFlatIP(self.obs_dim + self.act_dim + 3)
        elif vector_selection == 2:  # o1,o2,a,r,d,1
            if index_type == 1:
                self.diversity_index=faiss.IndexHNSWFlat(self.obs_dim + self.act_dim+ self.obs_dim+3, 32, faiss.METRIC_INNER_PRODUCT)
                self.diversity_diffusion_index=faiss.IndexHNSWFlat(self.obs_dim + self.act_dim+ self.obs_dim+3, 32, faiss.METRIC_INNER_PRODUCT)
            elif index_type ==2:
                self.diversity_index=faiss.IndexFlatIP(self.obs_dim + self.act_dim+ self.obs_dim+3)
                self.diversity_diffusion_index=faiss.IndexHNSWFlat(self.obs_dim + self.act_dim+ self.obs_dim+3, 32, faiss.METRIC_INNER_PRODUCT)
    
    def used_queue_add(self, item):
        self.used_queue.append(item)  # 
        if len(self.used_queue) > self.used_queue_max_length:
            self.used_queue.popleft()  # 

    def sample_data(self, batch_size):
        #print('REDQRLPDAgent sample_data')
        diffusion_batch_size = int(batch_size * self.diffusion_sample_ratio)
        online_batch_size = int(batch_size - diffusion_batch_size)
        # Sample from the diffusion buffer
        if self.diffusion_buffer.size < diffusion_batch_size:
            if self.vector_selection == 0:
                return super().sample_data(batch_size)
            else:
                batch = self.replay_buffer.sample_batch(batch_size,self.diversity_search(self.replay_buffer,self.diversity_index, batch_size))
                obs_tensor = Tensor(batch['obs1']).to(self.device)
                obs_next_tensor = Tensor(batch['obs2']).to(self.device)
                acts_tensor = Tensor(batch['acts']).to(self.device)
                rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
                done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
                return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
    
                return self.diversity_search(self.replay_buffer,self.diversity_index, batch_size)
        
        if self.vector_selection == 0:
            diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
            online_batch = self.replay_buffer.sample_batch(batch_size=online_batch_size)
        else:
            if self.diffusion_random==0:   
                diffusion_batch_idxs=self.diversity_search(self.diffusion_buffer,self.diversity_diffusion_index, diffusion_batch_size)
                diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size,idxs=diffusion_batch_idxs)
            else:
                diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
            online_batch_idxs=self.diversity_search(self.replay_buffer,self.diversity_index, online_batch_size)
            online_batch = self.replay_buffer.sample_batch(batch_size=online_batch_size,idxs=online_batch_idxs)
        obs_tensor = combine_two_tensors(online_batch['obs1'], diffusion_batch['obs1']).to(self.device)
        obs_next_tensor = combine_two_tensors(online_batch['obs2'], diffusion_batch['obs2']).to(self.device)
        acts_tensor = combine_two_tensors(online_batch['acts'], diffusion_batch['acts']).to(self.device)
        rews_tensor = combine_two_tensors(online_batch['rews'], diffusion_batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = combine_two_tensors(online_batch['done'], diffusion_batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def add_vector_to_index(self, myIndex, o, a, r, o2, d):
        if self.vector_selection == 1:
            newBuffer=np.concatenate((o2-o,a,np.array([r]),np.array([d])))
            newBuffer=newBuffer.reshape(1,-1)
            if self.normalization==1:
                newBuffer /= np.linalg.norm(newBuffer, axis=1, keepdims=True)
            #  
            t_prime = np.concatenate([-2 * newBuffer[0], [np.linalg.norm(newBuffer)**2]])
            #print('t_prime',t_prime)
            
        t_prime = t_prime.reshape(1, -1)

        myIndex.add(t_prime)
        print('our index points number:',myIndex.ntotal)

    def reset_diffusion_buffer(self):
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                             size=self.diffusion_buffer.max_size)

    def diversity_search(self,my_replay_buffer,myIndex, batch_size):
        #print('REDQRLPDAgent diversity_search')
        print('batch_size',batch_size)
        randome_number=0

        ID_indx=[]

        buffer_size=my_replay_buffer.get_buffer_size()
        idxs = np.random.randint(0, buffer_size, size=1)
        randome_number=randome_number+1
        #print(idxs)
        ID_indx.append(idxs[0])
        #self.used_queue_add(idxs[0])

        o, a, r, o2, d = my_replay_buffer.get_one(idxs[0])

        if self.vector_selection==1:
            v_0=np.concatenate((o2-o, a,np.array([r]),np.array([d])))
        elif self.vector_selection==2:
            v_0=np.concatenate((o,o2, a,np.array([r]),np.array([d])))
        #print(v_0)
        q = np.concatenate((v_0, [1]))
        q = q.reshape(1, -1)
        

        D, I = myIndex.search(q, int(batch_size/4))
        
        flag_append=0
        for item in I[0]:
            if item not in ID_indx:# and item not in self.used_queue:
                ID_indx.append(item)
                #self.used_queue_add(item)
                flag_append=1
                break  # 
        if flag_append==0:
            randome_number=randome_number+1
            while True:
                idxs = np.random.randint(0, buffer_size, size=1)
                if idxs[0] not in ID_indx:# and idxs[0] not in self.used_queue:
                    ID_indx.append(idxs[0])
                    #self.used_queue_add(idxs[0])
                    break
        for i in range( batch_size-2):
            #print(i,self.batch_size,':',ID_indx)
            o, a, r, o2, d = my_replay_buffer.get_one(ID_indx[i+1])
            if self.vector_selection==1:
                v_temp=np.concatenate((o2-o, a, np.array([r]),  np.array([d])))
            elif self.vector_selection==2:
                v_temp=np.concatenate((o,o2, a, np.array([r]),  np.array([d])))
            # 
            v_0 = v_0 + v_temp
            #print(v_0)


            q_prime = np.concatenate((v_0, [i+2]))  # 
            q_prime = q_prime.reshape(1, -1)
            #if self.normalization==1:
            #    q_prime /= np.linalg.norm(q_prime, axis=1, keepdims=True)
            #q_prime[0][-3]=r
        
            D, I = myIndex.search(q_prime, int(batch_size/4))
            
            flag_append=0
            for item in I[0]:
                if item not in ID_indx:# and item not in self.used_queue:
                    ID_indx.append(item)
                    #self.used_queue_add(item)
                    flag_append=1
                    break  # 
            if flag_append==0:
                randome_number=randome_number+1

                if buffer_size<2000:
                    
                    all_indices = np.arange(buffer_size)
                    
                    available_indices = np.setdiff1d(all_indices, ID_indx)
                    available_indices = np.setdiff1d(available_indices, self.used_queue)
                    
                    
                    random_index = np.random.choice(available_indices)
                    ID_indx.append(random_index)
                    self.used_queue_add(random_index)

                else:
                    while True:
                        idxs = np.random.randint(0, buffer_size, size=1)
                        if idxs[0] not in ID_indx and idxs[0] not in self.used_queue:
                            ID_indx.append(idxs[0])
                            self.used_queue_add(idxs[0])
                            break
               

        print('randome_number',randome_number)
        if len(ID_indx) != batch_size:
            print('!!!!! len(ID_indx) != batch_size')

        #obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size,ID_indx)
        #print(type(ID_indx))
        #print(ID_indx)
        return ID_indx

        batch = my_replay_buffer.sample_batch(batch_size,ID_indx)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor