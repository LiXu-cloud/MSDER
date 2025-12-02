import numpy as np

class EDER:
    def __init__(self, obs_dim, act_dim, size, diversity_threshold=0.1, group_size=2):
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        :param diversity_threshold: threshold to determine diversity
        :param group_size: size of each group for matrix formation
        """
        print("EDER  __init__")
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.diversity_threshold = diversity_threshold
        self.group_size = group_size
        
        
        self.diversity_values = []  # To store diversity values for each group
        self.total_diversity = 0.0  # Global variable to maintain total diversity


    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.size % self.group_size == 0:
            group_indices = np.arange( self.ptr- self.group_size,self.ptr)
            matrix = self.obs_buf[group_indices]
            norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
            normalized_matrix = matrix / norm_matrix  # Avoid division by zero if needed
            L_matrix=np.dot(normalized_matrix,normalized_matrix.T)
            L_C = self.decompose_matrix(L_matrix)
            if L_C is not None:
                self.calculate_diversity_value(L_C)
            else:
                self.diversity_values.append(0)
                self.total_diversity += 0

    #rejection sampling is applied to select  samples with higher diversity
    def is_diverse(self, new_obs):
        if self.size == 0:
            return True
        distances = np.linalg.norm(self.obs_buf[:self.size] - new_obs, axis=1)
        return np.min(distances) > self.diversity_threshold


    def decompose_matrix(self, matrix):
        """
        Perform Cholesky decomposition on the given matrix
        """
        try:
             # Check if the matrix is positive definite
            if np.all(np.linalg.eigvals(matrix) > 0):
                L_C = np.linalg.cholesky(matrix)  # Perform Cholesky decomposition
                return L_C
            else:
                print("Covariance matrix is not positive definite. Adding regularization.")
                matrix += np.eye(matrix.shape[0]) * 1e-5  # Add a small regularization term
                L_C = np.linalg.cholesky(matrix)


           # L_C = np.linalg.cholesky(np.cov(matrix.T))  # Covariance matrix of the states
            return L_C
        except np.linalg.LinAlgError:
            print("Cholesky decomposition failed for the matrix.")
            return None

    def calculate_diversity_value(self, L_C):
        """
        Calculate the diversity value based on the Cholesky decomposed matrix
        """
        diag_elements = np.diagonal(L_C)
        determinant = np.prod(diag_elements ** 2)  # Product of squares of diagonal elements
        self.diversity_values.append(determinant)
        self.total_diversity += determinant  # Update the global diversity total

    def sample(self, batch_size=32):
        """
        Sample a mini-batch from the buffer based on diversity values
        :param batch_size: size of minibatch
        :return: mini-batch data as a dictionary
        """
        if self.size<batch_size:
            idxs = np.random.randint(0, self.size, size=batch_size)
            return idxs
            return dict(obs1=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)

        print('EDER sample function: self.size>=batch_size')
        num_groups_to_sample = batch_size // self.group_size  # Number of groups to sample

        if self.total_diversity == 0 or num_groups_to_sample == 0:
            # If total diversity is zero or batch size is too small, sample uniformly
            group_indices = np.random.choice(self.size // self.group_size, size=num_groups_to_sample, replace=False)
        else:
            # Calculate probabilities based on diversity values
            probabilities = np.array(self.diversity_values) / self.total_diversity
            group_indices = np.random.choice(self.size // self.group_size, size=num_groups_to_sample, replace=False, p=probabilities)

        # Collect the sampled data
        idxs=[]
        obs1_batch = []
        obs2_batch = []
        acts_batch = []
        rews_batch = []
        done_batch = []
        
        for idx in group_indices:
            group_start = idx * self.group_size
            idxs.extend(range(group_start,group_start + self.group_size))
            obs1_batch.append(self.obs_buf[group_start:group_start + self.group_size])
            obs2_batch.append(self.obs2_buf[group_start:group_start +self.group_size])
            acts_batch.append(self.acts_buf[group_start:group_start + self.group_size])
            rews_batch.append(self.rews_buf[group_start:group_start + self.group_size])
            done_batch.append(self.done_buf[group_start:group_start + self.group_size])
        print('idxs:',idxs)
        return idxs
        return dict(obs1=np.concatenate(obs1_batch),
                    obs2=np.concatenate(obs2_batch),
                    acts=np.concatenate(acts_batch),
                    rews=np.concatenate(rews_batch),
                    done=np.concatenate(done_batch))


    def get_buffer_size(self):
        return self.size

    def get_one(self, id):
        return self.obs_buf[id], self.acts_buf[id], self.rews_buf[id], self.done_buf[id]

    def get_state_matrices(self):
        """
        Get the list of state matrices for each group
        :return: list of state matrices
        """
        return self.state_matrices

    def get_decomposed_matrices(self):
        """
        Get the list of Cholesky decomposed matrices
        :return: list of decomposed matrices
        """
        return self.decomposed_matrices

    def get_diversity_values(self):
        """
        Get the list of diversity values for each group
        :return: list of diversity values
        """
        return self.diversity_values

    def get_total_diversity(self):
        """
        Get the total diversity value
        :return: total diversity value
        """
        return self.total_diversity