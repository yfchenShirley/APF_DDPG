import numpy as np
import heapq
import os





class Transition:
    def __init__(self, episode_reward, experience, terminal):
        self.data = (episode_reward, experience, terminal)

    def __lt__(self, other):
        return self.data[0] < other.data[0]


class PriorityMemory:
    def __init__(self, memory_size: int, batch_size: int = 32, good_ratio: float=0.1):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.buffer = []
        self.good_ratio = good_ratio
        
    def add(self, experience, episode_reward, terminal):
        transition = Transition(episode_reward, experience, terminal)
        if len(self.buffer) == self.memory_size:
            lowest = heapq.heappop(self.buffer)
            # print('lowest episode reward : ', lowest.data[0])
            if lowest.data[0] < episode_reward:
                heapq.heappush(self.buffer, transition)
            else:
                heapq.heappush(self.buffer, lowest)
        else:
            heapq.heappush(self.buffer, transition)
    
    def get_lowest_rewards(self):
        lowest = heapq.heappop(self.buffer)
        heapq.heappush(self.buffer, lowest)
        return lowest.data[0]


    def sample(self, continuous=False):

        buffsize = len(self.buffer)
        ratio = self.good_ratio
        if buffsize < (self.batch_size//2)/ratio:
            return [], []

        good_data = heapq.nlargest(int(buffsize*ratio),enumerate(self.buffer),key=lambda x:x[1])
        good_indices, good_trans = zip(*good_data)
        bad_indices = [item for item in list(range(buffsize)) if item not in good_indices]
        threshold = good_trans[-1].data[0]

        if good_trans[0].data[0] == self.buffer[bad_indices[-1]].data[0]:
            return [], []

        good_indexes = np.random.choice(good_indices, size=self.batch_size//2, replace=False)
        bad_indexes = np.random.choice(bad_indices, size=self.batch_size//2, replace=False)

        #import pdb; pdb.set_trace()
        good_traj, bad_traj = [], []
        for i in np.concatenate((good_indexes,bad_indexes)):# or (not self.buffer[i].data[2])
            if (self.buffer[i].data[0] < threshold):
                bad_traj += self.buffer[i].data[1]
            else:
                good_traj += self.buffer[i].data[1]

        return good_traj, bad_traj

    def get_nlargest_trajectories(self, N_good):
        good_data = heapq.nlargest(int(N_good),enumerate(self.buffer),key=lambda x:x[1])
        good_indices, good_trans = zip(*good_data)
        for index in good_indices:
            print(f"index {index}, episodic reward: {self.buffer[index].data[0]}, done: {self.buffer[index].data[2]}")



    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def save(self, folder_name):
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/apf_buffer.npy', self.buffer)

    def load(self, folder_name):
        self.buffer = np.load(folder_name + '/apf_buffer.npy', allow_pickle=True).tolist()
