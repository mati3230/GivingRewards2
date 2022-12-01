import numpy as np
import gym
from gym import spaces
import h5py
import os
from collections import deque

from exp_utils import load_exp_data
from sp_utils import p_vec_from, get_points_neighbours
from superpoint_graph import SuperpointGraph
from partition.partition import Partition
from partition.density_utils import densities_np, densities_np_osize


class SuperpointGrowingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, n_scenes=1, mapping=[3,2,1], ignore=[], dataset="scannet", density_method="abs", obj_punish=0.25):
        super(SuperpointGrowingEnv, self).__init__()
        if dataset != "scannet" and dataset != "pcg":
            raise Exception("Only 'scannet' and 'pcg' can be used as datasets")
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.dataset = dataset
        self.action_space = spaces.Discrete(2)
        hf = h5py.File("./exp_stats_" + dataset + ".h5", "r")
        if dataset == "pcg":
            self.min_f = np.array(hf["min"])[:-1]
            self.max_f = np.array(hf["max"])[:-1]
        else:
            self.min_f = np.array(hf["min"])
            self.max_f = np.array(hf["max"])
        shape = self.min_f.shape
        #diff_f = np.abs(self.max_f - self.min_f)
        #high = np.max(diff_f)
        self.observation_space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        self.exp_dir = "./exp_data_" + dataset
        self.exp_files = os.listdir(self.exp_dir)
        self.n_scenes = n_scenes
        self.current_scene = 0
        self.last_scene = -1
        if n_scenes == -1:
            self.n_scenes = len(self.exp_files)
        self.classes = [3,2,1]
        self.mapping = mapping
        self.remapping = True
        if self.classes == self.mapping:
            self.remapping = False
        self.ignore = ignore
        self.actions = []
        self.initial_reward = 0
        self.density_method = density_method
        self.obj_punish = obj_punish


    def compute_reward(self):
        #if not self.done:
        #    return 0
        partition = Partition(partition=self.partition_vec)
        if self.density_method == "abs":
            density_func = densities_np
        else:
            density_func = densities_np_osize
        classification, densities, differences = self.partition_gt.classify(
            partition_B=partition, density_function=density_func, return_differences=True)
        
        if len(self.ignore) > 0:
            for ig in self.ignore:
                classification[classification == ig] = 0

        # sum of densities of all free superpoints
        no_match_densities = 0
        for i in range(classification.shape[0]):
            # only zeros -> is_free
            is_free = np.unique(classification[i]).shape[0] == 1
            if is_free:
                no_match_densities += np.sum(densities[i])

        matches = np.zeros(classification.shape, dtype=bool)
        matches[classification > 0] = True

        S = partition.n_uni
        Omega = self.partition_gt.n_uni
        L = max(S/Omega, Omega/S)

        if self.remapping:
            for i in range(len(self.classes)):
                c = self.classes[i]
                classification[classification == c] = self.mapping[i]

        if self.density_method == "abs":
            max_sum = np.sum(self.mapping[0] * self.partition_gt.counts)
        else:
            max_sum = np.sum(self.mapping[0] * self.partition_gt.n_uni)
        reward = np.sum(classification * densities)
        reward /= max_sum
        if self.done:
            # substract number of non-matched points and densities 
            # of matched superpoints with other objects, divided by the number of points
            # TODO check if matches * differences is correct
            if self.density_method == "abs":
                reward -= (np.sum(matches * differences) + no_match_densities) / self.n_points
            else:
                reward -= (np.sum(matches * differences) + no_match_densities)
            # punishment for false partition size
            reward -= (self.obj_punish * L)
            # punishment for doing nothing
            if np.sum(self.actions) <= 10:
                reward -= 100

        return reward


    def update_partition_vec(self):
        """Update the superpoint value of the main superpoint. This function can
        be used after a union."""
        n_sp = self.superpoint_graph.get_n_superpoints()
        P_idxs = self.superpoint_graph.get_sp_point_idxs(self.main_sp_idx)

        self.partition_vec[P_idxs] = self.main_sp_idx


    def on_union(self):
        """
        update the neighbours_to_visit list
        already considered neighbours will not be in this array
        """
        neighbours = self.superpoint_graph.get_neighbours_of_superpoint(self.main_sp_idx)
        self.neighbours_to_visit.clear()
        for idx in neighbours:
            self.neighbours_to_visit.append(idx)


    def unify_superpoints(self, action):
        """Union of neighbour with main superpoint.

        Parameters
        ----------
        action : int
            Should the superpoints be unified?
        """
        #print("main_idx", self.main_sp_idx, "neighb_idx", self.neighbour_sp_idx)
        if action == 1:
            # remove that neighbour from the todo list
            if self.neighbour_sp_idx in self.to_do:
                self.to_do.remove(self.neighbour_sp_idx)

            self.superpoint_graph.unify(self.main_sp_idx, self.neighbour_sp_idx)

            self.on_union()
            self.update_partition_vec()
            #if not self._stepwise_reward:
            #    self.compute_reward()
        else:
            self.superpoint_graph.break_connection(self.main_sp_idx, self.neighbour_sp_idx)


    def step(self, action):
        self.actions.append(action)
        self.unify_superpoints(action)
        self.setup(action)
        reward = self.last_reward
        if action == 1 or self.done:
            reward = self.compute_reward()
        delta_reward = reward - self.last_reward
        self.last_reward = reward
        observation = self.next_state()
        info = {}
        done = self.done
        return observation, delta_reward, done, info


    def set_main_sp(self):
        self.main_sp_idx = self.to_do.popleft()
        #print("main", self.main_sp_idx)


    def next_neighbour(self):
        """ Sets the next neighbour as union candidate. """
        self.neighbour_sp_idx = self.neighbours_to_visit.pop()
        #print("neighbour", self.neighbour_sp_idx)


    def next_superpoint(self):
        """Select next the main superpoint and its neighbours."""
        self.neighbours_to_visit = deque()
        # no more work - done
        if len(self.to_do) == 0:
            self.setup(action=0, last_one=True)
            return

        # set main superpoint and the neighbours
        self.set_main_sp()
        neighbour_idxs = self.superpoint_graph.get_neighbours_of_superpoint(
            self.main_sp_idx)
        for idx in neighbour_idxs:
            self.neighbours_to_visit.append(idx)

        # if no neighbours are available
        if len(self.neighbours_to_visit) == 0:
            last_one = len(self.to_do) == 0

            self.setup(action=0, last_one=last_one)
            return
        # set next merging candidate
        self.next_neighbour()


    def setup(self, action, last_one=False):
        """Prepare to superpoint the next object.

        Parameters
        ----------
        action : int
            Should the superpoints be unified?
        last_one : boolean
            Last action in the environment.
        """
        if last_one:
            self.done = True
        else:
            if len(self.neighbours_to_visit) == 0:
                # set the next main superpoint
                self.next_superpoint()
            else:
                # set the next neighbour superpoint that can be unified
                self.next_neighbour()


    def reset(self):
        self.current_scene += 1
        if self.current_scene >= self.n_scenes:
            self.current_scene = 0
        self.done = False
        if self.current_scene != self.last_scene:
            #print("load")
            # init superpoint graph
            self.exp_dict, self.sp_idxs = load_exp_data(
                fdir=self.exp_dir, fname=self.exp_files[self.current_scene])
            pns = get_points_neighbours(sp_idxs=self.sp_idxs, sndrs=self.exp_dict["senders"],
                rcvrs=self.exp_dict["receivers"])
            self.superpoint_graph = SuperpointGraph(pns_orig=pns)
            p_vec_gt = self.exp_dict["p_gt"]
            self.partition_gt = Partition(partition=p_vec_gt)
            self.n_points = self.exp_dict["xyz"].shape[0]
            self.last_scene = self.current_scene
            self.partition_vec = p_vec_from(n=self.n_points, sp_idxs=self.sp_idxs)
            self.initial_reward = self.compute_reward()
        self.last_reward = self.initial_reward


        self.superpoint_graph.reset()

        # inits
        self.to_do = deque()
        self.to_do.extend(list(range(self.superpoint_graph.get_n_superpoints())))
        self.neighbours_to_visit = deque()
        self.main_sp_idx = 0
        self.neighbour_sp_idx = 0
        self.actions = []


        self.next_superpoint()

        observation = self.next_state()
        return observation
    

    def next_state(self):
        node_features = self.exp_dict["node_features"]
        if self.dataset == "pcg":
            f1 = node_features[self.main_sp_idx][:-1]
            f2 = node_features[self.neighbour_sp_idx][:-1]
        else:
            f1 = node_features[self.main_sp_idx]
            f2 = node_features[self.neighbour_sp_idx]
        f = np.abs(f1 - f2) / (self.max_f - self.min_f)
        return f.astype(np.float32)


    def render(self, mode="human"):
        return
    

    def close (self):
        return


"""
from stable_baselines3.common.env_checker import check_env
if __name__ == "__main__":
    env = SuperpointGrowingEnv(dataset="pcg")
    check_env(env)
    for i in range(10):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            action = np.random.randint(low=0, high=2, size=1)[0]
            #action = 0
            obs, reward, done, info = env.step(action=action)
            #print(reward)
            step += 1
            if done:
                print(i, step, reward)
                break
"""
