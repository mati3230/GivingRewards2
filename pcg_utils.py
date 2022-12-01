import os
import numpy as np
def get_scenes(blacklist=[]):
    pcg_dir = "C:/Users/MIREVI User/AppData/LocalLow/POG/SceneGenerator/data"
    scenes = os.listdir(pcg_dir)
    return scenes, pcg_dir

def get_ground_truth(pcg_dir, scene):
    pc = np.loadtxt(pcg_dir + "/" + scene, delimiter=";", skiprows=1)
    xyz = pc[:, :3]
    rgb = pc[:, 3:6]
    label_col = -2
    count_col = -1
    uni_labels = np.unique(pc[:, label_col])
    nr = 0
    p_vec_gt = pc[:, count_col]
    p_vec_gt = p_vec_gt.astype(np.uint32)
    for i in range(uni_labels.shape[0]):
        label = uni_labels[i]
        label_idxs = np.where(pc[:, label_col] == label)[0]
        p_vec_gt[label_idxs] += nr
        max_count = np.max(pc[label_idxs, count_col])
        nr += int(max_count)
    return xyz, rgb, p_vec_gt