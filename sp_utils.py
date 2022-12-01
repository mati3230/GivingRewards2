import numpy as np


def p_vec_from(n, sp_idxs):
    p_vec = -np.ones((n, ), dtype=np.int32)
    for i in range(len(sp_idxs)):
        P_idxs = sp_idxs[i]
        p_vec[P_idxs] = i
    return p_vec

def get_points_neighbours(sp_idxs, sndrs, rcvrs):
    sortation = np.argsort(sndrs)
    senders = sndrs[sortation]
    receivers = rcvrs[sortation]
    uni_senders, uni_idxs, uni_counts = np.unique(senders, return_index=True, return_counts=True)
    
    point_idxs_list = []
    neighbours_list = []

    for i in range(uni_senders.shape[0]):
        node = uni_senders[i]
        start = uni_idxs[i]
        stop = start + uni_counts[i]
        P_idxs = sp_idxs[i]
        neighbours = np.unique(receivers[start:stop])
        point_idxs_list.append(P_idxs)
        neighbours_list.append(neighbours)
    pns = (point_idxs_list, neighbours_list)
    return pns