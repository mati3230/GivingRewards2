import os
import numpy as np
import h5py


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def save_exp_data(fdir, fname, exp_dict):
    if fdir[-1] != "/":
        fdir += "/"
    if not fname.endswith(".h5"):
        fname += ".h5"
    hf = h5py.File("{0}{1}".format(fdir, fname), "w")
    for k, v in exp_dict.items():
        hf.create_dataset(k, data=v)
    hf.close()


def load_exp_data(fdir, fname):
    if fdir[-1] != "/":
        fdir += "/"
    if not fname.endswith(".h5"):
        fname += ".h5"
    hf = h5py.File("{0}{1}".format(fdir, fname), "r")
    #print(np.array(hf["acc_gnn"], copy=True))
    exp_dict = {
        "xyz": np.array(hf["xyz"], copy=True),
        "rgb": np.array(hf["rgb"], copy=True),
        "sortation": np.array(hf["sortation"], copy=True),
        "node_features": np.array(hf["node_features"], copy=True),
        "senders": np.array(hf["senders"], copy=True),
        "receivers": np.array(hf["receivers"], copy=True),
        "p_gt": np.array(hf["p_gt"], copy=True),
        "p_cp": np.array(hf["p_cp"], copy=True)
    }
    sp_idxs = []
    size = np.unique(exp_dict["senders"]).shape[0]
    for i in range(size):
        sp = np.array(hf[str(i)], copy=True)
        sp_idxs.append(sp)
    hf.close()
    return exp_dict, sp_idxs


def get_unions(graph_dict, alpha):
    senders = graph_dict["senders"]
    receivers = graph_dict["receivers"]

    unions = np.zeros((senders.shape[0], ), dtype=np.bool)
    for i in range(senders.shape[0]):
        e_i = senders[i]
        e_j = receivers[i]
        unions[i] = alpha[e_i] == alpha[e_j]
    return unions