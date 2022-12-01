import numpy as np
import os
from exp_utils import mkdir


def file_exists(filepath):
    return os.path.isfile(filepath)


def save(net_vars, directory, filename):
    mkdir(directory)
    if len(net_vars) == 0:
        raise Exception("At least one variable is expected")
    var_dict = {}
    for var_ in net_vars:
        #print(str(var_.name))
        var_dict[str(var_.name)] = np.array(var_.value())
    np.savez(directory + "/" + filename + ".npz", **var_dict)


def load(net_vars, directory, filename):
    ending = ".npz"
    if filename.endswith(".npz"):
        ending = ""

    filepath = directory + "/" + filename + ending
    if not file_exists(filepath):
        raise Exception("File path '" + filepath + "' does not exist")
    model_data = np.load(filepath, allow_pickle=True)
    if len(net_vars) != len(model_data):
        keys = list(model_data.keys())
        print("Expected:", len(net_vars), "layer; Got:", len(model_data), "layer, file:", filepath)
        if len(net_vars) == 0 or len(model_data) == 0:
            raise Exception("You have to apply a prediction with, e.g., random data to initialize the weights of the network.")
        for i in range(min(len(net_vars), len(model_data))):
            print(net_vars[i].name, "\t", keys[i])
        print("Expected:")
        for i in range(len(net_vars)):
            print(net_vars[i].name)
        raise Exception("data mismatch")
    i = 0
    for key, value in model_data.items():
        varname = str(net_vars[i].name)
        if np.isnan(value).any():
            raise Exception("loaded value is NaN")
        if key != varname:
            raise Exception(
                "Variable names mismatch: " + key + ", " + varname)
        net_vars[i].assign(value)
        i += 1