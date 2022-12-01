import numpy as np
import argparse
import os

import tensorflow as tf
try:
    from stable_baselines3 import PPO, DQN
except:
    print("Change the conda env to use the stable baselines")
try:
    from graph_nets import utils_tf
    import graph_nets as gn
    from partition.felzenszwalb import partition_from_probs
except:
    print("Change the conda env to use the graph_nets")

from superpoint_growing_env import SuperpointGrowingEnv
from exp_stats import render_point_cloud
from network_utils import load
from mlp import MLP


def graph_conv(model_fn, input_graphs, training):
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    temporary_graph_sent = input_graphs.replace(edges=nodes_at_sender_edges)
    nodes_with_aggregated_edges = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_mean)(temporary_graph_sent)

    updated_nodes = model_fn(0.5 * (input_graphs.nodes + nodes_with_aggregated_edges), is_training=training)
    output_graphs = input_graphs.replace(nodes=updated_nodes)
    return output_graphs


def fast_dot(a, b):
    dot = a*b
    dot = tf.reduce_sum(dot, axis=-1)
    a_n = tf.linalg.norm(a, axis=-1)
    b_n = tf.linalg.norm(b, axis=-1)
    dot /= ((a_n * b_n) + 1e-6)
    return dot, a_n, b_n



def get_method(mdir, mfile):
    mname = mfile.split(".")[0]
    if mname == "best_model":
        method_all = mdir.split("/")[-1]
        method_all = method_all.split("_")
        dataset = method_all[0]
        opti_alg = method_all[1]
        method = "drl"
        return method, opti_alg
    mname = mname.split("_")[0]
    if mname == "mlp":
        method = "imilearn"
    elif mname == "gnn":
        method = "gnn"
    else:
        raise Exception("Unknown Method: '{0}', mdir: '{1}', mfile: '{2}'".format(mname, mdir, mfile))
    return method, None


def get_input_graphs(env, dataset):
    exp_dict = env.exp_dict
    senders = exp_dict["senders"]
    receivers = exp_dict["receivers"]
    node_features = exp_dict["node_features"]
    if dataset == "pcg":
        node_features = node_features[:, :-1]

    graph_dict = {
        "nodes": node_features,
        "senders": senders,
        "receivers": receivers
    }
    
    input_graphs = {"nodes": node_features, "senders": senders, "receivers": receivers, "edges": None, "globals": None}
    input_graphs = utils_tf.data_dicts_to_graphs_tuple([input_graphs])
    return input_graphs, graph_dict


def conv(input_graphs, mlps):
    x = input_graphs
    for j in range(len(mlps)):
        x = graph_conv(model_fn=mlps[j], input_graphs=x, training=False)
    return x

def load_gnn(dataset, env, mdir, mfile):
    net_arch = [8,4]
    if dataset == "pcg":
        net_arch = [7,4]
    activations = len(net_arch) * [tf.nn.relu]

    mlps = []
    for i in range(len(net_arch)):
        layer_size = net_arch[i]
        activation = activations[i]
        mlp = MLP(layer_dims=[layer_size], activations=[activation], name="mlp"+str(i), dropout=0)
        mlps.append(mlp)
    obs = env.reset()

    input_graphs, _ = get_input_graphs(env=env, dataset=dataset)

    conv(input_graphs=input_graphs, mlps=mlps)

    names = mfile.split("_")
    for j in range(len(mlps)):
        name = names[0] + "_" + str(j) + "_" + names[-1]
        print("Load file: '{0}'".format(name))
        load(net_vars=mlps[j].variables, directory=mdir, filename=name)
    return mlps

def load_imilearn(dataset, env, mdir, mfile):
    net_arch = [8,8,4,1]
    if dataset == "pcg":
        net_arch = [7,7,4,1]
    activations = (len(net_arch)-1) * [tf.nn.relu]
    activations.append(tf.math.sigmoid)
    mlp = MLP(layer_dims=net_arch, activations=activations, name="mlp", dropout=0)
    obs = env.reset()
    obs = obs.reshape(1, obs.shape[0])
    mlp(obs)
    load(net_vars=mlp.variables, directory=mdir, filename=mfile)
    return mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pcg", type=str, help="Name of the training dataset.")
    parser.add_argument("--mdir", default="", type=str, help="Directory of the model file that should be loaded.")
    parser.add_argument("--mfile", default="best_model.zip", type=str, help="Name of the model that should be loaded.")
    parser.add_argument("--gpu",type=bool,default=False,help="Should gpu be used")
    parser.add_argument("--render",type=bool,default=False,help="Should gpu be used")
    args = parser.parse_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # method \in \{drl, imilearn, gnn\}, opti_alg \in \{ppo, dqn\}
    method, opti_alg = get_method(mdir=args.mdir, mfile=args.mfile)
    print("Method: '{0}', Optimisation Algorithm: '{1}'".format(method, opti_alg))

    mpath = "{0}/{1}".format(args.mdir, args.mfile)

    env = SuperpointGrowingEnv(mapping=[4,2,1], ignore=[], dataset=args.dataset)
    
    predict_func = None

    if method == "drl":
        if opti_alg == "dqn":
            model = DQN.load(mpath, env=env)
        elif opti_alg == "ppo":
            model = PPO.load(mpath, env=env)
        else:
            raise Exception("Unknown optimisation algorithm: '{0}'".format(opti_alg))
        def pf(model, obs):
            return model.predict(obs, deterministic=True)
        predict_func = pf
    elif method == "imilearn":
        model = load_imilearn(dataset=args.dataset, env=env, mdir=args.mdir, mfile=args.mfile)
        def pf(model, obs):
            obs = obs.reshape(1, obs.shape[0])
            return model(obs), None
        predict_func = pf
    elif method == "gnn":
        model = load_gnn(dataset=args.dataset, env=env, mdir=args.mdir, mfile=args.mfile)
    else:
        raise Exception("Unknown method: '{0}'".format(method))

    obs = env.reset()
    done = False

    P = np.hstack((env.exp_dict["xyz"], env.exp_dict["rgb"]))

    stats = []
    stats.append("Number of points: ")
    stats.append(str(P.shape[0]))
    stats.append("\n")

    if method == "gnn":
        input_graphs, graph_dict = get_input_graphs(env=env, dataset=args.dataset)
        x = conv(input_graphs=input_graphs, mlps=model)
        fi = tf.gather(x.nodes, indices=x.senders)
        fj = tf.gather(x.nodes, indices=x.receivers)
        dot, _, _ = fast_dot(fi, fj)
        p = (dot + 1) / 2
        p = p.numpy()
        p_vec = partition_from_probs(graph_dict=graph_dict, sim_probs=p, k=0.425, P=P, sp_idxs=env.sp_idxs)
    else:
        reward = 0
        while not done:
            action, _states = predict_func(model=model, obs=obs)
            obs, rewards, done, info = env.step(action)
            reward += rewards
        p_vec = env.partition_vec
        stats.append("Episode Reward: ")
        stats.append(str(reward))
        stats.append("\n")
    

    uni_p_vec = np.unique(p_vec)
    stats.append("Size Agent Partition: ")
    stats.append(str(uni_p_vec.shape[0]))
    stats.append("\n")

    if args.render:
        print("Colour")
        render_point_cloud(P=P)
    if args.render:
        print("Agent Partition")
        render_point_cloud(P=P, partition_vec=p_vec)
    #
    p_vec = env.exp_dict["p_gt"]
    uni_p_vec = np.unique(p_vec)
    stats.append("Size Ground Truth Partition: ")
    stats.append(str(uni_p_vec.shape[0]))
    stats.append("\n")
    if args.render:
        print("Ground Truth Partition")
        render_point_cloud(P=P, partition_vec=p_vec)
    #
    p_vec = env.exp_dict["p_cp"]
    uni_p_vec = np.unique(p_vec)
    stats.append("Size CP Partition: ")
    stats.append(str(uni_p_vec.shape[0]))
    stats.append("\n")
    if args.render:
        print("Cut Pursuit Partition")
        render_point_cloud(P=P, partition_vec=p_vec)
    
    with open(args.mdir + "/stats.txt", "w") as f:
        f.writelines(stats)


if __name__ == "__main__":
    main()