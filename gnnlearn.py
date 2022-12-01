import argparse
import os
import h5py
import numpy as np
import tensorflow as tf
import datetime
import math
from graph_nets import utils_tf
import graph_nets as gn
from exp_utils import load_exp_data, get_unions
from partition.partition import Partition
from partition.density_utils import densities_np
from mlp import MLP
from network_utils import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pcg", type=str, help="Name of the training dataset.")
    parser.add_argument("--h5_dir", default="./exp_data", type=str, help="Directory where we save the h5 files.")
    parser.add_argument("--n_scenes", default=1, type=int, help="Number of scenes that should be used.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument("--gpu",type=bool,default=False,help="Should gpu be used")
    args = parser.parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    exp_dir = "./exp_data_" + args.dataset
    exp_files = os.listdir(exp_dir)

    hf = h5py.File("./exp_stats_" + args.dataset + ".h5", "r")
    if args.dataset == "pcg":
        min_f = np.array(hf["min"])[:-1]
        max_f = np.array(hf["max"])[:-1]
    else:
        min_f = np.array(hf["min"])
        max_f = np.array(hf["max"])
    diff_f = max_f - min_f

    dataset = []
    for i in range(args.n_scenes):
        exp_file = exp_files[i]
        exp_dict, sp_idxs = load_exp_data(fdir=exp_dir, fname=exp_file)
        senders = exp_dict["senders"]
        receivers = exp_dict["receivers"]
        n_edges = senders.shape[0]
        node_features = exp_dict["node_features"]
        if args.dataset == "pcg":
            node_features = node_features[:, :-1]
        input_graphs = {"nodes": node_features, "senders": senders, "receivers": receivers, "edges": None, "globals": None}
        input_graphs = utils_tf.data_dicts_to_graphs_tuple([input_graphs])

        p_vec_gt = exp_dict["p_gt"]
        p_vec_cp = exp_dict["p_cp"]
        p_gt = Partition(partition=p_vec_gt)
        p_cp = Partition(partition=p_vec_cp)
        densities = p_gt.compute_densities(p_cp, densities_np)
        alpha = p_cp.alpha(densities)
        unions_gt = get_unions(graph_dict=exp_dict, alpha=alpha)
        unions_gt = np.array(unions_gt, dtype=np.float32)

        dataset.append((input_graphs, unions_gt))

    print("dataset has {0} samples".format(len(dataset)))
    net_arch = [8,4]
    if args.dataset == "pcg":
        net_arch = [7,4]
    activations = len(net_arch) * [tf.nn.relu]

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

    mlps = []
    for i in range(len(net_arch)):
        layer_size = net_arch[i]
        activation = activations[i]
        mlp = MLP(layer_dims=[layer_size], activations=[activation], name="mlp"+str(i), dropout=0)
        mlps.append(mlp)

    
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/" + args.dataset + "/gnn/" + current_time
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    
    train_step = 0

    bce = tf.keras.losses.BinaryCrossentropy()
    while(True):
        for i in range(len(dataset)):
            with tf.GradientTape() as tape:
                (x, y) = dataset[i]
                for j in range(len(mlps)):
                    x = graph_conv(model_fn=mlps[j], input_graphs=x, training=True)    
                fi = tf.gather(x.nodes, indices=x.senders)
                fj = tf.gather(x.nodes, indices=x.receivers)
                dot, _, _ = fast_dot(fi, fj)
                p = (dot + 1) / 2
                bce_loss = bce(y, p)

                vars_ = tape.watched_variables()
                grads = tape.gradient(bce_loss, vars_)
                optimizer.apply_gradients(zip(grads, vars_))
            with train_summary_writer.as_default():
                tf.summary.scalar("train/loss", bce_loss, step=train_step)
            train_summary_writer.flush()
            train_step += 1
            if i % 100 ==  0:
                for j in range(len(mlps)):
                    save(net_vars=mlps[j].variables, directory="./models/gnn_" + current_time, filename="gnn_" + str(j) + "_" + str(train_step))


if __name__ == "__main__":
    main()