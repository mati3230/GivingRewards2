import argparse
import os
import h5py
import numpy as np
import tensorflow as tf
import datetime
import math
from tqdm import tqdm
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
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
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

    X = []
    Y = []

    #n_total_edges = 0
    #n_unions = 0
    for i in tqdm(range(args.n_scenes), desc="Prepare Dataset"):
        exp_file = exp_files[i]
        exp_dict, sp_idxs = load_exp_data(fdir=exp_dir, fname=exp_file)
        senders = exp_dict["senders"]
        receivers = exp_dict["receivers"]
        n_edges = senders.shape[0]
        node_features = exp_dict["node_features"]

        p_vec_gt = exp_dict["p_gt"]
        p_vec_cp = exp_dict["p_cp"]
        p_gt = Partition(partition=p_vec_gt)
        p_cp = Partition(partition=p_vec_cp)
        densities = p_gt.compute_densities(p_cp, densities_np)
        alpha = p_cp.alpha(densities)
        unions_gt = get_unions(graph_dict=exp_dict, alpha=alpha)
        #n_total_edges += unions_gt.shape[0]
        #n_unions += np.sum(unions_gt)
        #print("{0}/{1} Union Decisions".format(np.sum(unions_gt), unions_gt.shape[0]))
        #"""
        for j in range(n_edges):
            sender = senders[j]
            receiver = receivers[j]
            fs = node_features[sender]
            fr = node_features[receiver]
            if args.dataset == "pcg":
                fs = fs[:-1]
                fr = fr[:-1]
            x = np.abs(fs - fr) / diff_f
            y = unions_gt[j]
            X.append(x)
            Y.append(y)
        #"""
    print("dataset has {0} samples".format(len(X)))
    #print("Ratio: {0}".format(n_unions / n_total_edges))
    #return
    net_arch = [8,8,4,1]
    if args.dataset == "pcg":
        net_arch = [7,7,4,1]
    activations = (len(net_arch)-1) * [tf.nn.relu]
    activations.append(tf.math.sigmoid)
    mlp = MLP(layer_dims=net_arch, activations=activations, name="mlp", dropout=0)
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    
    n_batches = math.floor(X.shape[0] / args.batch_size) 
    
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/" + args.dataset + "/" + current_time
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    
    train_step = 0

    bce = tf.keras.losses.BinaryCrossentropy()
    while(True):
        order = np.arange(X.shape[0])
        np.random.shuffle(order)
        X_ = X[order]
        Y_ = Y[order]
        for i in range(n_batches):
            with tf.GradientTape() as tape:
                start = i * args.batch_size
                stop = start + args.batch_size
                x = X_[start:stop]
                p = mlp(x)
                p = tf.reshape(p, (p.shape[0], ))
                y = Y_[start:stop]

                #pos_bce_loss = -y * tf.math.log(p)
                #neg_bce_loss = -(1-y) * tf.math.log(1-p)

                bce_loss = bce(y, p)

                """
                isnan_pos = tf.reduce_any(tf.math.is_nan(pos_bce_loss))
                isnan_neg = tf.reduce_any(tf.math.is_nan(neg_bce_loss))
                if isnan_neg or isnan_pos:
                    print("nan detected")
                    print("y:")
                    print(y)
                    print("p:")
                    print(p)
                    print("pos_bce_loss:")
                    print(pos_bce_loss)
                    print("neg_bce_loss:")
                    print(neg_bce_loss)
                    return
                    continue
                """

                #bce_loss = pos_bce_loss + neg_bce_loss
                #bce_loss = tf.reduce_mean(bce_loss)
                #pos_bce_loss = tf.reduce_mean(pos_bce_loss)
                #neg_bce_loss = tf.reduce_mean(neg_bce_loss)

                vars_ = tape.watched_variables()
                grads = tape.gradient(bce_loss, vars_)
                optimizer.apply_gradients(zip(grads, vars_))
            with train_summary_writer.as_default():
                tf.summary.scalar("train/loss", bce_loss, step=train_step)
                #tf.summary.scalar("train/pos_bce_loss", pos_bce_loss, step=train_step)
                #tf.summary.scalar("train/neg_bce_loss", neg_bce_loss, step=train_step)
            train_summary_writer.flush()
            train_step += 1
            if i == 0:
                save(net_vars=mlp.variables, directory="./models/imilearn_" + current_time, filename="mlp_" + str(train_step))


if __name__ == "__main__":
    main()