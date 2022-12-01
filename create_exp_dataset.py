import argparse
import os
from tqdm import tqdm
import numpy as np

#from scannet_utils import get_scenes, get_ground_truth
import scannet_utils
import pcg_utils
from ai_utils import graph
from exp_utils import mkdir, save_exp_data
from io_utils import file_exists


def main():
    parser = argparse.ArgumentParser()
    # cp params
    parser.add_argument("--reg_strength", default=0.1, type=float, help="Regularization strength for the minimal partition. Increase lambda for a coarser partition. ")
    parser.add_argument("--k_nn_geof", default=45, type=int, help="Number of neighbors for the geometric features.")
    parser.add_argument("--k_nn_adj", default=10, type=int, help="Adjacency structure for the minimal partition.")
    parser.add_argument("--lambda_edge_weight", default=1., type=float, help="Parameter determine the edge weight for minimal part.")
    parser.add_argument("--d_se_max", default=0, type=float, help="Max length of super edges.")
    parser.add_argument("--max_sp_size", default=-1, type=int, help="Maximum size of a superpoint.")
    parser.add_argument("--dataset", default="scannet", type=str, help="Name of the dataset (scannet or pcg).")
    #
    #parser.add_argument("--pkg_size", default=5, type=int, help="Number of packages to save a csv")
    parser.add_argument("--h5_dir", default="./exp_data", type=str, help="Directory where we save the h5 files.")
    args = parser.parse_args()
    if args.dataset != "scannet" and args.dataset != "pcg":
        raise Exception("Only 'scannet' and 'pcg' can be used as datasets")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    h5_dir = args.h5_dir + "_" + args.dataset
    mkdir(h5_dir)

    if args.dataset == "scannet":
        scenes, _, scannet_dir = scannet_utils.get_scenes(blacklist=[])
    else:
        scenes, pcg_dir = pcg_utils.get_scenes(blacklist=[])
    n_scenes = len(scenes)
    
    desc = "Exp Data"
    verbose = False
    for i in tqdm(range(n_scenes), desc=desc, disable=verbose):
        scene_name = scenes[i]
        if file_exists(h5_dir + "/" + scene_name + ".h5"):
            continue
        if args.dataset == "scannet":
            mesh, p_vec_gt, file_gt = scannet_utils.get_ground_truth(scannet_dir=scannet_dir, scene=scene_name)
            xyz = np.asarray(mesh.vertices)
            rgb = np.asarray(mesh.vertex_colors)
        else:
            xyz, rgb, p_vec_gt = pcg_utils.get_ground_truth(pcg_dir=pcg_dir, scene=scene_name)
        sortation = np.argsort(p_vec_gt)
        sortation = sortation.astype(np.uint32)
        xyz = xyz[sortation]
        rgb = rgb[sortation]
        P = np.hstack((xyz, rgb))
        p_vec_gt = p_vec_gt[sortation]

        graph_dict, sp_idxs, part_cp = graph(
            cloud=P,
            k_nn_adj=args.k_nn_adj,
            k_nn_geof=args.k_nn_geof,
            lambda_edge_weight=args.lambda_edge_weight,
            reg_strength=args.reg_strength,
            d_se_max=args.d_se_max,
            max_sp_size=args.max_sp_size,
            verbose=verbose,
            return_p_vec=True)

        exp_dict = {
            "xyz": xyz,
            "rgb": rgb,
            "sortation": sortation,
            "node_features": graph_dict["nodes"],
            "senders": graph_dict["senders"],
            "receivers": graph_dict["receivers"],
            "p_gt": p_vec_gt,
            "p_cp": part_cp
        }
        for j in range(len(sp_idxs)):
            sp = sp_idxs[j]
            exp_dict[str(j)] = np.array(sp)
        save_exp_data(fdir=h5_dir, fname=scene_name, exp_dict=exp_dict)


if __name__ == "__main__":
    main()