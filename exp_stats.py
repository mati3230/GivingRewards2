import os
import numpy as np
import argparse
from exp_utils import load_exp_data, save_exp_data
import open3d as o3d


def coordinate_system():
    line_set = o3d.geometry.LineSet()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def render_pc(pcd, animate=False, x_speed=2.5, y_speed=0.0, width=1920, left=0):
    if animate:
        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(x_speed, y_speed)
            return False
        o3d.visualization.draw_geometries_with_animation_callback(
            [pcd, coordinate_system()], rotate_view, width=width, left=left)
    else:
        o3d.visualization.draw_geometries([pcd, coordinate_system()], width=width, left=left)


def render_point_cloud(
        P, partition_vec=None, animate=False, x_speed=2.5, y_speed=0.0, width=1920, left=0, colors_dir=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    if partition_vec is not None:
        col_mat = np.zeros((P.shape[0], 3))
        colors_path = "colors.npz"
        if colors_dir is not None:
            colors_path = colors_dir + "/" + colors_path
        data = np.load(colors_path)
        colors = data["colors"]
        superpoints = np.unique(partition_vec)
        n_superpoints = superpoints.shape[0]
        for i in range(n_superpoints):
            superpoint_value = superpoints[i]
            idx = np.where(partition_vec == superpoint_value)[0]
            color = colors[i, :] / 255
            col_mat[idx, :] = color
        pcd.colors = o3d.utility.Vector3dVector(col_mat)
    else:
        try:
            # print(P[:5, 3:6] / 255.0)
            pcd.colors = o3d.utility.Vector3dVector(P[:, 3:6])
        except Exception as e:
            print(e)
    render_pc(pcd=pcd, animate=animate, x_speed=x_speed, y_speed=y_speed, width=width, left=left)
    return pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scannet", type=str, help="Name of the dataset (scannet or pcg).")
    parser.add_argument("--render", default=False, type=bool, help="Activates the rendering of the point cloud.")
    args = parser.parse_args()
    if args.dataset != "scannet" and args.dataset != "pcg":
        raise Exception("Only 'scannet' and 'pcg' can be used as datasets")
    fdir = "./exp_data_" + args.dataset
    exp_files = os.listdir(fdir)
    exp_dict, sp_idxs = load_exp_data(fdir=fdir, fname=exp_files[0])
    node_features = exp_dict["node_features"]
    g_min_features = np.zeros((node_features.shape[1], )) + np.inf
    g_max_features = np.zeros((node_features.shape[1], )) - np.inf

    for fname in exp_files:
        exp_dict, sp_idxs = load_exp_data(fdir=fdir, fname=fname)
        node_features = exp_dict["node_features"]

        min_features = np.min(node_features, axis=0)
        max_features = np.max(node_features, axis=0)

        tmp_min_features = np.vstack((g_min_features[None, :], min_features[None, :]))
        tmp_max_features = np.vstack((g_max_features[None, :], max_features[None, :]))

        g_min_features = np.min(tmp_min_features, axis=0)
        g_max_features = np.max(tmp_max_features, axis=0)
    print("min:", g_min_features)
    print("max:", g_max_features)
    print("overall min:", np.min(g_min_features))
    print("overall max:", np.max(g_max_features))
    save_exp_data(fdir="./", fname="exp_stats_" + args.dataset, exp_dict={
            "min": g_min_features,
            "max": g_max_features
        })
    if args.render:
        P = np.hstack((exp_dict["xyz"], exp_dict["rgb"]))
        print("Point Cloud")
        render_point_cloud(P=P)
        p_vec = exp_dict["p_gt"]
        print("Ground Truth Partition")
        render_point_cloud(P=P, partition_vec=p_vec)
        p_vec = exp_dict["p_cp"]
        print("Cut Pursuit Partition")
        render_point_cloud(P=P, partition_vec=p_vec)

if __name__ == "__main__":
    main()