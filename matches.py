import random
import numpy as np
import time
from partition.partition import Partition
from partition.density_utils import densities_np


def compute_error(gt_p_vec, a_p_vec, gt_uni, gt_indices, gt_label_counts):
    assignments = {}
    
    n_errornous_points = 0
    intervals = []

    label_values, label_counts = np.unique(a_p_vec, return_counts = True)
    diff = np.absolute(gt_label_counts.shape[0] - label_counts.shape[0])

    # walls: orig_values[:6], objects: orig_values[6:]
    for i in range(gt_uni.shape[0]):
        idx = gt_indices[i] # start idx of a original label
        count = gt_label_counts[i] # how many original labels follow after the start idx
        obj_labels = a_p_vec[idx : idx + count] # np array of estimated in the range of a original label
        
        # which estimated labels exist in the range of a original label and how often they occur
        sorted_labels, obj_counts = np.unique(obj_labels, return_counts=True) 
        # ( int, int, np.array, np.array )
        intervals.append( (idx, count, sorted_labels, obj_counts, i) )

    #intervals = bubble_sort(intervals)
    n_unlabelled_points = 0
    u_wall = 0
    u_obj = 0
    e_wall = 0
    e_obj = 0
    for i in range(len(intervals)):
        obj_idx = intervals[i][0] # orig
        len_points = intervals[i][1] # orig
        sorted_labels = intervals[i][2] # estimated
        counts = intervals[i][3] # estimated
        orig_label = intervals[i][4] # orig
        #print(orig_label)

        # estimated labels over the range of a true segment
        obj_labels = a_p_vec[obj_idx:obj_idx+len_points]

        u_points = len(np.argwhere(obj_labels == -1))
        n_unlabelled_points += u_points
        if orig_label < 6: # wall
            u_wall += u_points
        else: # object
            u_obj += u_points

        # check if a label is already assigned
        is_already_assigned = True
        # check if a label is available
        no_label_available = False

        while is_already_assigned:
            # no more label available for assignment - this happens if there are less cluster than predicted
            if counts.size == 0:
                no_label_available = True
                break
            # get the index of the most frequent cluster
            j = np.argmax(counts)
            # get the most frequent cluster label
            if sorted_labels[j] == -1:
                is_already_assigned = True
            else:
                chosen_label = sorted_labels[j]
                # check if label is already assigned
                is_already_assigned = chosen_label in assignments.values()
            if(is_already_assigned):
                # if so delete the label and take the second most label and so forth
                sorted_labels = np.delete(sorted_labels, j)
                counts = np.delete(counts, j)
            else:
                break
        # if there are no more labels, consider all the points as misclustered
        # n_labels > n_orig_labels
        e_points=0
        if no_label_available:
            false_points = np.argwhere(obj_labels != -1)
            e_points=len(false_points)
            n_errornous_points += e_points
        else:
            # save the assignet label for next iterations
            assignments[i] = chosen_label
            # filter the objec  ts with the wrong labels
            false_points = np.argwhere(obj_labels != chosen_label)
            false_points = np.argwhere(obj_labels[false_points] != -1)
            # increment the n_errornous_points for every false point
            e_points=len(false_points)
            n_errornous_points += e_points
        if orig_label < 6: # wall
            e_wall += e_points 
        else: # object
            e_obj += e_points
    return n_errornous_points, n_unlabelled_points, diff, u_wall, u_obj, e_wall, e_obj


def main():
    n_points = int(1e7)
    print("Use {0} points".format(n_points))
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    high = random.randint(10, 100)
    gt_p_vec = np.random.randint(low=0, high=high, size=n_points)

    high = random.randint(10, 100)
    a_p_vec = np.random.randint(low=0, high=high, size=n_points)

    sortation = np.argsort(gt_p_vec)
    a_p_vec = a_p_vec[sortation]
    gt_p_vec = gt_p_vec[sortation]

    gt_uni, gt_indices, gt_label_counts = np.unique(gt_p_vec, return_index=True, return_counts=True)
    a_uni = np.unique(a_p_vec)
    print("{0} gt objs, {1} superpoints".format(gt_uni.shape[0], a_uni.shape[0]))


    t1 = time.time()
    n_errornous_points, n_unlabelled_points, diff, u_wall, u_obj, e_wall, e_obj = compute_error(
        gt_p_vec=gt_p_vec, a_p_vec=a_p_vec, gt_uni=gt_uni, gt_indices=gt_indices, gt_label_counts=gt_label_counts)
    t2 = time.time()
    #print(n_errornous_points, n_unlabelled_points, diff, u_wall, u_obj, e_wall, e_obj)
    print("Old reward: {0}s".format(t2-t1))

    partition_gt = Partition(partition=gt_p_vec)
    partition = Partition(partition=a_p_vec)
    density_func = densities_np
    t1 = time.time()
    classification, densities, differences = partition_gt.classify(
            partition_B=partition, density_function=density_func, return_differences=True)
    t2 = time.time()
    print("New reward: {0}s".format(t2-t1))
    #print(classification)


if __name__ == "__main__":
    main()