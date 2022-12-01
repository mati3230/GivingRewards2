# Giving Rewards 2

## Requirements

* python 3.8.13

```
pip install -r requirements.txt
```

The **libcp**, **libgeo** and the **libply_c** from the [3DPartitionAlgorithms](https://github.com/mati3230/3DPartitionAlgorithms) repository are required. 

## Quickstart

1) Create a dataset with the [Point Cloud Ray Tracer](https://github.com/mati3230/PointCloudRayTracer)
2) Execute the following command (pcg encodes a dataset created by the Point Cloud Ray Tracer): 
```
python create_feature_ds.py --dataset pcg 
```
3) Create a stats file containing min and max feature values:
```
python exp_stats.py --dataset pcg 
```
4) Train with deep reinforcement learning:
```
python train.py 
```
The tensorboard will be stored in the logs folder. The trained model can be found in the models folder.

You can also apply the imitation learning with the best matching object function:
```
python imilearn.py 
```

Use
```
python gnnlearn.py 
```
to apply the GNN-based method.

## Citation

Coming soon