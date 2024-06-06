# Photo-SLAM Evaluation Toolkit
### [Homepage](https://huajianup.github.io/research/Photo-SLAM/) | [Paper](https://arxiv.org/abs/2311.16728)

**Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras** <br>
[Huajian Huang](https://huajianup.github.io)<sup>1</sup>, [Longwei Li](https://github.com/liquorleaf)<sup>2</sup>, Hui Cheng<sup>2</sup>, and [Sai-Kit Yeung](https://saikit.org/)<sup>1</sup> <br>
The Hong Kong University of Science and Technology<sup>1</sup>, Sun Yat-Sen University<sup>2</sup> <br>
In Proceedings of Computer Vision and Pattern Recognition Conference (CVPR), 2024<br>
![image](https://huajianup.github.io/thumbnails/Photo-SLAM_v2.gif "photo-slam")


## Prerequisites
To use this toolkit, you have to ensure your results on each dataset are stored in the correct format. For example, 
```
results
├── replica_mono_0
│   ├── office0
│   ├── ....
│   └── room2
├── replica_rgbd_0
│   ├── office0
│   ├── ....
│   └── room2
│
└── [replica/tum/euroc]_[mono/stereo/rgbd]_num  ....
    ├── scene_1
    ├── ....
    └── scene_n
```

### Install required python packages
```
# You need to install a compatible Pytorch as well.
# such as conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/HuajianUP/Photo-SLAM-eval.git
pip install evo numpy scipy scikit-image lpips pillow tqdm plyfile
```

### (Optional) Install submodel for rendering
```
# If you have installed original GS submodels, you can skip these steps.

pip install submodules/simple-knn/ 
pip install submodules/diff-gaussian-rasterization/
```

### Convert Replica GT camera pose files to suitable pose files to run the EVO package
```
python shapeReplicaGT.py --replica_dataset_path PATH_TO_REPLICA_DATASET
```

### Copy TUM camera.yaml to the corresponding dataset path
Since images on some sequences of TUM dataset contain distortion, we need to undistort the ground truth images before evaluation.
In addition, the file `camera.yaml` is used as an indicator in `run.py`.
```
cp TUM/fr1/camera.yaml PATH_TO_TUM_DATASET/rgbd_dataset_freiburg1_desk
cp TUM/fr2/camera.yaml PATH_TO_TUM_DATASET/rgbd_dataset_freiburg2_xyz
```

## Evaluation
To get all the metrics, you can run 
```
python onekey.py --dataset_center_path PATH_TO_ALL_DATASET --result_main_folder RESULTS_PATH
```
Finally, you are supposed to get two files including `RESULTS_PATH/log.txt` and `RESULTS_PATH/log.csv`.
