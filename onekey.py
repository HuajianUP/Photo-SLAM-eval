import os


gt_dataset = {"replica": {"path": "/homes/huajian/Dataset/Replica/",
                           "scenes": ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2' ]}, 
            "tum": {"path": "/homes/huajian/Dataset/TUM",
                    "scenes": ['rgbd_dataset_freiburg3_long_office_household', 'rgbd_dataset_freiburg2_xyz', 'rgbd_dataset_freiburg1_desk']},
            "eth3D": {"path": "/homes/huajian/Dataset/ETH3D",
                        "scenes": ["desk_3", "mannequin_1", "mannequin_3", "planar_2", "planar_3", "table_7"]}
            }

# path the all results
result_main_folder = os.path.join("../result/3080ti/")
results = sorted(os.listdir(result_main_folder))

for result in results:
    print("processing", result)
    # support datasetName_cameratype_xx
    gt_dataset_name = result.split("_")[0].lower()
    gt_dataset_path = gt_dataset[gt_dataset_name]["path"]
    gt_dataset_scenes = gt_dataset[gt_dataset_name]["scenes"]
    for scene in gt_dataset_scenes:
        result_path = os.path.join(result_main_folder, result, scene)
        gt_path = os.path.join(gt_dataset_path, scene)
        if not os.path.exists(os.path.join(result_path, "eval.txt")):
            if "mono" in result.lower():
                os.system("python run.py {} {} --correct_scale".format(result_path, gt_path))
            else:
                os.system("python run.py {} {}".format(result_path, gt_path))


""" 
path = 
scenes =[dir for dir in os.listdir(path) if os.path.exists(os.path.join(path, dir, "traj.txt"))]
print(scenes)

for scene in scenes:
    print("processing", scene)
    gt_path = os.path.join(path, scene)

    result_path = os.path.join("result/3080ti/replica_mono", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {} --correct_scale".format(result_path, gt_path))

    result_path = os.path.join("result/3080ti/replica_mono_final", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {} --correct_scale".format(result_path, gt_path))

    result_path = os.path.join("result/3080ti/replica_rgbd", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {}".format(result_path, gt_path))


path = "/homes/huajian/Dataset/TUM" 
scenes =[dir for dir in os.listdir(path) if os.path.exists(os.path.join(path, dir, "rgb.txt"))] 
print(scenes) 
for scene in scenes:     
    print("processing", scene)     
    gt_path = os.path.join(path, scene)
    result_path = os.path.join("result/3080ti/tum_mono-3", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {} --correct_scale".format(result_path, gt_path))

    result_path = os.path.join("result/3080ti/tum_rgbd-2", scene)
    if not os.path.exists(os.path.join(result_path, "eval.txt")):
        os.system("python run.py {} {}".format(result_path, gt_path))
"""
""" 
path = "../../dataset/ETH3D" 
scenes =[dir for dir in os.listdir(path) 
if os.path.exists(os.path.join(path, dir, "rgb.txt"))] 
print(scenes) 
for scene in scenes:     
    print("processing", scene)     
    input_folder = os.path.join(path, scene)   

"""  