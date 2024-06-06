import os
import glob
import numpy as np
import argparse
"""
This is a scipt to calcuate quantitative results stored in the result_main_folder for the evluation dataset
Outputs are two files including "result_main_folder/log.txt" and "result_main_folder/log.csv"
"""

parser = argparse.ArgumentParser(description="evaluation script")
parser.add_argument("-d", "--dataset_center_path", type=str, required=True)
parser.add_argument("-r", "--result_main_folder", type=str, required=True)
args = parser.parse_args()

# dataset_center_path = "/homes/huajian/Dataset"
# result_main_folder = os.path.join("../result/4090/results/")

dataset_center_path = args.dataset_center_path
result_main_folder = args.result_main_folder

gt_dataset = {"replica": {"path": os.path.join(dataset_center_path, "/Replica/"),
                           "scenes": ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2' ]}, 
            "tum": {"path": os.path.join(dataset_center_path, "/TUM"),
                    "scenes": ['rgbd_dataset_freiburg3_long_office_household', 'rgbd_dataset_freiburg2_xyz', 'rgbd_dataset_freiburg1_desk']},
            "eth3d": {"path": os.path.join(dataset_center_path, "/ETH3D"),
                        "scenes": ["desk_3", "mannequin_1", "mannequin_3", "planar_2", "planar_3", "table_7"]},
            "kitti": {"path": os.path.join(dataset_center_path, "/KITTI"),
                        "scenes": ["00", "01","02","03","04","05","06","07","08","09","10"]},
            "euroc": {"path": os.path.join(dataset_center_path, "/EuRoC"),
                      "scenes": ["MH_01_easy", "MH_02_easy","V1_01_easy","V2_01_easy"]}
            }

# path the all results
results = [m for m in sorted(os.listdir(result_main_folder)) if os.path.isdir(os.path.join(result_main_folder, m))]

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


logs = [] 
camera_type = ['mono', 'rgbd', 'stereo']
#### get the result file ####
for gt_dataset_name in gt_dataset:
    # mono
    scenes = gt_dataset[gt_dataset_name]["scenes"]
    for camera in camera_type:
        results = sorted(glob.glob(os.path.join(result_main_folder, "{}_{}*".format(gt_dataset_name, camera))))
        for result in results:
            print(result)             
            logs.append(result+"\n")
            for scene in scenes:
                # T	R PSNR SSIM	LPIPS Tracking speed Rendering speed
                T, R, T_std = None, None, None
                if os.path.exists(os.path.join(result, scene, "metrics_traj.txt")):
                    with open(os.path.join(result, scene, "metrics_traj.txt")) as fin:
                        lines = fin.readlines()
                        ape_T = lines[7].split()
                        assert ape_T[0] == "rmse", result
                        T = ape_T[-1]
                        T_std = lines[9].split()[-1]
                        ape_R = lines[17].split()
                        assert ape_R[0] == "rmse", result
                        R = ape_R[-1]
                PSNR, SSIM, LPIPS, Tracking_fps, Rendering_fps =  None, None, None, None, None
                if os.path.exists(os.path.join(result, scene, "eval.txt")):    
                    with open(os.path.join(result, scene, "eval.txt")) as fin:
                        PSNR = fin.readline().split()[-1]
                        SSIM = fin.readline().split()[-1]
                        LPIPS = fin.readline().split()[-1]
                        Tracking_time = fin.readline().split()[-1]
                        Tracking_fps = fin.readline().split()[-1]
                        Rendering_time = fin.readline().split()[-1]
                        Rendering_fps = fin.readline().split()[-1]
                
                render_path = glob.glob(os.path.join(result, scene, "*shutdown", "render_time.txt"))
                if len(render_path)>0:
                   render_time = np.loadtxt(render_path[0], delimiter=' ', dtype=np.unicode_)
                   render_time = render_time[:, 1].astype(np.float32)
                   Rendering_fps = 1000/np.mean(render_time)

                result_str = "{} {} {} {} {} {} {} {} {}\n".format(scene, T, R, PSNR, SSIM, LPIPS, Tracking_fps, Rendering_fps, T_std)
                print(result_str)
                logs.append(result_str)
#rgbd
with open(os.path.join(result_main_folder,'log.txt'), 'w') as out_file:         
    for log in logs:   
        out_file.write(log)
import csv
with open(os.path.join(result_main_folder,'log.csv'), 'w') as out_file:         
    writer = csv.writer(out_file)    
    writer.writerow(('scene', 'T', 'R', 'PSNR', 'SSIM', 'LPIPS', 'Tracking FPS', 'Rendering FPS',"T_std"))     
    for log in logs:   
        writer.writerow(log.split())


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

