import os
import sys
import time
import torch
from tqdm import tqdm, trange
import numpy as np
from PIL import Image
import json
import glob
from renderer import render
from argparse import ArgumentParser, Namespace
from gaussian_model import GaussianModel
from utils import *
from pytorch_msssim import ssim
import yaml
import cv2
def loadReplica(path):
    color_paths = sorted(glob.glob(os.path.join(path, "results/frame*.jpg")))
    #print(path, color_paths)
    tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp

def loadTUM(path):
    if os.path.exists(os.path.join(path, "rgb3")):
        color_paths = sorted(glob.glob(os.path.join(path, "rgb3/*.png")))
    else:
        color_paths = sorted(glob.glob(os.path.join(path, "rgb/*.png")))
    #print(path, color_paths)
    tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp

def loadKITTI(path):
    color_paths = sorted(glob.glob(os.path.join(path, "image_2/*.png")))
    #print(path, color_paths)
    tstamp = np.loadtxt(os.path.join(path, "times.txt"), delimiter=' ', dtype=np.unicode_).astype(np.float32)
    #tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp

def loadEuRoC(path):
    #print(path)
    color_paths = sorted(glob.glob(os.path.join(path, 'mav0/cam0/data/*.png')))    
    #print(color_paths) 
    tstamp = [np.float64(x.split('/')[-1][:-4])/1e9 for x in color_paths]
    return color_paths, tstamp

def associate_frames(tstamp_image, tstamp_pose, max_dt=0.08):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        j = np.argmin(np.abs(tstamp_pose - t))
        if (np.abs(tstamp_pose[j] - t) < max_dt):
            associations.append((i, j))
    return associations

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="evaluation script parameters")
    parser.add_argument("result_path", type=str, default = None)
    parser.add_argument("gt_path", type=str, default = None)
    parser.add_argument("--correct_scale", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    args = parser.parse_args()
    sh_degree = 3
    gaussians = GaussianModel(sh_degree)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    dirs = os.listdir(args.result_path)
    # load model
    width, height, fovx, fovy = 0,0,0,0
    ts = []
    Rs = []
    render_time = 0
    for file_name in dirs:
        #print(file_name)
        if "shutdown" in file_name:
            iter = file_name.split("_")[0]
            ply_path = os.path.join(args.result_path, file_name, "ply/point_cloud/iteration_{}".format(iter), "point_cloud.ply")
            gaussians.load_ply(ply_path)
            with open (os.path.join(args.result_path, file_name, "ply", "cameras.json"), "r") as fin:
                camera_paras = json.load(fin)
            #print(camera_paras[0])
            width, height, fx, fy = camera_paras[0]["width"], camera_paras[0]["height"], camera_paras[0]["fx"], camera_paras[0]["fy"]
            fovx = focal2fov(fx, width)
            fovy = focal2fov(fy, height)
            
            render_time = np.loadtxt(os.path.join(args.result_path, file_name, "render_time.txt"), delimiter=' ', dtype=np.unicode_)
            render_time = render_time[:, 1].astype(np.float32)
            """ 
            for camera_para in camera_paras:
                ts.append(camera_para["position"])
                Rs.append(camera_para["rotation"])
                world_view_transform2 = getWorld2View2(np.array(camera_para["rotation"]), np.array(camera_para["position"]))
                cam = MiniCam(width, height, fovx, fovy, world_view_transform2)
                render_image = render(cam, gaussians, background)["render"]
                predict_image_np = render_image.detach().cpu().numpy().transpose(1, 2, 0)
                predict_image_img = Image.fromarray(np.uint8(predict_image_np*255))
                predict_image_img.save(os.path.join(args.result_path, "image2", "{}.jpg".format(camera_para["img_name"])))
            """

    #load gt
    if "replica" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadReplica(args.gt_path)
    elif "kitti" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadKITTI(args.gt_path)   
    elif "euroc" in args.gt_path.lower():
        print(args.gt_path)
        gt_color_paths, gt_tstamp = loadEuRoC(args.gt_path)  
    else:
        gt_color_paths, gt_tstamp = loadTUM(args.gt_path)
    
    ## render and evaluation
    pose_path = os.path.join(args.result_path, "CameraTrajectory_TUM.txt")
    poses, tstamp = loadPose(pose_path)
    #print(gt_tstamp)
    associations = associate_frames(tstamp, gt_tstamp)
    distortion, K = None, None
    crop_edge = 0
    if os.path.isfile(os.path.join(args.gt_path, "camera.yaml")):
        with open(os.path.join(args.gt_path, "camera.yaml"), 'r') as fin:
            camera_model = yaml.safe_load(fin)
        K = np.eye(3)     
        K[0, 0] = camera_model["fx"]     
        K[1, 1] = camera_model["fy"]     
        K[0, 2] = camera_model["cx"]   
        K[1, 2] = camera_model["cy"]  
        crop_edge = camera_model["crop_edge"]
        distortion = np.array(camera_model['distortion']) if 'distortion' in camera_model else None
        re_u, re_v = cv2.initUndistortRectifyMap(K, distortion, None, K, (camera_model["W"], camera_model["H"]), m1type=cv2.CV_32FC1)

    #if not os.path.exists(os.path.join(args.result_path, "eval.txt")):
    os.makedirs(os.path.join(args.result_path, "image"), exist_ok=True)
    psnr_list, ssim_list, lpips_list, time_list= [], [], [], []
    #for c2w, stamp in tqdm(zip(poses, tstamp), desc="rendering"):
    for index  in trange(len(associations), desc="rendering {}".format(args.result_path.split("/")[-1])):
        (result_indx, gt_indx) = associations[index]
        w2c = np.linalg.inv(poses[result_indx])
        cam = MiniCam(width, height, fovx, fovy, w2c)
        t0 = time.time()
        render_image = render(cam, gaussians, background)["render"]
        t1 = time.time() - t0
        render_image = render_image.permute(1, 2, 0)
        gt_image = Image.open(gt_color_paths[gt_indx])
        gt_image = np.array(gt_image)
        if(len(gt_image.shape)<3):
            gt_image = np.broadcast_to(gt_image[..., None], (gt_image.shape[0],gt_image.shape[1],3))

        if distortion is not None:
            #print(K, distortion)
            gt_image_mask = np.ones_like(gt_image)
            #gt_image = cv2.undistort(np.array(gt_image), K, distortion)
            gt_image = cv2.remap(np.array(gt_image), re_u, re_v, interpolation=cv2.INTER_LINEAR, borderValue=[0,0,0])
            #os.makedirs(os.path.join(args.result_path, "image_gt"), exist_ok=True)
            #cv2.imwrite(os.path.join(args.result_path, "image_gt", gt_color_paths[gt_indx].split("/")[-1]), gt_image[:,:,[2,1,0]])
            #gt_image_mask = cv2.undistort(gt_image_mask, K, distortion)
            gt_image_mask = cv2.remap(gt_image_mask, re_u, re_v, interpolation=cv2.INTER_LINEAR, borderValue=[0,0,0])
            gt_image_mask = torch.from_numpy(np.array(gt_image_mask)).to("cuda")
            render_image = gt_image_mask * render_image
            

        gt_image_torch = torch.from_numpy(np.array(gt_image)).to("cuda") / 255.0
        val_loss = img2mse(render_image, gt_image_torch)
        val_psnr = mse2psnr(val_loss, render_image.device)
        #val_ssim = calculate_ssim(color_np, gt_color_np, channel_axis=-1, data_range=gt_color_np.max() - gt_color_np.min())
        val_ssim = ssim(render_image.permute([2, 1, 0]).unsqueeze(0), gt_image_torch.permute([2, 1, 0]).unsqueeze(0), data_range=1).item()
        val_lpips = LPIPS.calculate(render_image.type(torch.float32), gt_image_torch.type(torch.float32), render_image.device)
        
        render_image = torch.clamp(render_image, 0.0, 1.0)
        predict_image_np = render_image.detach().cpu().numpy()
        predict_image_img = Image.fromarray(np.uint8(predict_image_np*255))
        predict_image_img.save(os.path.join(args.result_path, "image", gt_color_paths[gt_indx].split("/")[-1]))
        
        psnr_list.append(val_psnr.item())
        ssim_list.append(val_ssim)
        lpips_list.append(val_lpips.item())
        time_list.append(t1)

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    time_list = np.array(time_list)
    np.savetxt(os.path.join(args.result_path, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(args.result_path, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(args.result_path, "lpips.txt"), lpips_list)

    with open(os.path.join(args.result_path, "TrackingTime.txt"), "r") as fin:
        tracking_time = fin.readlines()
    #print(tracking_time)
    tracking_time = np.array(tracking_time[:-3]).astype(np.float32)
    #tracking_time = np.loadtxt(os.path.join(args.result_path, "TrackingTime.txt"), delimiter=' ', dtype=np.unicode_)
    #tracking_time = tracking_time[:-3].astype(np.float32) # supposed last three lines is comment
    #np.savetxt(os.path.join(args.result_path, "renderng_time.txt"), time_list)
    #np.savetxt(os.path.join(args.result_path, "eval.txt"), [np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)])
    with open(os.path.join(args.result_path, "eval.txt"), "w") as fout:
        fout.write("psnr: {}\n".format(np.mean(psnr_list)))
        fout.write("ssim: {}\n".format(np.mean(ssim_list)))
        fout.write("lpips: {}\n".format(np.mean(lpips_list)))
        fout.write("tracking s: {}\n".format(np.mean(tracking_time)))
        fout.write("tracking FPS: {}\n".format(1/np.mean(tracking_time)))

        fout.write("rendering ms: {}\n".format(np.mean(render_time)))
        fout.write("rendering FPS: {}\n".format(1000/np.mean(render_time)))

    #gt = view.original_image[0:3, :, :]

    #### pose evaluation
    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    # load estimated poses
    traj_est = file_interface.read_tum_trajectory_file(pose_path)
    # load gt pose
    if "kitti" in args.gt_path.lower():
        def loadKITTIPose(gt_path):
            scene = gt_path.split("/")[-1]
            gt_file = gt_path.replace(scene, 'poses/{}.txt'.format(scene))
            pose_quat = []
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()
                    #print(line)
                    c2w = np.array(list(map(float, line))).reshape(3, 4)
                    #print(c2w)
                    quat = np.zeros(7)
                    quat[:3] = c2w[:3, 3]
                    quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                    pose_quat.append(quat)
            pose_quat = np.array(pose_quat)

            return pose_quat
        
        pose_quat = loadKITTIPose(args.gt_path)
        traj_ref = PoseTrajectory3D(positions_xyz=pose_quat[:,:3],         
                                        orientations_quat_wxyz=pose_quat[:,3:],         
                                        timestamps=np.array(gt_tstamp))

        #scene = args.gt_path.split("/")[-1]
        #gt_file = args.gt_path.replace(scene, 'poses/{}.txt'.format(scene))
        #print(gt_file)
        #traj_ref = file_interface.read_kitti_poses_file(gt_file)
    elif "replica" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, 'pose_TUM.txt')
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        #if not os.path.isfile(gt_file):
    elif "euroc" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, 'mav0/state_groundtruth_estimate0/data.csv')
        traj_ref = file_interface.read_euroc_csv_trajectory(gt_file)
    else:
        gt_file = os.path.join(args.gt_path, 'groundtruth.txt')   
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.1)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=args.correct_scale)
    result_rotation_part = main_ape.ape(traj_ref, traj_est, est_name='rot', pose_relation=PoseRelation.rotation_part, 
                                        align=True, correct_scale=args.correct_scale)

    out_path=os.path.join(args.result_path, "metrics_traj.txt")
    with open(out_path, 'w') as fp:
        fp.write(result.pretty_str())
        fp.write(result_rotation_part.pretty_str())
    print(result)

    if args.show_plot:
        from evo.tools import plot
        from evo.tools.plot import PlotMode
        import matplotlib.pyplot as plt
        import copy
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=True)
        fig = plt.figure()
        traj_by_label = {
            "estimate (not aligned)": traj_est,
            "estimate (aligned)": traj_est_aligned,
            "reference": traj_ref
        }
        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        plt.show()
