import numpy as np
import torch
import shutil
import os
import imageio

from collections import OrderedDict

from color_utils import rgb2lab_torch, lab2rgb_torch
from glob import glob
from os import path, mkdir
from skimage import img_as_ubyte, io, img_as_float
from skimage.transform import resize
from skimage.measure import compare_psnr
from skimage.color import rgb2lab, lab2rgb
from tqdm import tqdm
from time import time




def draw_scalar_value(writer, scalar_folder, tag, scalar_value, iteration):
    writer.add_scalars("learning_stats/{}".format(scalar_folder),
                       {
                           tag: scalar_value,
                       },
                       iteration)

def draw_images(writer, gt_images, result_images, iteration, tag="train"):
    for i in range(len(gt_images)):
        if (tag == "train") or ("val" in tag):
            writer.add_image("{}_examples/example{}".format(tag, i),
                             np.concatenate((img_as_ubyte(gt_images[i].transpose(2, 0, 1)),
                                             img_as_ubyte(result_images[i]).transpose(2, 0, 1)), 2), iteration)

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)[..., None]
    return np.repeat(gray, 3, 2)        


def infer_batch(batch, refinement_net, local_transfer_net, global_transferer,
                use_only_local=False, zero_global=False,  verbose=False):
    I1_lab = rgb2lab_torch(batch[0].cuda(), use_gpu=True)
    Ik_lab = rgb2lab_torch(batch[2].cuda(), use_gpu=True)

    G1_array = np.array([rgb2gray(img.cpu().numpy()) for img in batch[0]])
    Gk_1_array = np.array([rgb2gray(img.cpu().numpy()) for img in batch[1]])
    Gk_array = np.array([rgb2gray(img.cpu().numpy()) for img in batch[2]])

    G1_tensor = torch.tensor(G1_array.transpose(0, 3, 1, 2), dtype=torch.double, requires_grad=False).cuda()
    Gk_1_tensor = torch.tensor(Gk_1_array.transpose(0, 3, 1, 2), dtype=torch.double, requires_grad=False).cuda()
    Gk_tensor = torch.tensor(Gk_array.transpose(0, 3, 1, 2), dtype=torch.double, requires_grad=False).cuda()

    t = time()
    local_batch_output = local_transfer_net.forward(Gk_1_tensor, Gk_tensor, batch[1].permute(0, 3, 1, 2).cuda())
    if verbose:
        print("Local inference time %.3f" % (time() - t))
    local_batch_output_lab = rgb2lab_torch(local_batch_output.permute(0, 2, 3, 1), use_gpu=True).permute(0, 3, 1, 2)
    
    t = time()
    global_batch_output = global_transferer.forward(G1_array, Gk_array, batch[0].cpu().numpy())
    if zero_global:
        global_batch_output *= 0
    if verbose:
        print("Global inference time %.3f" % (time() - t))
    global_batch_output = torch.tensor(global_batch_output.transpose(0, 3, 1, 2), dtype=torch.double, requires_grad=False)
    global_batch_output_lab = rgb2lab_torch(global_batch_output.permute(0, 2, 3, 1), use_gpu=False).permute(0, 3, 1, 2)

    stacked_input_refinement = torch.cat([Gk_tensor, local_batch_output_lab, global_batch_output_lab.cuda()], dim=1)
    if use_only_local:
        refinement_output_lab = local_batch_output_lab.permute(0, 2, 3, 1)
        gt_l = Ik_lab[..., 0][..., None]
        predicted_ab = refinement_output_lab[..., 1:]
        result_lab = torch.cat((gt_l, predicted_ab), dim=3)
        result_rgb = lab2rgb_torch(result_lab, use_gpu=True).cpu()
    else:
        refinement_output_lab = refinement_net(stacked_input_refinement).permute(0, 2, 3, 1)
        gt_l = Ik_lab[..., 0][..., None]
        predicted_ab = refinement_output_lab[..., 1:]
        result_lab = torch.cat((gt_l, predicted_ab), dim=3)
        result_rgb = lab2rgb_torch(result_lab, use_gpu=True).cpu()
    
    return (result_lab, result_rgb, 
            local_batch_output.permute(0, 2, 3, 1), global_batch_output.permute(0, 2, 3, 1))


def frame_to_tensor(frame):
    return torch.tensor(frame, dtype=torch.double, requires_grad=False).cuda()


def load_video_frames(path, resize_size=(128, 128, 3), resize_type="resize", left_crop_point=None, anti_aliasing=False):
    frames = []
    frames_num = len(glob(path + "/*"))
    for i in range(frames_num):
        frame = io.imread("%s/%05d.jpg" % (path, i))
        if resize_type == "resize":
            frames.append(resize(frame, resize_size, anti_aliasing=anti_aliasing))
        if resize_type == "crop":
            frames.append(img_as_float(frame[left_crop_point[0]: left_crop_point[0] + resize_size[0],
                                left_crop_point[1]: left_crop_point[1] + resize_size[1], :]))
    return frames

def load_all_test_videos(video_pathes, resize_size=(128, 128, 3), anti_aliasing=False):
    test_video_frames = []
    for video_path in video_pathes:
        test_video_frames.append(load_video_frames(video_path, resize_size, anti_aliasing=anti_aliasing))
    return test_video_frames

def inference_test_video(frames, refinement_net, local_transfer_net, global_transferer, use_only_local=False, zero_global=False):
    I0 = frames[0]
    I_prev = frames[0]
    Gk_1 = rgb2gray(I_prev)
    G0 = Gk_1.copy()
    output_rgb_frames = [I0]
    output_local = [I0]
    output_global = [I0]
    gray_frames = [G0]
    for cur_frame in tqdm(frames[1:]):
        gray_frames.append(rgb2gray(cur_frame))
        batch = (frame_to_tensor(I0[None, ...]), 
                 frame_to_tensor(I_prev[None, ...]),
                 frame_to_tensor(cur_frame[None, ...]))
        _, result_rgb, result_local, result_global = infer_batch(batch, refinement_net, local_transfer_net, 
                                                                 global_transferer, use_only_local, zero_global)
        output_rgb_frames.append(result_rgb[0].detach().cpu().numpy())
        output_local.append(result_local[0].detach().cpu().numpy())
        output_global.append(result_global[0].detach().cpu().numpy())
        I_prev = output_rgb_frames[-1]
        
    return gray_frames, output_rgb_frames, output_local, output_global

# def inference_train_video(frames, refinement_net, local_transfer_net, global_transferer, 
#                           use_only_local=False, zero_global=False):
#     I0 = frames[0]
#     I_prev = frames[0]
#     Gk_1 = rgb2gray(I_prev)
#     G0 = Gk_1.copy()
#     output_rgb_frames = [I0]
#     output_lab_frames = [rgb2lab(I0)]
#     output_local = [I0]
#     output_global = [I0]
#     gray_frames = [G0]
#     for cur_frame in tqdm(frames[1:]):
#         gray_frames.append(rgb2gray(cur_frame))
#         batch = (frame_to_tensor(I0[None, ...]), 
#                  frame_to_tensor(I_prev[None, ...]),
#                  frame_to_tensor(cur_frame[None, ...]))
#         result_lab, result_rgb, result_local, result_global = infer_batch(batch, refinement_net, local_transfer_net, 
#                                                                  global_transferer, use_only_local, zero_global)
#         output_rgb_frames.append(result_rgb[0])
#         output_local.append(result_local[0])
#         output_global.append(result_global[0])
#         I_prev = output_rgb_frames[-1]
        
#     return gray_frames, output_rgb_frames, output_local, output_global


def make_video_from_frames(output_path, frames):
    writer = imageio.get_writer(output_path, fps=20)
    for im in frames:
        writer.append_data((im * 255).astype(np.uint8))
    writer.close()

def inference_all_test_videos(video_pathes,
                              refinement_net, local_transfer_net, global_transferer,
                              save_dir_path, it, use_only_local=False, save_result=True, resize_size=(128, 128, 3)):
    videos_frames = load_all_test_videos(video_pathes, resize_size)
    result_videos_refinement = []
    result_videos_local = []
    result_videos_global = []
    
    if save_result:
        try:
            os.mkdir(os.path.join(save_dir_path,  str(it)))
        except:
            pass
    for i, video_frames in tqdm(enumerate(videos_frames)):
        gt_gray_frames, result_rbg_frames, result_local, result_global = inference_test_video(video_frames, 
                                                                                              refinement_net, 
                                                                                              local_transfer_net, 
                                                                                              global_transferer,
                                                                                              use_only_local)
        
        result_videos_refinement.append(result_rbg_frames)
        result_videos_local.append(result_local)
        result_videos_global.append(result_global)
        
        result_frames = [np.concatenate((gray, loc, gl, refine), axis=1) for gray, loc, gl, refine in zip(gt_gray_frames, 
                                                                                 result_local, 
                                                                                 result_global,
                                                                                 result_rbg_frames)]
        
        if save_result:
            make_video_from_frames(os.path.join(save_dir_path,  str(it), str(i) + ".mp4"), result_frames)
    
    return result_videos_refinement, result_videos_local, result_videos_global
        

def calculate_psnr_result(rgb_gt_videos, rgb_predicted_videos):
    videos_psnr = []
    for i in range(len(rgb_gt_videos)):
        predicted_frames = rgb_predicted_videos[i][1:]
        gt_frames = rgb_gt_videos[i][1:]
        frames_psnr = []
        for gt, p in zip(gt_frames, predicted_frames):
            #print(rgb2lab(gt).max())
            #gt_lab = 
            frames_psnr.append(compare_psnr(gt, p))
            #lab_gt = rgb2lab(gt)
            #lab_pred = rgb2lab(p)
            #frames_psnr.append(compare_psnr(rgb2lab(gt), rgb2lab(p), data_range=(100, 255, 255)))
        videos_psnr.append(np.mean(frames_psnr))
    return np.mean(videos_psnr)