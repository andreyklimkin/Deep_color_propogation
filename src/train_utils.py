import numpy as np
import torch
import shutil

from collections import OrderedDict
from os import path, mkdir
from skimage import img_as_ubyte
from skimage.transform import resize


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
                                            
            