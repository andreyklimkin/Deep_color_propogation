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
    
# def draw_photos(writer, batch, iteration, tag="train"):
#     for i in range(len(batch[0])):
#         if (tag == "train") or ("val" in tag):
#             original_imgs, corrupted_imgs, reconstructed_imgs = batch
#             writer.add_image("{}_examples/example{}".format(tag, i),
#                              np.concatenate((img_as_ubyte(original_imgs[i]), img_as_ubyte(corrupted_imgs[i]),
#                                              img_as_ubyte(reconstructed_imgs[i])), 1).transpose(2, 0, 1), iteration)
#         elif tag == "test":
#             original_imgs, reconstructed_imgs = batch
#             writer.add_image("{}_examples/example{}".format(tag, i),
#                              np.concatenate((img_as_ubyte(original_imgs[i]),
#                                              img_as_ubyte(reconstructed_imgs[i])), 1).transpose(2, 0, 1), iteration)