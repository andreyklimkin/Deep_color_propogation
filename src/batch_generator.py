import os
import numpy as np

from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


from skimage import img_as_float, io
from skimage.transform import resize

from tqdm import tqdm


class WraptorDataLoader(DataLoader):
    
    def __init__(self, *args, **kwargs):
        super(WraptorDataLoader, self).__init__(*args, **kwargs)
        self.iterator = iter(self)
        
    def get_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration as e:
            self.iterator = iter(self)
            batch = next(self.iterator)
            
        return batch


class BatchGenerator(Dataset):
    def __init__(self, img_dirs, is_train=True, need_crop=True, crop_size=256, need_resize=True, resize_shape=(256, 256, 3)):
        self.pairs_I0Ik_paths = []
        self.need_crop = need_crop
        self.crop_size = crop_size
        
        self.need_resize = need_resize
        self.resize_shape = resize_shape
        self.is_train = is_train
        
        
        for dir_path in tqdm(img_dirs):
            I0_path = "%s/%05d.jpg" % (dir_path, 0)
            for i in range(len(glob(dir_path + "/*"))):
                Ik_path = "%s/%05d.jpg" % (dir_path, i)
                self.pairs_I0Ik_paths.append((I0_path, Ik_path))
        
    
    def __len__(self):
        return len(self.pairs_I0Ik_paths)

    def __getitem__(self, idx):
        I0 = img_as_float(io.imread(self.pairs_I0Ik_paths[idx][0]))
        Ik = img_as_float(io.imread(self.pairs_I0Ik_paths[idx][1]))
        if np.sum((I0 - Ik) ** 2) < 1e-6:
            Ik_1 = I0.copy()
        else:
            Ik_1 = img_as_float(io.imread(self.pairs_I0Ik_paths[idx - 1][1]))
        
        if self.is_train:
            if self.need_crop:
                random_crop_h = np.random.randint(0, I0.shape[0] - self.crop_size)
                random_crop_w = np.random.randint(0, I0.shape[1] - self.crop_size)
                I0 = I0[random_crop_h: random_crop_h + self.crop_size, random_crop_w: random_crop_w + self.crop_size, :]
                Ik_1 = Ik_1[random_crop_h: random_crop_h + self.crop_size, random_crop_w: random_crop_w + self.crop_size, :]
                Ik = Ik[random_crop_h: random_crop_h + self.crop_size, random_crop_w: random_crop_w + self.crop_size, :]

            if self.need_resize:
                I0 = resize(I0, self.resize_shape)
                Ik_1 = resize(Ik_1, self.resize_shape)
                Ik = resize(Ik, self.resize_shape)
            
        return (I0.astype(np.double), Ik_1.astype(np.double), Ik.astype(np.double))
