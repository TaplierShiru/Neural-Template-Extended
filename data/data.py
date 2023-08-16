import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import h5py
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ImNetImageSamples(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 auto_encoder=None,
                 max_batch=32,
                 sample_interval=1,
                 image_idx=None,
                 sample_voxel_size: int = 64,
                 sample_class: bool = False,
                 label_txt_path: str = None,
                 use_depth=False,
                 image_preferred_color_space=1):
        super(ImNetImageSamples, self).__init__()
        data_dict = h5py.File(data_path, 'r')
        self.data_voxels = data_dict['voxels'][:]
        self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.data_voxels.shape[1], self.data_voxels.shape[2],
                                                         self.data_voxels.shape[3]])
        self.data_values = data_dict['values_' + str(sample_voxel_size)][:].astype(np.float32)
        self.data_points = (data_dict['points_' + str(sample_voxel_size)][:].astype(np.float32) + 0.5) / 256 - 0.5
        
        if sample_class:
            # TODO: By default they stored as uint8, maybe by default store them as int32?
            self.data_classes = data_dict['classes'][:].astype(np.int32)
        else:
            self.data_classes = None

        ### get file
        if label_txt_path is None:
            label_txt_path = data_path[:-5] + '.txt'
            
        if os.path.isfile(label_txt_path):
            self.obj_paths = [line.rstrip('\n') for line in open(label_txt_path, mode='r').readlines()]
        else:
            # TODO: It could brake some inference notebooks
            self.obj_paths = None

        ### extract the latent vector
        if auto_encoder is not None:
            self.extract_latent_vector(data_voxels = self.data_voxels, auto_encoder = auto_encoder, max_batch = max_batch)

        ### interval
        self.sample_interval = sample_interval

        ### pixels
        self.crop_size = 128
        # 24 137 1
        if len(data_dict['pixels'].shape) == 5:
            _, self.view_num, self.view_size, _, image_color_size = data_dict['pixels'].shape
        else:
            _, self.view_num, self.view_size, _ = data_dict['pixels'].shape
            image_color_size = 1
        image_color_size_min = min(image_preferred_color_space, image_color_size)
        if image_preferred_color_space != image_color_size:
            print(
                f'Preferred color space is {image_preferred_color_space} but loaded are {image_color_size}. '
                f'Preferred or minimum value {image_color_size_min} will be used as more prioritized option.'
            )
        image_color_size = image_color_size_min

        self.crop_edge = self.view_size - self.crop_size
        if self.crop_edge < 0:
            pixels_to_add = self.crop_edge * (-1) # Invert
            new_view_size = self.view_size + pixels_to_add
            new_data_pixels = np.zeros(
                (data_dict['pixels'].shape[0], self.view_num, new_view_size, new_view_size, image_color_size), 
                dtype=data_dict['pixels'].dtype
            )
            new_data_pixels[:, :, :self.view_size, :self.view_size] = data_dict['pixels'][:, :, :, :, :image_color_size]

            self.data_pixels = np.transpose(
                new_data_pixels,
                # (N, n view, H, W, color) -> (N, n view, color, H, W)
                (0, 1, -1, 2, 3)
            )

            if use_depth:
                new_data_depths = np.zeros(
                    (data_dict['depths'].shape[0], self.view_num, new_view_size, new_view_size, 1), 
                    dtype=data_dict['depths'].dtype
                )
                new_data_depths[:, :, :self.view_size, :self.view_size] = data_dict['depths'][:]

                self.data_depths = np.transpose(
                    new_data_depths,
                    # (N, n view, H, W, color) -> (N, n view, color, H, W)
                    (0, 1, -1, 2, 3)
                ) 
        else:
            offset_x = int(self.crop_edge / 2)
            offset_y = int(self.crop_edge / 2)
            if len(data_dict['pixels'].shape) == 5:
                self.data_pixels = np.transpose(
                    data_dict['pixels'][:, :, offset_y:offset_y + self.crop_size, offset_x:offset_x + self.crop_size, :image_color_size],
                    # (N, n view, H, W, color) -> (N, n view, color, H, W)
                    (0, 1, -1, 2, 3)
                )
            else:
                self.data_pixels = np.transpose(
                    data_dict['pixels'][:, :, offset_y:offset_y + self.crop_size, offset_x:offset_x + self.crop_size, None],
                    # (N, n view, H, W, color) -> (N, n view, color, H, W)
                    (0, 1, -1, 2, 3)
                )
            
            if use_depth:
                self.data_depths = np.transpose(
                    data_dict['depths'][:, :, offset_y:offset_y + self.crop_size, offset_x:offset_x + self.crop_size],
                    # (N, n view, H, W, color) -> (N, n view, color, H, W)
                    (0, 1, -1, 2, 3)
                )

        self.image_idx = image_idx
        self.use_depth = use_depth
        self.image_color_size = image_color_size

    def __len__(self):
        return self.data_pixels.shape[0] // self.sample_interval

    def __getitem__(self, idx):

        idx = idx * self.sample_interval

        if self.image_idx is None:
            view_index = np.random.randint(0, self.view_num)
        else:
            view_index = self.image_idx

        image = self.data_pixels[idx, view_index].astype(np.float32) / 255.0
        if self.use_depth:
            # TODO: Is it okey to normilize them as image? Or should they be around center 0?
            depth = self.data_depths[idx, view_index].astype(np.float32) / 255.0
            image = np.concatenate([image, depth], axis=0)


        if hasattr(self, 'latent_vectors'):
            latent_vector_gt = self.latent_vectors[idx]
        else:
            latent_vector_gt = None

        if self.data_classes is not None:
            obj_class = self.data_classes[idx]
            processed_inputs = image, latent_vector_gt, obj_class
        else:
            processed_inputs = image, latent_vector_gt

        return processed_inputs, idx


    def extract_latent_vector(self, data_voxels,  auto_encoder, max_batch):
        num_batch = int(np.ceil(data_voxels.shape[0] / max_batch))

        results = []
        with tqdm(range(num_batch), unit='batch') as tlist:
            for i in tlist:
                batched_voxels = data_voxels[i*max_batch:(i+1)*max_batch].astype(np.float32)
                batched_voxels = torch.from_numpy(batched_voxels).float().to(device)

                latent_vectors = auto_encoder.encoder(batched_voxels).detach().cpu().numpy()
                results.append(latent_vectors)

        if len(results) == 1:
            self.latent_vectors = results
        else:
            self.latent_vectors = np.concatenate(tuple(results), axis = 0)



class ImNetSamples(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 sample_voxel_size: int,
                 interval=1,
                 sample_class: bool = False,
                 label_txt_path: str = None):
        super(ImNetSamples, self).__init__()
        self.sample_voxel_size = sample_voxel_size
        data_dict = h5py.File(data_path, 'r')
        self.data_points = (data_dict['points_' + str(self.sample_voxel_size)][:].astype(np.float32) + 0.5) / 256 - 0.5
        self.data_values = data_dict['values_' + str(self.sample_voxel_size)][:].astype(np.float32)
        self.data_voxels = data_dict['voxels'][:]
        self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.data_voxels.shape[1], self.data_voxels.shape[2],
                                                         self.data_voxels.shape[3]])

        if sample_class:
            # TODO: By default they stored as uint8, maybe by default store them as int32?
            self.data_classes = data_dict['classes'][:].astype(np.int32)
        else:
            self.data_classes = None

        ### get file
        if label_txt_path is None:
            label_txt_path = data_path[:-5] + '.txt'
            
        if os.path.isfile(label_txt_path):
            self.obj_paths = [line.rstrip('\n') for line in open(label_txt_path, mode='r').readlines()]
        else:
            # TODO: It could brake some inference notebooks
            self.obj_paths = None

        ## interval
        self.interval = interval


    def __len__(self):
        return self.data_points.shape[0] // self.interval

    def __getitem__(self, idx):

        idx = idx * self.interval

        if self.data_classes is not None:
            obj_class = self.data_classes[idx]
            processed_inputs = (
                self.data_voxels[idx].astype(np.float32), 
                self.data_points[idx].astype(np.float32),
                self.data_values[idx],
                obj_class
            )
        else:
            processed_inputs = (
                self.data_voxels[idx].astype(np.float32), 
                self.data_points[idx].astype(np.float32),
                self.data_values[idx]
            )


        return processed_inputs, idx
