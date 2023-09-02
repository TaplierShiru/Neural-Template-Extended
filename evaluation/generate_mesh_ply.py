import numpy as np
import random
import torch
import importlib
import os
import argparse

try:
    from models.network import AutoEncoder
except ModuleNotFoundError:
    # Append base path with all needed code
    import pathlib
    import sys
    base_path, _ = os.path.split(pathlib.Path(__file__).parent.resolve())
    sys.path.append(base_path)
    # Try again
    from models.network import AutoEncoder

from data.data import ImNetImageSamples, ImNetSamples
from torch.multiprocessing import Pool, Process, set_start_method

from eval_utils import sample_points_polygon_vox64_njit
from utils.other_utils import write_ply_point_normal

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def extract_one_input(args):
    # TODO: Input type could be removed, but may be its better to keep it?
    args_list, network_path, config_path, device_id, sample_normal_points, input_type, aggregate_embedding = args
    print(f'Start generation with number of args {len(args_list)} and device id {device_id}')

    torch.cuda.set_device(device_id)
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    network_state_dict = torch.load(network_path)
    network_state_dict, is_old_style_weights = AutoEncoder.process_state_dict(network_state_dict, type = 1)
    if is_old_style_weights:
        config = AutoEncoder.fix_old_weights_config(config)
    
    network = AutoEncoder(config=config).cuda(device_id)
    network.load_state_dict(network_state_dict)
    network.eval()

    with torch.no_grad():
        for i, args in enumerate(args_list):
            if i % int(len(args_list) * 0.1) == 0:
                print(f'Device id {device_id} ; done is {round(i/len(args_list)*100, 2)}') 
            input_data, store_file_folder_path, resolution, max_batch, space_range, thershold, with_surface_point = args
            os.makedirs(store_file_folder_path, exist_ok=True)
            store_file_path = os.path.join(store_file_folder_path, 'obj.ply')
            if os.path.exists(store_file_path[:-4] + '_deformed.ply'):
                print(f"{store_file_path} exists!")
                continue
            
            if not isinstance(input_data, list) or (isinstance(input_data, list) and len(input_data) == 1):
                input_data = torch.from_numpy(input_data[0] if isinstance(input_data, list) else input_data).float().cuda(device_id)
                result = network.save_bsp_deform(
                    inputs=input_data, file_path=store_file_path, resolution=resolution, max_batch=max_batch,
                    space_range=space_range, thershold_1=thershold, embedding=None,
                    return_voxel_and_values=True
                )
            elif len(input_data) > 1 and aggregate_embedding:
                embeddings_list = []
                for input_data_single in input_data:
                    input_data_single = torch.from_numpy(input_data_single).float().cuda(device_id)
                    input_data_single = input_data_single.unsqueeze(0)
                    embeddings_list.append(
                        network(input_data_single)
                    )
                embedding = torch.stack(embeddings_list).mean(dim=0) # TODO: mean or median?
                result = network.save_bsp_deform(
                    inputs=None, file_path=store_file_path, resolution=resolution, max_batch=max_batch,
                    space_range=space_range, thershold_1=thershold, embedding=embedding,
                    return_voxel_and_values=True
                )
            else:
                raise Exception('Unknown mode for network.')

            if hasattr(config, 'sample_class') and config.sample_class:
                (vertices, polygons, vertices_deformed, polygons_deformed, 
                    embedding, vertices_convex, bsp_convex_list, 
                    predicted_class, convex_predictions_sum, point_value_prediction) = result
                np.save(os.path.join(store_file_folder_path, 'predicted_class_logits.npy'), predicted_class)
            else:
                (vertices, polygons, vertices_deformed, polygons_deformed, 
                    embedding, vertices_convex, bsp_convex_list, 
                    convex_predictions_sum, point_value_prediction) = result

            if sample_normal_points:
                # Invert convex voxel cube
                inverted_convex_predictions = np.logical_not(convex_predictions_sum > 1e-4).astype(np.float32)
                # Sample surface points
                sampled_points_normals = sample_points_polygon_vox64_njit(
                    vertices_deformed, polygons_deformed, 
                    inverted_convex_predictions, 16384, # 4096 * 4
                )
                point_coord = np.reshape(sampled_points_normals[:,:3]+sampled_points_normals[:,3:]*1e-4, [1,-1,3])
                point_coord = torch.from_numpy(point_coord).to(device_id)

                _, sample_points_value, _, _ = network.decoder(embedding, point_coord)
                    
                sample_points_value = sample_points_value.detach().cpu().numpy()
                # Take normals only for points which are inside object
                sampled_points_normals = sampled_points_normals[sample_points_value[0,:,0]>1e-4]

                np.random.shuffle(sampled_points_normals)
                # TODO: Is constraints with 4096 points needed here?
                write_ply_point_normal(store_file_path[:-4] + '_normals.ply', sampled_points_normals[:4096])


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def main(args):
    ## import config here
    spec = importlib.util.spec_from_file_location('*', args.config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    ## dataload
    ### create dataset
    if args.input_type == 'image':
        samples = ImNetImageSamples(
            data_path=args.data_path, 
            label_txt_path=args.obj_txt_file,
            image_idx=23, # Last image, stick to BSP-Net calc
            use_depth=hasattr(config, 'use_depth') and config.use_depth,
            image_preferred_color_space=config.image_preferred_color_space if hasattr(config, 'image_preferred_color_space') else 1
        )
    elif args.input_type == 'voxels':
        # TODO: In some cases sample_voxel_size could have different size, add to args - not now cause not needed and dont used
        samples = ImNetSamples(data_path=args.data_path, sample_voxel_size=64, label_txt_path=args.obj_txt_file)
    else:
        raise Exception(f'Unknown input type {args.input_type}. ')

    ## loading index
    sample_interval = 1
    resolution = 64
    max_batch = 20000 if args.input_type == 'image' else 100000
    thershold = 0.01
    with_surface_point = True # TODO: Is it needed here?

    device_count = torch.cuda.device_count() if args.max_number_gpu < 0 else args.max_number_gpu
    worker_nums = int(device_count * args.device_ratio)

    print(f'Start generation with device_count={device_count} and worker_nums={worker_nums}')
    if args.sample_normal_points:
        print('Also generate normal points for each final obj')
    
    if args.aggregate_embedding:
        if args.num_input_data_aggregation == None:
            print('Aggregation is True, but number of data is missing. Skip aggregation.')
        else:
            print(f'Aggregate input embedding by {args.num_input_data_aggregation if args.num_input_data_aggregation != -1 else samples.view_num} samples')
        
        if args.input_type != 'image':
            raise Exception('Aggregation supported only by images!')

    def get_input_data(samples, i, num_input_data_aggregation, aggregate_embedding, view_use_indx_list):
        if aggregate_embedding:
            if num_input_data_aggregation is not None and num_input_data_aggregation == -1 and view_use_indx_list is None:
                indx_view_iterator = range(samples.view_num)
            elif view_use_indx_list is not None and len(view_use_indx_list) > 0:
                indx_view_iterator = view_use_indx_list
            else:
                return [samples[i][0][0] for _ in range(num_input_data_aggregation)]
            gathered_data_list = []
            for indx_view in indx_view_iterator:
                samples.image_idx = indx_view
                gathered_data_list.append(
                    samples[i][0][0]
                )
            # Can be set to None, we dont care here about value 23 
            samples.image_idx = None
            return gathered_data_list
        
        return samples[i][0][0]

    generate_args = [
        (
            get_input_data(samples, i, args.num_input_data_aggregation, args.aggregate_embedding, args.view_use_indx_args), 
            os.path.join(args.save_folder, samples.obj_paths[i]), 
            resolution, max_batch, (-0.5, 0.5), 
            thershold, with_surface_point
        ) 
        for i in range(len(samples)) if i % sample_interval == 0]
    random.shuffle(generate_args)
    print(f'Number of arguments equal to {len(generate_args)}')

    splited_args = split(generate_args, worker_nums)
    final_args = [
        (
            splited_args[i], args.network_path, args.config_path, 
            i % device_count, args.sample_normal_points,
            args.input_type, args.aggregate_embedding
        ) 
        for i in range(worker_nums)
    ]
    set_start_method('spawn')

    if args.test:
        extract_one_input(final_args[0][0])
        return

    if worker_nums > 1:
        pool = Pool(worker_nums)
        pool.map(extract_one_input, final_args)
    else:
        extract_one_input(final_args[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str,
                        help='Path to the config python file. Example: `/your/path/to/config.py`.')
    parser.add_argument('--data-path', type=str,
                        help='Path to h5 file which will be used for generation. Example: `/your/path/to/data.hdf5`.')
    parser.add_argument('--input-type', choices=['image', 'voxels'], default='image',
                        help='Type of input data for loaded config and network. Default is image. ')
    parser.add_argument('--obj-txt-file', type=str,
                        help='Path to txt file with obj names to corresponding index in data path file. Example: `/your/path/to/obj_data.txt`.')
    parser.add_argument('--network-path', type=str,
                        help='Path to the pth file of trained model. ')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save prepared data.', default='./')
    parser.add_argument('--sample-normal-points', action='store_true',
                        help='If provided when normal points will be generated for ply. ')
    parser.add_argument('--aggregate-embedding', action='store_true',
                        help='If provided when embedding will be aggregated using several number of inputs equal to number in `num-input-data-aggregation`. ')
    parser.add_argument('--num-input-data-aggregation', type=int, default=None,
                        help='Number of input data for aggregation of embedding vectors. '
                        'By default equal to None, i.e. will be not used, if equal to -1 when will be used all views.')
    parser.add_argument('--view-use-indx-args', nargs='*', default=None,
                        help='If input data is `image` and `aggregate-embedding` is used, '
                        'when listed indexes in `view-use-indx-args` will be used for aggregation and object generation.')
    parser.add_argument('--max-number-gpu', type=int, default=-1,
                        help='Max number of GPUs to use. By default equal to -1, i.e. will be used all GPUs.')
    parser.add_argument('--device-ratio', type=int, default=1,
                        help='Number of processes per single gpu. By default equal to 1.')
    parser.add_argument('--test', action='store_true',
                        help='If equal to True, when test launch will be started.')
    args = parser.parse_args()
    main(args)
