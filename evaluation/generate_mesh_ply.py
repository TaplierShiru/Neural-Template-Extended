import numpy as np
import random
import torch
import importlib
import os
from models.network import AutoEncoder, ImageAutoEncoder
from data.data import ImNetImageSamples
from utils.debugger import MyDebugger
from torch.multiprocessing import Pool, Process, set_start_method

from .utils import sample_points_polygon_vox64_njit, write_ply_point_normal

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def extract_one_input(args):
    args_list, network_path, config_path, device_id = args

    torch.cuda.set_device(device_id)
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    network = ImageAutoEncoder(config=config).cuda(device_id)


    ### set autoencoder
    assert hasattr(config, 'auto_encoder_config_path') and os.path.exists(config.auto_encoder_config_path)
    auto_spec = importlib.util.spec_from_file_location('*', config.auto_encoder_config_path)
    auto_config = importlib.util.module_from_spec(auto_spec)
    auto_spec.loader.exec_module(auto_config)

    auto_encoder = AutoEncoder(config=auto_config).cuda(device_id)
    network.set_autoencoder(auto_encoder)

    network_state_dict = torch.load(network_path)
    for key, item in list(network_state_dict.items()):
        if key[:7] == 'module.':
            network_state_dict[key[7:]] = item
            del network_state_dict[key]
    network.load_state_dict(network_state_dict)
    network.eval()

    for args in args_list:
        image_inputs, store_file_path, resolution, max_batch, space_range, thershold, sample_normal_points, with_surface_point = args
        image_inputs = torch.from_numpy(image_inputs).float().cuda(device_id)

        ## new embedding
        embedding = network.image_encoder(image_inputs.unsqueeze(0))
        if os.path.exists(store_file_path):
            print(f"{store_file_path} exists!")
            continue
        else:
            (vertices, polygons, vertices_deformed, polygons_deformed, 
                embedding, vertices_convex, bsp_convex_list, 
                convex_predictions_sum, point_value_prediction) = network.auto_encoder.save_bsp_deform(
                inputs=None, file_path=store_file_path, resolution=resolution, max_batch=max_batch,
                space_range=space_range, thershold_1=thershold, embedding=embedding,
                return_voxel_and_values=True
            )

            if sample_normal_points:
                # sample surface points
                sampled_points_normals = sample_points_polygon_vox64_njit(vertices, polygons, convex_predictions_sum.copy(), 16384)
                point_coord = np.reshape(sampled_points_normals[:,:3]+sampled_points_normals[:,3:]*1e-4, [1,-1,3])
                point_coord = torch.from_numpy(point_coord).to(device_id)
                # point_coord = np.concatenate([point_coord, np.ones([1,point_coord.shape[1],1],np.float32) ],axis=2)
                
                _, sample_points_value, _, _ = network.auto_encoder.decoder(embedding, point_coord)
                sample_points_value = sample_points_value.detach().cpu().numpy()
                sampled_points_normals = sampled_points_normals[sample_points_value[0,:,0]>1e-4]
                
                np.random.shuffle(sampled_points_normals)
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
    model_type = f"AutoEncoder-{config.encoder_type}-{config.decoder_type}" if config.network_type == 'AutoEncoder' else f"AutoDecoder-{config.decoder_type}"
    samples = ImNetImageSamples(data_path=args.data_path)

    ### debugger
    debugger = MyDebugger(
        f'Mesh-visualization-Image-{os.path.basename(config.data_folder)}-{model_type}',
        is_save_print_to_file=False
    )
    #debugger.set_direcotry_name(some dir)

    ## loading index
    sample_interval = 1
    resolution = 64
    max_batch = 20000
    save_deformed = True
    thershold = 0.01
    with_surface_point = True
    with open(args.obj_txt_file, mode='r') as fr:
        indx2category_model_id_path_list = list(map(lambda x: x.strip(), fr.readlines()))

    device_count = torch.cuda.device_count()
    device_ratio = 1
    worker_nums = int(device_count * device_ratio)

    generate_args = [
        (
            samples[i][0][0], 
            os.path.join(args.save_folder, samples.obj_paths[i], 'obj.ply'), 
            resolution, max_batch, (-0.5, 0.5), thershold,
            args.sample_normal_points, with_surface_point
        ) 
        for i in range(len(samples)) if i % sample_interval == 0]
    random.shuffle(generate_args)

    splited_args = split(generate_args, worker_nums)
    final_args = [
        (splited_args[i], args.network_path, config_path, i % device_count) 
        for i in range(worker_nums)
    ]
    set_start_method('spawn')

    # for arg in args:
    #     extract_one_input(arg)

    if device_count > 1:
        pool = Pool(device_count)
        pool.map(extract_one_input, final_args)
    else:
        extract_one_input(final_args[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str,
                        help='Path to the config python file. Example: `/your/path/to/config.py`.')
    parser.add_argument('--data-path', type=str,
                        help='Path to h5 file which will be used for generation. Example: `/your/path/to/data.hdf5`.')
    parser.add_argument('--obj-txt-file', type=str,
                        help='Path to txt file with obj names to corresponding index in data path file. Example: `/your/path/to/obj_data.txt`.')
    parser.add_argument('--network-path', type=str,
                        help='Path to the pth file of trained model. ')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save prepared data.', default='./')
    parser.add_argument('--sample-normal-points', action='store_true',
                        help='If provided when normal points will be generated for ply. ')
    args = parser.parse_args()
    main(args)
