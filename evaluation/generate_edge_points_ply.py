import traceback
import numpy as np
import random
import argparse
import importlib
import os

from tqdm import tqdm
from multiprocessing import Pool

try:
    from utils.ply_utils import read_ply_point_normal
except ModuleNotFoundError:
    # Append base path with all needed code
    import pathlib
    import sys
    base_path, _ = os.path.split(pathlib.Path(__file__).parent.resolve())
    sys.path.append(base_path)
    # Try again
    from utils.ply_utils import read_ply_point_normal

from utils.other_utils import write_ply_point_normal


def calculate_edges(in_points, in_normals):
    num_points = in_points.shape[0]

    points_tiled_mat1 = np.tile(in_points.reshape(num_points, 1, 3), [1,num_points,1])
    points_tiled_mat2 = np.tile(in_points.reshape(1, num_points, 3), [num_points,1,1])
    dist_points = np.square(points_tiled_mat1 - points_tiled_mat2).sum(axis=2)
    close_index = (dist_points<1e-2).astype(np.int8)

    normals_tiled_mat1 = np.tile(in_normals.reshape(num_points, 1, 3), [1,num_points,1])
    normals_tiled_mat2 = np.tile(in_normals.reshape(1, num_points, 3), [num_points,1,1])
    dist_normals = np.square(normals_tiled_mat1 * normals_tiled_mat2).sum(axis=2)
    all_edge_index = (np.abs(dist_normals) < 0.1).astype(np.int8)

    edge_index = (close_index * all_edge_index).max(axis=1)

    all_points = np.concatenate([in_points, in_normals], axis=1)
    points_on_edges = all_points[edge_index > 0.5]
    np.random.shuffle(points_on_edges)

    return points_on_edges


# Used for Pool.map
def write_edges_points_to_ply(args):
    in_normals_ply, obj_path, save_path = args
    class_name = obj_path.split('/')[0]
    try:
        vertices_normal_in, normals_in = read_ply_point_normal(in_normals_ply)
        points_on_edges = calculate_edges(vertices_normal_in, normals_in)

        # Make sure folder for file exist
        save_folder, _ = os.path.split(save_path)
        os.makedirs(save_folder, exist_ok=True)
        write_ply_point_normal(
            save_path, 
            # In the original code, number of saved points lower\equal to 4096 (aka slice [:4096]) - but why? IDK
            # In our case (Template model) this constraint is harm to final evaluation
            # In case BSP-NET, it generate much lower number of points, so we dont bother about it
            points_on_edges
        )
    except:
        traceback.print_exc()
        print(f'Exception triggeret at object {obj_path}. Skip it')



def main(args):
    if args.predicted_folder is not None and args.normals_gt_folder is not None and \
            args.save_pd_folder is None and args.save_gt_folder is None:
        raise Exception('Path for save-pd-folder and save-gt-folder is None, but atleast one path must be provided')
    elif args.predicted_folder is not None and args.normals_gt_folder is not None and args.save_pd_folder is None:
        args.save_pd_folder = args.save_gt_folder
    elif args.predicted_folder is not None and args.normals_gt_folder is not None and args.save_gt_folder is None:
        args.save_gt_folder = args.save_pd_folder
    
    if args.predicted_folder is not None and args.save_pd_folder is None:
        raise Exception('Path for save-pd-folder is None, but sample is expected.')

    if args.normals_gt_folder is not None and args.save_gt_folder is None:
        raise Exception('Path for save-gt-folder is None, but sample is expected.')

    # name of objects
    obj_paths = [line.rstrip('\n') for line in open(args.obj_txt_file, mode='r').readlines()]

    # Sample for predicted objects
    if args.predicted_folder is not None and os.path.isdir(args.predicted_folder):
        print('Sample edge points for predicted objects...')
        # Setup args for generation
        eval_args = [
            (
                os.path.join(args.predicted_folder, obj_paths[i], 'obj_normals.ply'), 
                obj_paths[i], 
                os.path.join(args.save_pd_folder, obj_paths[i], 'obj_edge_pd_normals.ply'),
            ) 
            for i in range(len(obj_paths))
        ]

        with Pool(args.num_workers) as p:
            metric_res = list(tqdm(p.imap(write_edges_points_to_ply, eval_args), total=len(eval_args)))
        print('Sample edge points for predicted objects is done!')

    # Sample for ground truth objects
    if args.normals_gt_folder is not None and os.path.isdir(args.normals_gt_folder):
        print('Sample edge points for target objects...')
        # Setup args for generation
        eval_args = [
            (
                os.path.join(args.normals_gt_folder, f'{obj_paths[i]}.ply'), 
                obj_paths[i], 
                os.path.join(args.save_gt_folder, f'{obj_paths[i]}.ply'),
            ) 
            for i in range(len(obj_paths))
        ]

        with Pool(args.num_workers) as p:
            metric_res = list(tqdm(p.imap(write_edges_points_to_ply, eval_args), total=len(eval_args)))
        print('Sample edge points for target objects is done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted-folder', type=str, default=None,
                        help='Path to the folder with predicted ply objects. If equal to None, when sample for predicted points will be skiped.')
    parser.add_argument('--obj-txt-file', type=str,
                        help='Path to txt file with obj names to corresponding index in data path file. Example: `/your/path/to/obj_data.txt`.')
    parser.add_argument('--normals-gt-folder', type=str, default=None,
                        help='Path to the folder with ply files with normals. If equal to None, when sample for target points will be skiped.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of process to calculate metric.')
    parser.add_argument('--save-gt-folder', type=str, default=None,
                        help='Path to save sampled points for ground truth data. '
                        'If equal to None, when path will be taken from save-pd-folder path, otherwise error will be dropped.')
    parser.add_argument('--save-pd-folder', type=str, default=None,
                        help='Path to save sampled points for predicted data. '
                        'If equal to None, when path will be taken from save-gt-folder path, otherwise error will be dropped.')
    args = parser.parse_args()
    main(args)
