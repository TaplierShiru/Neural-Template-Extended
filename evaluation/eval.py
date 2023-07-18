import traceback
import numpy as np
import random
import argparse
import importlib
import os

from tqdm import tqdm
from multiprocessing import Pool

try:
    from data.data import ImNetImageSamples
except ModuleNotFoundError:
    # Append base path with all needed code
    import pathlib
    import sys
    base_path, _ = os.path.split(pathlib.Path(__file__).parent.resolve())
    sys.path.append(base_path)
    # Try again
    from data.data import ImNetImageSamples

from utils.ply_utils import read_ply_point_normal, read_ply_point


def calculate_cd(gt_points, pd_points):
    gt_num_points = gt_points.shape[0]
    pd_num_points = pd_points.shape[0]

    gt_points_tiled = np.tile(gt_points.reshape(gt_num_points, 1, 3), [1, pd_num_points, 1])
    pd_points_tiled = np.tile(pd_points.reshape(1, pd_num_points, 3), [gt_num_points, 1, 1])

    dist = ((gt_points_tiled - pd_points_tiled) ** 2).sum(axis=2)
    match_pd_gt = np.argmin(dist, axis=0)
    match_gt_pd = np.argmin(dist, axis=1)

    dist_pd_gt = ((pd_points - gt_points[match_pd_gt]) ** 2).mean() * 3
    dist_gt_pd = ((gt_points - pd_points[match_gt_pd]) ** 2).mean() * 3
    chamfer_distance = dist_pd_gt + dist_gt_pd

    return chamfer_distance


def calculate_normal_consistency(gt_points, pd_points, gt_normals, pd_normals):
    gt_num_points = gt_points.shape[0]
    pd_num_points = pd_points.shape[0]

    gt_points_tiled = np.tile(gt_points.reshape(gt_num_points, 1, 3), [1, pd_num_points, 1])
    pd_points_tiled = np.tile(pd_points.reshape(1, pd_num_points, 3), [gt_num_points, 1, 1])

    dist = ((gt_points_tiled - pd_points_tiled) ** 2).sum(axis=2)
    match_pd_gt = np.argmin(dist, axis=0)
    match_gt_pd = np.argmin(dist, axis=1)

    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_pd_gt = (np.abs((pd_normals * gt_normals[match_pd_gt]).sum(axis=1))).mean()
    normals_dot_gt_pd = (np.abs((gt_normals * pd_normals[match_gt_pd]).sum(axis=1))).mean()
    normal_consistency = (normals_dot_pd_gt + normals_dot_gt_pd) / 2

    return normal_consistency


# Used for Pool.map
def calculate_cd_and_nt(args):
    vertices_gt, pd_ply, _, pd_normals_ply, gt_normal_ply, obj_path = args
    class_name = obj_path.split('/')[0]
    try:
        if gt_normal_ply is not None and pd_normals_ply is not None:
            vertices_normal_pd, normals_pd = read_ply_point_normal(pd_normals_ply)
            vertices_normal_gt, normals_gt = read_ply_point_normal(gt_normal_ply)

            normal_cons_calc = calculate_normal_consistency(
                vertices_normal_gt, vertices_normal_pd, 
                normals_gt, normals_pd
            )
        else:
            pass # Need func to read ply files (predicted) ...
        vertices_pd = read_ply_point(pd_ply)
        cd_calc = calculate_cd(vertices_gt, vertices_pd)

        return class_name, cd_calc, normal_cons_calc
    except:
        traceback.print_exc()
        print(f'Exception triggeret at object {obj_path}. Skip it')



def main(args):
    # dataload
    # create dataset
    samples = ImNetImageSamples(data_path=args.data_path, label_txt_path=args.obj_txt_file)

    # Setup parameters 
    cd_res_per_class_dict = dict([
        (name, []) 
        for name in list(set(map(lambda x: x.split('/')[0], samples.obj_paths)))
    ])

    normal_cons_res_per_class_dict = dict([
        (name, []) 
        for name in list(set(map(lambda x: x.split('/')[0], samples.obj_paths)))
    ])

    eval_args = [
        (
            samples.data_points[i][np.squeeze(samples.data_values[i] > 1e-4)], 
            os.path.join(args.predicted_folder, samples.obj_paths[i], 'obj_deformed.ply'), 
            os.path.join(args.predicted_folder, samples.obj_paths[i], 'obj_orginal.ply'), 
            os.path.join(args.predicted_folder, samples.obj_paths[i], 'obj_normals.ply')
                if args.normals_gt_folder is not None else None, 
            os.path.join(args.normals_gt_folder, f'{samples.obj_paths[i]}.ply') 
                if args.normals_gt_folder is not None else None,
            samples.obj_paths[i]
        ) 
        for i in range(len(samples))
    ]

    with Pool(args.num_workers) as p:
        metric_res = list(tqdm(p.imap(calculate_cd_and_nt, eval_args), total=len(eval_args)))

    for class_name, cd_calc, normal_cons_calc in filter(lambda x: x is not None, metric_res):
        if cd_res_per_class_dict.get(class_name) is None:
            pass
        else:
            cd_res_per_class_dict[class_name].append(cd_calc)

        if normal_cons_res_per_class_dict.get(class_name) is None:
            pass
        else:
            normal_cons_res_per_class_dict[class_name].append(normal_cons_calc)
    
    # Calculate for each class (category) and mean
    cd_res_mean_per_class_dict = dict([
        (k, sum(v) / len(v))
        for k,v in cd_res_per_class_dict.items()
    ])
    cd_res_mean = sum(cd_res_mean_per_class_dict.values()) / len(cd_res_mean_per_class_dict)

    if args.normals_gt_folder is not None:
        normal_cons_res_mean_per_class_dict = dict([
            (k, sum(v) / len(v))
            for k,v in normal_cons_res_per_class_dict.items()
        ])
        normal_cons_mean = sum(normal_cons_res_mean_per_class_dict.values()) / len(normal_cons_res_mean_per_class_dict)

    with open(args.save_txt, 'w') as fp:
        fp.write('Result per class\n')

        for k, v in cd_res_mean_per_class_dict.items():
            metric_str = f'class={k} cd={v} '

            if args.normals_gt_folder is not None:
                metric_str += f' nc={normal_cons_res_mean_per_class_dict[k]}\n'
            
            fp.write(metric_str)
        
        fp.write(f'Mean cd={cd_res_mean} nc={normal_cons_mean}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted-folder', type=str,
                        help='Path to the folder with predicted ply objects.')
    parser.add_argument('--data-path', type=str,
                        help='Path to h5 file which will be used for generation. Example: `/your/path/to/data.hdf5`.')
    parser.add_argument('--obj-txt-file', type=str,
                        help='Path to txt file with obj names to corresponding index in data path file. Example: `/your/path/to/obj_data.txt`.')
    parser.add_argument('--normals-gt-folder', type=str, default=None,
                        help='Path to the folder with ply files with normals.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of process to calculate metric.')
    parser.add_argument('-s', '--save-txt', type=str, default='./metrics.txt', 
                        help='Path to save calcualted metrics.')
    args = parser.parse_args()
    main(args)
