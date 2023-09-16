import traceback
import argparse
import math

import trimesh
from trimesh import transformations
from lfd import LightFieldDistance


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def calculate_single_ldf_metric(single_gt_filepath: str, single_pd_filepath: str, save_file: str, dataset_version: str):
    try:
        gt_mesh_ply: trimesh.Trimesh = as_mesh(
            trimesh.load(single_gt_filepath, file_type='obj', process=False, maintain_order=True, validate=False)
        )
        pd_mesh_ply: trimesh.Trimesh = as_mesh(
            trimesh.load(single_pd_filepath, file_type='ply', process=False, maintain_order=True, validate=False)
        )
        if dataset_version == 1:
            angle = -math.pi / 2 # -90 degrees
            dir_yaxis = [0, 1, 0]
            center = [0, 0, 0]
            
            rot_matrix = transformations.rotation_matrix(angle, dir_yaxis, center)
            pd_mesh_ply.apply_transform(rot_matrix)
        
        lfd_value: float = LightFieldDistance(verbose=True).get_distance(
            gt_mesh_ply, pd_mesh_ply
        )

        with open(save_file, 'w') as fw:
            fw.write(str(lfd_value))
        print(lfd_value)

    except Exception as e:
        traceback.print_exc()
        print(f'Something goes wrong with single_pd_filepath={single_pd_filepath} and single_gt_filepath={single_gt_filepath}, skip them...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-gt-filepath', type=str,
                        help='Path to ground truth obj file from ShapeNet.')
    parser.add_argument('--single-pd-filepath', type=str,
                        help='Path to predicted truth ply file.')
    parser.add_argument('--save-file', type=str, 
                        help='File path to write result.')
    parser.add_argument('-v', '--dataset-version', choices=[1, 2], type=int, default=1, 
                        help='Version of the input dataset. For version 1, rotation between generated and target object could be different. '
                       'In this repo, generated obj is align with version 2. To align with version 1 rotation by -90 for axis y must be done. '
                       'For any other stuff manual change required. ')
    args = parser.parse_args()
    calculate_single_ldf_metric(args.single_gt_filepath, args.single_pd_filepath, args.save_file, args.dataset_version)
    