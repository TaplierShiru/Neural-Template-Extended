import traceback
from typing import List
import os
from multiprocessing import Process
import subprocess
import argparse
from dataclasses import dataclass
import sys
from contextlib import contextmanager

from tqdm import tqdm


class_name_list_all = [
    "02691156_airplane",
    "02828884_bench",
    "02933112_cabinet",
    "02958343_car",
    "03001627_chair",
    "03211117_display",
    "03636649_lamp",
    "03691459_speaker",
    "04090263_rifle",
    "04256520_couch",
    "04379243_table",
    "04401088_phone",
    "04530566_vessel",
]


@dataclass
class ProcessedSingleData:
    category: str
    score: float


@dataclass
class StoredPathData:
    category: str
    pd_ply_filepath: str
    gt_ply_filepath: str
    save_res_file: str

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    How to use it:
    ```
        import os

        with stdout_redirected(to=filename):
            print("from Python")
            os.system("echo non-Python applications are also supported")
        # Or to skip any print
        with stdout_redirected():
            print("from Python")
            os.system("echo non-Python applications are also supported")
    ```

    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different


def calculate_ldf_metric(args_files_path_list: List[StoredPathData], start_indx: int, end_indx: int, indx_process: int, dataset_version: str):
    for (files_info, indx) in zip(args_files_path_list, range(start_indx, end_indx)):
        if (indx - start_indx) % 30:
            print(f'{indx_process} done {indx - start_indx}/{end_indx - start_indx}')
        try:
            if os.path.isfile(files_info.save_res_file):
                continue
            
            # This state will silent all prints from trimesh and lfd calculation
            # Its possible to direct these prints to some file, but they do not needed here
            with stdout_redirected():
                subprocess.run(
                    f'Xvfb :{indx_process} -screen 0 1900x1080x24+32 & export DISPLAY=:{indx_process} && python3 calc_single_ldf.py ' +\
                        f'--single-gt-filepath {files_info.gt_ply_filepath} --single-pd-filepath {files_info.pd_ply_filepath} ' +\
                        f'--save-file {files_info.save_res_file} -v {dataset_version}', 
                    shell=True
                )
        except Exception as e:
            traceback.print_exc()
            print(f'Something goes wrong with file={files_info} and indx={indx}, skip them...')
            continue


def main(args):
    output_file = open(
        args.output_file,
        'w', newline=''
    )
    ldf_metric_dict = dict()

    print(f'ShapeNet dataset version: {args.dataset_version}')

    for category_with_name in class_name_list_all:
        category, name_class = category_with_name.split('_')
        ldf_metric_dict[category] = {'mean': None, 'all': []}
        print(f'Prepare dataset with category {category} and name {name_class}')
        os.makedirs(os.path.join(args.save_folder, category), exist_ok=True)
        
        args_files_path_list = list(map(
            lambda model_id: StoredPathData(
                category=category, 
                gt_ply_filepath=os.path.join(args.gt_folder, category, model_id, 'models/model_normalized.obj' if args.dataset_version == 2 else 'model.obj'), 
                pd_ply_filepath=os.path.join(args.pd_folder, category, model_id, 'obj_deformed.ply'), 
                save_res_file=os.path.join(args.save_folder, category, f'{model_id}.txt'),
            ),
            os.listdir(os.path.join(args.pd_folder, category))
        ))
        num_of_files = len(args_files_path_list)

        # Write category and model id paths to separate file
        output_file.write(f'{category} {name_class} ')
        
        # Make list for each process
        number_files_per_process = num_of_files // args.num_process
        args_to_generate_dataset_per_process = [
            (
                args_files_path_list[i * number_files_per_process: (i+1) * number_files_per_process],
                i * number_files_per_process, (i+1) * number_files_per_process
            )
            for i in range(args.num_process-1)
        ]
        # Append the remaining files for last process
        last_indx_start = (args.num_process-1) * number_files_per_process
        last_files = args_files_path_list[last_indx_start:]
        args_to_generate_dataset_per_process.append(
            (
                last_files,
                last_indx_start, 
                last_indx_start + len(last_files)
            )
        )
        
        workers = [
            Process(target=calculate_ldf_metric, args = (args_files_path_list, start_indx, end_indx, indx, args.dataset_version)) 
            for indx, (args_files_path_list, start_indx, end_indx) in enumerate(args_to_generate_dataset_per_process)
        ]

        for p in workers:
            p.start()

        for p in workers:
            p.join()

        for single_files_path in args_files_path_list:
            with open(single_files_path.save_res_file, 'r') as fr:
                ldf_value = float(fr.readline())
            ldf_metric_dict[single_files_path.category]['all'].append(ldf_value)
        current_metric_mean = sum(ldf_metric_dict[category]['all']) / len(ldf_metric_dict[category]['all'])
        ldf_metric_dict[category]['mean'] = current_metric_mean
        output_file.write(f'mean={current_metric_mean} \n')
        output_file.flush()
    
    avg_mean = sum([s_res['mean'] for s_res in ldf_metric_dict.values()]) / len(ldf_metric_dict.keys())
    output_file.write(f'avg mean={avg_mean} \n')
    output_file.close()
    print("finished")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=str,
                        help='Path to write calculated metrics. Example: ./metrics.txt')
    parser.add_argument('-p', '--pd_folder', type=str,
                        help='Path to folder predicted ply objects.')
    parser.add_argument('-g', '--gt_folder', type=str,
                        help='Path to folder with original obj files from ShapeNet.')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save calculated ldf for each model.')
    parser.add_argument('-v', '--dataset-version', choices=[1, 2], type=int, default=1, 
                        help='Version of the input dataset. For version 1, rotation between generated and target object could be different. '
                       'In this repo, generated obj is align with version 2. To align with version 1 rotation by -90 for axis y must be done. '
                       'For any other stuff manual change required. ')
    parser.add_argument('-n', '--num-process', type=int, 
                        help='Number of process.', default=4)
    args = parser.parse_args()
    main(args)
    