# Neural-Template
[Neural Template: Topology-aware Reconstruction and Disentangled Generation of 3D Meshes (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Hui_Neural_Template_Topology-Aware_Reconstruction_and_Disentangled_Generation_of_3D_Meshes_CVPR_2022_paper.html)

![gallery](figures/gallery.png)

# Whats new compare to original repository
- Additional module - **Recognition module**. Mainly for classification task. **Encoder** give latent space vector which should be classified via **Recognition network** (which for now simple several MLPs);
- In some areas/cases clean code to better understand how this system works;
- Evaluation code to calculate: Chamfer distance (CD), Normal Consistency (NC), Point-to-surface distance (P2F) and Light field distance (LFD). Currently in development, will be added later.
- Examples could be found in `examples` folder with different examples of how run model.
- Weights of different model for notebooks in `examples` could be found [here](https://drive.google.com/drive/folders/1yKiK6nrX88UVsWSTFqLczdPSRcb1AWcX?usp=sharing).

# Environments
You can create and activate a conda environment for this project using the following commands:
```angular2html
conda env create -f environment.yml
conda activate NeuralTemplate
```

# Dataset
For the dataset, we use the dataset provided by [IM-NET](https://github.com/czq142857/IM-NET-pytorch) for training and evaluation. We provide another zip file ([link](https://drive.google.com/file/d/177bC-AresW8tMq54_q84K6Eav_hGxUZE/view?usp=sharing)) which should be unzipped in ```data``` before any training or inference.

# Training
There are altogether three commands for three training phases (Continuous training, Discrete training, Image encoder training).

For the continuous phase, you can train the model by using the following command:
```angular2html
python train/implicit_trainer.py --resume_path ./configs/config_continuous.py
```

For the discrete phase, you should first specify the ```network_resume_path``` variable in the ```configs/config_discrete.py``` as the model path in the previous phase. Then, you can run the following command:
```angular2html
python train/implicit_trainer.py --resume_path ./configs/config_discrete.py
```

Lastly, for the Image encoder training, you should specify the ```auto_encoder_config_path```  and ```auto_encoder_resume_path``` in ```config_image.py``` similarly, and run the following command to start training:
```angular2html
python train/image_trainer.py --resume_path ./configs/config_image.py
```

# Evaluation
We provide the pre-trained models used in the paper for reproducing the results. You can unzip the file ([link](https://drive.google.com/file/d/1--C2xUp0yao_nHDNvEpL3a1ZpTVC139J/view?usp=sharing)) in the ```pretrain``` folder.

Next command will produce .ply files from ether voxels or images.

For the evaluation of the autoencoder (input are voxels), you can use this command:
```angular2html
python utils/mesh_visualization.py
```

For the evaluation of the single view reconstruction task, you can use this command:
```angular2html
python utils/image_mesh_visualization.py
```

If you need test your own trained model, you change the ```testing_folder``` variable in these two python files.

To get ***metrics***, you could run next commands:

**Firstly**, we need to generate ply objects for the test set:
```bash
python3 evaluation/generate_mesh_ply.py \
    --config-path ./pretrain/phase_2_model/config.py \ 
    --network-path ./pretrain/phase_2_model/model_epoch_2_300.pth \
    --data-path ./data/all_vox256_img/all_vox256_img_test.hdf5 \
    --obj-txt-file ./data/all_vox256_img/all_vox256_img_test.txt \
    -s ./debug/generated_ae_ply_test \
    --input-type voxels --max-number-gpu 2 --device-ratio 2
```

Notice that loaded model is for voxels as input data, and generated ply objects will be from voxels as input.
If you need to generate ply objects from input data as images, command will be:
```bash
python3 evaluation/generate_mesh_ply.py \
    --config-path ./pretrain/image_encoder/config.py \ 
    --network-path ./pretrain/image_encoder/model_epoch_1000.pth \
    --data-path ./data/all_vox256_img/all_vox256_img_test.hdf5 \
    --obj-txt-file ./data/all_vox256_img/all_vox256_img_test.txt \
    -s ./debug/generated_svr_ply_test \
    --input-type image --max-number-gpu 2 --device-ratio 2
```

**Secondly**, on generated ply objects calculate metrics. 

To calculate Chamfer distance (CD):
```bash
python3 evaluation/eval.py \
    --predicted-folder ./debug/generated_svr_ply_test \
    --data-path ./data/all_vox256_img/all_vox256_img_test.hdf5 \
    --obj-txt-file ./data/all_vox256_img/all_vox256_img_test.txt \
    --num-workers 6 \
    -s ./debug/metrics_cd_ae.txt
```

This command will generate `metrics_cd_ae.txt` file with metrics for each class in ShapeNet dataset.

If saved model trained to classify objects, when on first stage (generation of ply stage) class id will be saved automaticly in each folder. On second stage for CD add `--sample-class`. Final `metrics_cd_ae.txt` will have additional data about classification results.

To calculate Light Field Distance (LFD), you need to setup special programm and get data from it. But there is solution via Docker and create special Python package to simplify the calculation. Refer to this [README](./evaluation/lfd/README.md).

### Metrics table for pretrained models

#### Chamfer distance (scaled by 1000). Reconstruction from voxels data.
<table>
    <tr style="text-align:center">
        <th></th>
        <th>mean </th>
        <th>airplane <br> 02691156 </th>
        <th>bench    <br> 02828884 </th>
        <th>cabinet  <br> 02933112 </th>
        <th>car      <br> 02958343 </th>
        <th>chair    <br> 03001627 </th>
        <th>display  <br> 03211117 </th>
        <th>lamp     <br> 03636649 </th>
        <th>speaker  <br> 03691459 </th>
        <th>rifle    <br> 04090263 </th>
        <th>couch    <br> 04256520 </th>
        <th>table    <br> 04379243 </th>
        <th>phone    <br> 04401088 </th>
        <th>vessel   <br> 04530566 </th>
    </tr>
    <tr style="text-align:center">
        <td>BSP Net</td>
        <td>0.75</td> <!-- mean !-->
        <td>0.41</td> <!-- 02691156 !-->
        <td>0.54</td> <!-- 02828884 !-->
        <td>0.75</td> <!-- 02933112 !-->
        <td>0.57</td> <!-- 02958343 !-->
        <td>0.74</td> <!-- 03001627 !-->
        <td>0.73</td> <!-- 03211117 !-->
        <td>1.59</td> <!-- 03636649 !--> 
        <td>1.16</td> <!-- 03691459 !-->
        <td>0.38</td> <!-- 04090263 !-->
        <td>0.68</td> <!-- 04256520 !-->
        <td>0.90</td> <!-- 04379243 !-->
        <td>0.49</td> <!-- 04401088 !-->
        <td>0.85</td> <!-- 04530566 !-->
    </tr>
    <tr style="text-align:center">
        <td>DT-Net <br> current repo</td>
        <td>0.64</td> <!-- mean !-->
        <td>0.31</td> <!-- 02691156 !-->
        <td>0.71</td> <!-- 02828884 !-->
        <td>0.69</td> <!-- 02933112 !-->
        <td>0.34</td> <!-- 02958343 !-->
        <td>0.68</td> <!-- 03001627 !-->
        <td>0.64</td> <!-- 03211117 !-->
        <td>1.24</td> <!-- 03636649 !--> 
        <td>0.78</td> <!-- 03691459 !-->
        <td>0.37</td> <!-- 04090263 !-->
        <td>0.60</td> <!-- 04256520 !-->
        <td>0.96</td> <!-- 04379243 !-->
        <td>0.48</td> <!-- 04401088 !-->
        <td>0.54</td> <!-- 04530566 !-->
    </tr>
</table>


#### Chamfer distance (scaled by 1000). Reconstruction from single image.
<table>
    <tr style="text-align:center">
        <th></th>
        <th>mean </th>
        <th>airplane <br> 02691156 </th>
        <th>bench    <br> 02828884 </th>
        <th>cabinet  <br> 02933112 </th>
        <th>car      <br> 02958343 </th>
        <th>chair    <br> 03001627 </th>
        <th>display  <br> 03211117 </th>
        <th>lamp     <br> 03636649 </th>
        <th>speaker  <br> 03691459 </th>
        <th>rifle    <br> 04090263 </th>
        <th>couch    <br> 04256520 </th>
        <th>table    <br> 04379243 </th>
        <th>phone    <br> 04401088 </th>
        <th>vessel   <br> 04530566 </th>
    </tr>
    <tr style="text-align:center">
        <td>BSP Net</td>
        <td>1.56</td> <!-- mean !-->
        <td>0.76</td> <!-- 02691156 !-->
        <td>1.23</td> <!-- 02828884 !-->
        <td>1.18</td> <!-- 02933112 !-->
        <td>0.84</td> <!-- 02958343 !-->
        <td>1.33</td> <!-- 03001627 !-->
        <td>1.85</td> <!-- 03211117 !-->
        <td>3.39</td> <!-- 03636649 !--> 
        <td>2.61</td> <!-- 03691459 !-->
        <td>0.89</td> <!-- 04090263 !-->
        <td>1.63</td> <!-- 04256520 !-->
        <td>1.64</td> <!-- 04379243 !-->
        <td>1.38</td> <!-- 04401088 !-->
        <td>1.58</td> <!-- 04530566 !-->
    </tr>
    <tr style="text-align:center">
        <td>DT-Net <br> current repo</td>
        <td>1.58</td> <!-- mean !-->
        <td>0.66</td> <!-- 02691156 !-->
        <td>1.47</td> <!-- 02828884 !-->
        <td>1.08</td> <!-- 02933112 !-->
        <td>0.49</td> <!-- 02958343 !-->
        <td>1.36</td> <!-- 03001627 !-->
        <td>2.36</td> <!-- 03211117 !-->
        <td>3.78</td> <!-- 03636649 !--> 
        <td>2.17</td> <!-- 03691459 !-->
        <td>0.99</td> <!-- 04090263 !-->
        <td>1.83</td> <!-- 04256520 !-->
        <td>1.92</td> <!-- 04379243 !-->
        <td>1.25</td> <!-- 04401088 !-->
        <td>1.24</td> <!-- 04530566 !-->
    </tr>
</table>


#### Light Field Descriptors (LFD). Reconstruction from voxels data.
<table>
    <tr style="text-align:center">
        <th></th>
        <th>mean </th>
        <th>airplane <br> 02691156 </th>
        <th>bench    <br> 02828884 </th>
        <th>cabinet  <br> 02933112 </th>
        <th>car      <br> 02958343 </th>
        <th>chair    <br> 03001627 </th>
        <th>display  <br> 03211117 </th>
        <th>lamp     <br> 03636649 </th>
        <th>speaker  <br> 03691459 </th>
        <th>rifle    <br> 04090263 </th>
        <th>couch    <br> 04256520 </th>
        <th>table    <br> 04379243 </th>
        <th>phone    <br> 04401088 </th>
        <th>vessel   <br> 04530566 </th>
    </tr>
    <tr style="text-align:center">
        <td>BSP Net</td>
        <td>3570.29</td> <!-- mean !-->
        <td>5039.14</td> <!-- 02691156 !-->
        <td>4174.47</td> <!-- 02828884 !-->
        <td>1574.36</td> <!-- 02933112 !-->
        <td>2553.22</td> <!-- 02958343 !-->
        <td>2880.59</td> <!-- 03001627 !-->
        <td>2726.24</td> <!-- 03211117 !-->
        <td>7153.81</td> <!-- 03636649 !--> 
        <td>1891.72</td> <!-- 03691459 !-->
        <td>6191.25</td> <!-- 04090263 !-->
        <td>2380.83</td> <!-- 04256520 !-->
        <td>2755.24</td> <!-- 04379243 !-->
        <td>2339.04</td> <!-- 04401088 !-->
        <td>4753.87</td> <!-- 04530566 !-->
    </tr>
    <tr style="text-align:center">
        <td>DT-Net <br> current repo</td>
        <td>3359.25</td> <!-- mean !-->
        <td>4582.39</td> <!-- 02691156 !-->
        <td>3930.30</td> <!-- 02828884 !-->
        <td>1568.40</td> <!-- 02933112 !-->
        <td>2452.96</td> <!-- 02958343 !-->
        <td>2682.05</td> <!-- 03001627 !-->
        <td>2538.04</td> <!-- 03211117 !-->
        <td>6832.69</td> <!-- 03636649 !--> 
        <td>1780.91</td> <!-- 03691459 !-->
        <td>5831.84</td> <!-- 04090263 !-->
        <td>2183.07</td> <!-- 04256520 !-->
        <td>2510.88</td> <!-- 04379243 !-->
        <td>2205.67</td> <!-- 04401088 !-->
        <td>4571.03</td> <!-- 04530566 !-->
    </tr>
</table>


#### Light Field Descriptors (LFD). Reconstruction from single image.
<table>
    <tr style="text-align:center">
        <th></th>
        <th>mean </th>
        <th>airplane <br> 02691156 </th>
        <th>bench    <br> 02828884 </th>
        <th>cabinet  <br> 02933112 </th>
        <th>car      <br> 02958343 </th>
        <th>chair    <br> 03001627 </th>
        <th>display  <br> 03211117 </th>
        <th>lamp     <br> 03636649 </th>
        <th>speaker  <br> 03691459 </th>
        <th>rifle    <br> 04090263 </th>
        <th>couch    <br> 04256520 </th>
        <th>table    <br> 04379243 </th>
        <th>phone    <br> 04401088 </th>
        <th>vessel   <br> 04530566 </th>
    </tr>
    <tr style="text-align:center">
        <td>BSP Net</td>
        <td>3946.31</td> <!-- mean !-->
        <td>5428.90</td> <!-- 02691156 !-->
        <td>4499.93</td> <!-- 02828884 !-->
        <td>1712.24</td> <!-- 02933112 !-->
        <td>2764.37</td> <!-- 02958343 !-->
        <td>3282.44</td> <!-- 03001627 !-->
        <td>3252.27</td> <!-- 03211117 !-->
        <td>7677.25</td> <!-- 03636649 !--> 
        <td>2258.08</td> <!-- 03691459 !-->
        <td>6587.36</td> <!-- 04090263 !-->
        <td>2728.94</td> <!-- 04256520 !-->
        <td>3055.21</td> <!-- 04379243 !-->
        <td>2715.77</td> <!-- 04401088 !-->
        <td>5339.28</td> <!-- 04530566 !-->
    </tr>
    <tr style="text-align:center">
        <td>DT-Net <br> current repo</td>
        <td>3866.41</td> <!-- mean !-->
        <td>5147.68</td> <!-- 02691156 !-->
        <td>4515.36</td> <!-- 02828884 !-->
        <td>1744.31</td> <!-- 02933112 !-->
        <td>2692.84</td> <!-- 02958343 !-->
        <td>3265.58</td> <!-- 03001627 !-->
        <td>3179.75</td> <!-- 03211117 !-->
        <td>7460.92</td> <!-- 03636649 !--> 
        <td>2244.66</td> <!-- 03691459 !-->
        <td>6410.54</td> <!-- 04090263 !-->
        <td>2691.44</td> <!-- 04256520 !-->
        <td>2969.33</td> <!-- 04379243 !-->
        <td>2664.94</td> <!-- 04401088 !-->
        <td>5275.94</td> <!-- 04530566 !-->
    </tr>
</table>

### Note: 
you need to set ```PYTHONPATH=<project root directory>``` before running any above commands.

### Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{hui2022template,
    title = {Neural Template: Topology-aware Reconstruction and Disentangled Generation of 3D Meshes},
    author = {Ka-Hei Hui* and Ruihui Li* and Jingyu Hu and Chi-Wing Fu(* joint first authors)},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```


