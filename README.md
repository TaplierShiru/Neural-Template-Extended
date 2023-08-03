# Neural-Template
[Neural Template: Topology-aware Reconstruction and Disentangled Generation of 3D Meshes (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Hui_Neural_Template_Topology-Aware_Reconstruction_and_Disentangled_Generation_of_3D_Meshes_CVPR_2022_paper.html)

![gallery](figures/gallery.png)

# Whats new compare to original repository
- Additional module - **Recognition module**. Mainly for classification task. **Encoder** give latent space vector which should be classified via **Recognition network** (which for now simple several MLPs);
- In some areas/cases clean code to better understand how this system works;
- Evaluation code to calculate: Chamfer distance (CD), Normal Consistency (NC), Point-to-surface distance (P2F) and Light field distance (LFD). Currently in development, will be added later.
- SOON...

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

To get metric, you could run next commands:

***IN DEVELOPMENT!***

### Metrics table

Chamfer distance (scaled by 1000). Reconstruction from voxels data.
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


Chamfer distance (scaled by 1000). Reconstruction from single image.
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
        <td>1.72</td> <!-- mean !-->
        <td>0.68</td> <!-- 02691156 !-->
        <td>1.79</td> <!-- 02828884 !-->
        <td>1.11</td> <!-- 02933112 !-->
        <td>0.56</td> <!-- 02958343 !-->
        <td>1.40</td> <!-- 03001627 !-->
        <td>2.97</td> <!-- 03211117 !-->
        <td>4.51</td> <!-- 03636649 !--> 
        <td>2.38</td> <!-- 03691459 !-->
        <td>0.99</td> <!-- 04090263 !-->
        <td>1.45</td> <!-- 04256520 !-->
        <td>1.91</td> <!-- 04379243 !-->
        <td>1.36</td> <!-- 04401088 !-->
        <td>1.22</td> <!-- 04530566 !-->
    </tr>
</table>


Normal consistency. Reconstruction from voxels data.
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
        <td>0.80</td> <!-- mean !-->
        <td>0.81</td> <!-- 02691156 !-->
        <td>0.78</td> <!-- 02828884 !-->
        <td>0.85</td> <!-- 02933112 !-->
        <td>0.83</td> <!-- 02958343 !-->
        <td>0.80</td> <!-- 03001627 !-->
        <td>0.85</td> <!-- 03211117 !-->
        <td>0.71</td> <!-- 03636649 !--> 
        <td>0.83</td> <!-- 03691459 !-->
        <td>0.71</td> <!-- 04090263 !-->
        <td>0.85</td> <!-- 04256520 !-->
        <td>0.82</td> <!-- 04379243 !-->
        <td>0.91</td> <!-- 04401088 !-->
        <td>0.73</td> <!-- 04530566 !-->
    </tr>
    <tr style="text-align:center">
        <td>DT-Net <br> current repo</td>
        <td>soon</td> <!-- mean !-->
        <td>soon</td> <!-- 02691156 !-->
        <td>soon</td> <!-- 02828884 !-->
        <td>soon</td> <!-- 02933112 !-->
        <td>soon</td> <!-- 02958343 !-->
        <td>soon</td> <!-- 03001627 !-->
        <td>soon</td> <!-- 03211117 !-->
        <td>soon</td> <!-- 03636649 !--> 
        <td>soon</td> <!-- 03691459 !-->
        <td>soon</td> <!-- 04090263 !-->
        <td>soon</td> <!-- 04256520 !-->
        <td>soon</td> <!-- 04379243 !-->
        <td>soon</td> <!-- 04401088 !-->
        <td>soon</td> <!-- 04530566 !-->
    </tr>
</table>


Normal consistency. Reconstruction from single image.
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
        <td>0.77</td> <!-- mean !-->
        <td>0.79</td> <!-- 02691156 !-->
        <td>0.74</td> <!-- 02828884 !-->
        <td>0.81</td> <!-- 02933112 !-->
        <td>0.82</td> <!-- 02958343 !-->
        <td>0.76</td> <!-- 03001627 !-->
        <td>0.80</td> <!-- 03211117 !-->
        <td>0.67</td> <!-- 03636649 !--> 
        <td>0.79</td> <!-- 03691459 !-->
        <td>0.67</td> <!-- 04090263 !-->
        <td>0.81</td> <!-- 04256520 !-->
        <td>0.79</td> <!-- 04379243 !-->
        <td>0.88</td> <!-- 04401088 !-->
        <td>0.70</td> <!-- 04530566 !-->
    </tr>
    <tr style="text-align:center">
        <td>DT-Net <br> current repo</td>
        <td>soon</td> <!-- mean !-->
        <td>soon</td> <!-- 02691156 !-->
        <td>soon</td> <!-- 02828884 !-->
        <td>soon</td> <!-- 02933112 !-->
        <td>soon</td> <!-- 02958343 !-->
        <td>soon</td> <!-- 03001627 !-->
        <td>soon</td> <!-- 03211117 !-->
        <td>soon</td> <!-- 03636649 !--> 
        <td>soon</td> <!-- 03691459 !-->
        <td>soon</td> <!-- 04090263 !-->
        <td>soon</td> <!-- 04256520 !-->
        <td>soon</td> <!-- 04379243 !-->
        <td>soon</td> <!-- 04401088 !-->
        <td>soon</td> <!-- 04530566 !-->
    </tr>
</table>


Light Field Descriptors (LFD). Reconstruction from voxels data.
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
        <td>3921.96</td> <!-- mean !-->
        <td>5537.46</td> <!-- 02691156 !-->
        <td>4527.83</td> <!-- 02828884 !-->
        <td>1953.88</td> <!-- 02933112 !-->
        <td>2854.70</td> <!-- 02958343 !-->
        <td>3225.88</td> <!-- 03001627 !-->
        <td>3161.02</td> <!-- 03211117 !-->
        <td>7245.25</td> <!-- 03636649 !--> 
        <td>2134.13</td> <!-- 03691459 !-->
        <td>6591.03</td> <!-- 04090263 !-->
        <td>2775.30</td> <!-- 04256520 !-->
        <td>3084.76</td> <!-- 04379243 !-->
        <td>2866.22</td> <!-- 04401088 !-->
        <td>5028.03</td> <!-- 04530566 !-->
    </tr>
    <tr style="text-align:center">
        <td>DT-Net <br> current repo</td>
        <td>3749.63</td> <!-- mean !-->
        <td>5171.50</td> <!-- 02691156 !-->
        <td>4320.86</td> <!-- 02828884 !-->
        <td>1951.22</td> <!-- 02933112 !-->
        <td>2788.22</td> <!-- 02958343 !-->
        <td>3044.63</td> <!-- 03001627 !-->
        <td>3033.83</td> <!-- 03211117 !-->
        <td>6950.49</td> <!-- 03636649 !--> 
        <td>2037.50</td> <!-- 03691459 !-->
        <td>6303.48</td> <!-- 04090263 !-->
        <td>2633.13</td> <!-- 04256520 !-->
        <td>2874.31</td> <!-- 04379243 !-->
        <td>2767.16</td> <!-- 04401088 !-->
        <td>4868.91</td> <!-- 04530566 !-->
    </tr>
</table>


Light Field Descriptors (LFD). Reconstruction from single image.
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
        <td>4236.66</td> <!-- mean !-->
        <td>5876.99</td> <!-- 02691156 !-->
        <td>4804.39</td> <!-- 02828884 !-->
        <td>2049.8 </td> <!-- 02933112 !-->
        <td>3020.48</td> <!-- 02958343 !-->
        <td>3561.79</td> <!-- 03001627 !-->
        <td>3633.37</td> <!-- 03211117 !-->
        <td>7725.46</td> <!-- 03636649 !--> 
        <td>2417.43</td> <!-- 03691459 !-->
        <td>6920.46</td> <!-- 04090263 !-->
        <td>3040.49</td> <!-- 04256520 !-->
        <td>3335.52</td> <!-- 04379243 !-->
        <td>3143.68</td> <!-- 04401088 !-->
        <td>5546.68</td> <!-- 04530566 !-->
    </tr>
    <tr style="text-align:center">
        <td>DT-Net <br> current repo</td>
        <td>4166.61</td> <!-- mean !-->
        <td>5698.13</td> <!-- 02691156 !-->
        <td>4794.26</td> <!-- 02828884 !-->
        <td>2087.51</td> <!-- 02933112 !-->
        <td>2963.32</td> <!-- 02958343 !-->
        <td>3571.69</td> <!-- 03001627 !-->
        <td>3479.32</td> <!-- 03211117 !-->
        <td>7542.94</td> <!-- 03636649 !--> 
        <td>2412.35</td> <!-- 03691459 !-->
        <td>6750.99</td> <!-- 04090263 !-->
        <td>3037.33</td> <!-- 04256520 !-->
        <td>3278.80</td> <!-- 04379243 !-->
        <td>3058.07</td> <!-- 04401088 !-->
        <td>5491.23</td> <!-- 04530566 !-->
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


