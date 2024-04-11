# How to calculate Light Field Distance (LFD) metric

## Method 1 (Not recommended). Build and calculate from source
Compile\use code from [here](https://github.com/Sunwinds/ShapeDescriptor).

Its original code for calculation of distance between features obtained via descriptors from two objects to get lfd distance.

BUT, its a bit annoying and could led to some time consuming stuff and etc... 
Personally do not recommend this, so there is not code for such approuch.

## Method 2 (Recomenned). Build and calculate via Docker
There is [repo](https://github.com/TaplierShiru/light-field-distance) which have Dockerfile and python interface to interact with ShapeDescriptor binaries.

How to setup it and calculate metrics?
> 1. Build Docker from Dockerfile. There are examples in the repo how to build it with test samples. Before you build it, you need not mount current git-cloned repo and some folders with data (preditced ply objects and target one objects from ShapeNetV1\V2);
> 2. Now, in the Docker go to git-cloned repo and to this folder (i.e. evaluation/lfd, where is the current README). To calculate lfd command could looks like:
```bash
python3 calc_lfd.py ./metrics_svr_tf1_ldf.txt \
    -g /path/to/datasets/shapenet/ShapeNetCore.v1 \
    -p ./../../debug/generated_ae_ply_test \
    -s ./../../debug/generated_ae_lfd_test -n 6
```
> 3. Result could be found in `metrics_lfd.txt`.