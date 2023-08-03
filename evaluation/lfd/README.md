# How to calculate Light Field Distance (LFD) metric

## Method 1 (Not recommended). Build and calculate from source
Compile\use code from [here](https://github.com/Sunwinds/ShapeDescriptor).

Its original code for calculation of distance between features obtained via descriptors from two objects to get lfd distance.

BUT, its a bit annoying and could led to some time consuming stuff and etc... 
Personally do not recommend this, so there is not code for such approuch

## Method 2 (Recomenned). Build and calculate via Docker
There is [repo](https://github.com/TaplierShiru/light-field-distance) which have python interface to interact with ShapeDescriptor binaries.

How to setup it and calculate metrics?
> 1. Build Docker from Dockerfile. There are examples in the repo how to build it with test samples. Before you build it, you need not mount current git-cloned repo and some folders with data (preditced ply objects and target one objects from ShapeNetV1\V2);
> 2. Now, in the Docker go to current path where placed README what you read now. To calculate lfd command could looks like:
```bash

```
> 3. Result could be found in `metrics_lfd.txt`.