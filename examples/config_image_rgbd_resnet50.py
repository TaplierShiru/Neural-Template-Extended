import torch
import torch.nn as nn

## config debugger
debug_base_folder = r'/path/to/svr_rgbd_resnet50'

#### setting for embedding
emb_dims = 1024
k = 20
dropout = 0.5
single_embbeding_size = 128
output_channels = single_embbeding_size * 3 # 128 for each module (three)

## setting for image encoder
img_ef_dim = 64
image_preferred_color_space = 3 # RGB 
image_invert_input = False # Should be as it is, otherwise preprocessing will be broken
use_depth = True
type_img_encoder = 'ImageEncoder' # ImageEncoder ImageEncoderOriginal
img_arch_type = 'resnet50' # resnet18 resnet34 resnet50
img_final_act_func = torch.sigmoid # torch.sigmoid torch.tanh None
type_block = 'ResNetBlockSMBN' # ResNetBlockSM ResNetBlockSMBN # affect only ResNeT18 and ResNet34
img_linear_use_bn = True # True False

#### setting for decoder
decoder_input_embbeding_size = 128
decoder_input_size = 3
decoder_activation = nn.LeakyReLU(negative_slope=0.02)
decoder_last_activation = lambda x: torch.clamp(x, 0, 1)

### setting for flow Decoder
flow_layers_dim = [256, 256, 256]
flow_input_dim = 3
flow_layer_type = 'concatsquash'
flow_activation = torch.nn.Softplus()
flow_trainable_T = True
flow_T = 1.0
flow_use_linear = True
atol = 1e-5
rtol = 1e-5
initial_mean = 0.5
ODE_networkt_type = 'ODE_ResNet'
flow_resnet_use_tanh = False
flow_resnet_use_softplus = False
flow_field_use_embedding_dim = True
flow_use_bsp_field = True
flow_use_split_dim = True
flow_use_mutilple_layers = True
flows_layers_cnt = 1
bsp_encoder_layers = [256, 512, 1024]
bsp_p_dim = 4096
bsp_c_dim = 32
bsp_phase = 1
bsp_thershold = 0.01
recognition_num_classes = 13

#### Training
data_worker = 1
coordinate_max_len = 500000
encoder_type = 'IMAGE' # Originally here 3DCNN, but I change code to readable state, so here now Image
decoder_type = 'Flow'
network_type = 'AutoEncoder'
lr = 5e-5
beta1 = 0.5
clamp_dist = 0.1
network_resume_path = None
optimizer_resume_path = None
data_type = 'IMNET'
data_folder = 'home'
data_path = r'/path/to/datasets/shapenet/ShapeNetDepthCore.v2.h5/splitted/dataset_train.hdf5'
sample_voxel_size = 16
sample_class = True # train with classes
class_loss_fn = nn.CrossEntropyLoss()
load_ram = False
batch_size = 32
loss_fn = nn.MSELoss() # nn.MSELoss
training_epochs = 1000
saving_intervals = 100
exp_idx = 200
starting_epoch = 0
starting_phase = 0
special_symbol = ''
half_batch_size_when_phase_2 = False
use_testing = False
testing_interval = 1
auto_encoder_config_path = r'/path/to/Neural-Template/pretrain/class_pretrain/phase_2_model/config.py'
auto_encoder_resume_path = r'/path/to/Neural-Template/pretrain/class_pretrain/phase_2_model/model_epoch_2_310.pth'
