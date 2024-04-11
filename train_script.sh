
#NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=2 train/image_trainer_ddp.py --resume_path configs/config_image_rgbd.py
python -m torch.distributed.launch --nproc_per_node=2 train/image_trainer_ddp.py --resume_path configs/config_image_rgbd_resnet50.py

# ResNet test
# python train/image_trainer.py --resume_path configs/config_image_resnet18.py
#python -m torch.distributed.launch --nproc_per_node=4 train/image_trainer_ddp.py --resume_path configs/config_image_resnet18bn.py
#python -m torch.distributed.launch --nproc_per_node=4 train/image_trainer_ddp.py --resume_path configs/config_image_resnet34.py
#python -m torch.distributed.launch --nproc_per_node=4 train/image_trainer_ddp.py --resume_path configs/config_image_resnet34bn.py
#python -m torch.distributed.launch --nproc_per_node=4 train/image_trainer_ddp.py --resume_path configs/config_image_resnet50.py


# Extend loss test
# python -m torch.distributed.launch --nproc_per_node=4 train/image_trainer_extended_ddp.py --resume_path configs/config_image_extended_add_after_700.py
# python -m torch.distributed.launch --nproc_per_node=4 train/image_trainer_extended_ddp.py --resume_path configs/config_image_extended.py
# python -m torch.distributed.launch --nproc_per_node=4 train/image_trainer_extended_ddp.py --resume_path configs/config_image_resnet50_extended_add_after_700.py
