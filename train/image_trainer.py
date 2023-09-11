import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse

try:
    from models.network import AutoEncoder
except ModuleNotFoundError:
    # Append base path with all needed code
    import pathlib
    import sys
    base_path, _ = os.path.split(pathlib.Path(__file__).parent.resolve())
    sys.path.append(base_path)
    # Try again
    from models.network import AutoEncoder

from data.data import ImNetImageSamples
from torch.utils.data import DataLoader
from utils.debugger import MyDebugger


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ImageTrainer(object):

    def __init__(self, config, debugger, auto_encoder_config):
        self.debugger = debugger
        self.config = config
        self.auto_encoder_config = auto_encoder_config

    def train_network(self):
        if self.config.encoder_type.upper() != 'IMAGE':
            print('In config encoder type is not Image, but training IS for Image. '
                  'Change encoder type to Image'
            )
            self.config.encoder_type = 'IMAGE'
        
        ### Network
        network_state_dict = torch.load(self.config.auto_encoder_resume_path)
        network_state_dict, _ = AutoEncoder.process_state_dict(network_state_dict, type = 1)
        voxel_auto_encoder = AutoEncoder(self.auto_encoder_config)
        voxel_auto_encoder.load_state_dict(network_state_dict)
        voxel_auto_encoder.to(device)
        print(f"Reloaded the Auto encoder from {self.config.auto_encoder_resume_path}")

        ### create dataset
        train_samples = ImNetImageSamples(data_path=self.config.data_path,
                                          auto_encoder = voxel_auto_encoder,
                                          sample_class=hasattr(config, 'sample_class') and config.sample_class,
                                          use_depth=hasattr(config, 'use_depth') and config.use_depth,
                                          image_preferred_color_space=config.image_preferred_color_space if hasattr(config, 'image_preferred_color_space') else 1)

        train_data_loader = DataLoader(dataset=train_samples,
                                       batch_size=self.config.batch_size,
                                       num_workers=self.config.data_worker,
                                       shuffle=True,
                                       drop_last=False)

        if hasattr(self.config, 'use_testing') and self.config.use_testing:
            test_samples = ImNetImageSamples(data_path=self.config.data_path[:-10] + 'test.hdf5',
                                        auto_encoder=voxel_auto_encoder,
                                        sample_class=hasattr(config, 'sample_class') and config.sample_class)
            test_data_loader = DataLoader(dataset=test_samples,
                                          batch_size=self.config.batch_size,
                                          num_workers=config.data_worker,
                                          shuffle=True,
                                          drop_last=False)

        ### set up network
        network = AutoEncoder(self.config)
        # Preload decoder weights to proper save later, or usage in some cases
        network.decoder.load_state_dict(voxel_auto_encoder.decoder.state_dict())
        # Remove it, not needed anymore
        del voxel_auto_encoder
        # Decoder parameters are not trained, only encoder one
        for param in network.decoder.parameters():
            param.requires_grad = False
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUS!")
            network = nn.DataParallel(network)
        network = network.to(device)
        
        loss_fn = torch.nn.MSELoss()

        self.config.auto_encoder_resume_path = None

        ## reload the network if needed
        if self.config.network_resume_path is not None:
            network_state_dict = torch.load(self.config.network_resume_path)
            network_state_dict = AutoEncoder.process_state_dict(network_state_dict)
            network.load_state_dict(network_state_dict)
            network.train()
            print(f"Reloaded the network from {self.config.network_resume_path}")
            self.config.network_resume_path = None

        if torch.cuda.device_count() > 1:
            optimizer = torch.optim.Adam(params=network.module.encoder.parameters(), lr=self.config.lr,
                                         betas=(self.config.beta1, 0.999))
        else:
            optimizer = torch.optim.Adam(params=network.encoder.parameters(), lr=self.config.lr,
                                         betas=(self.config.beta1, 0.999))

        if self.config.optimizer_resume_path is not None:
            optimizer_state_dict = torch.load(self.config.optimizer_resume_path)
            optimizer.load_state_dict(optimizer_state_dict)
            print(f"Reloaded the optimizer from {self.config.optimizer_resume_path}")
            self.config.optimizer_resume_path = None

        for idx in range(self.config.starting_epoch, self.config.training_epochs + 1):

            ## training testing
            with tqdm(train_data_loader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {idx}')
                losses = []

                is_training = True
                losses = self.evaluate_one_epoch(is_training, loss_fn, losses, network, optimizer, tepoch)
                print(f"Train Loss for epoch {idx} : {np.mean(losses)}")

                if idx % self.config.saving_intervals == 0:
                    save_model_path = self.debugger.file_path(f'model_epoch_{idx}.pth')
                    save_optimizer_path = self.debugger.file_path(f'optimizer_epoch_{idx}.pth')
                    torch.save(network.state_dict(), save_model_path)
                    torch.save(optimizer.state_dict(), save_optimizer_path)
                    print(f"Epoch {idx} model saved at {save_model_path}")
                    print(f"Epoch {idx} optimizer saved at {save_optimizer_path}")
                    self.config.network_resume_path = save_model_path  ## add this resume after the whole things are compelete

                    if hasattr(self.config, 'use_testing') and self.config.use_testing:
                        losses = []
                        is_training = False
                        with tqdm(test_data_loader, unit='batch') as tepoch:
                            tepoch.set_description(f'Epoch {idx}')
                            losses = self.evaluate_one_epoch(is_training, loss_fn, losses, network, optimizer, tepoch)
                            print(f"Test Loss for epoch {idx} : {np.mean(losses)}")


    def evaluate_one_epoch(self, is_training, loss_fn, losses, network, optimizer, tepoch):
        ###
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ###
        for inputs, samples_indices in tepoch:
            if hasattr(config, 'sample_class') and self.config.sample_class:
                input_images, latent_vector_gt, class_ground_truth = inputs
                # Keep class info for eval and future loss updates
                class_ground_truth = class_ground_truth.to(device)
            else:
                input_images, latent_vector_gt = inputs
            input_images, latent_vector_gt = input_images.to(device), latent_vector_gt.to(device)

            if is_training:
                network.train()
                optimizer.zero_grad()
            else:
                network.eval()

            pred_latent_vector = network(input_images)
            loss = loss_fn(pred_latent_vector, latent_vector_gt)
            losses.append(loss.item())

            if is_training:
                loss.backward()
                optimizer.step()

            tepoch.set_postfix(loss=f'{np.mean(losses)}')

        return losses


if __name__ == '__main__':
    import importlib

    ## additional args for parsing
    optional_args = [("network_resume_path", str), ("optimizer_resume_path", str), ("starting_epoch", int),
                     ("special_symbol", str), ("resume_path", str), ("starting_phase", int)]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)

    args = parser.parse_args()
    ## Resume setting
    resume_path = None

    ## resume from path if needed
    if args.resume_path is not None:
        resume_path = args.resume_path

    if resume_path is None:
        from configs import config
        resume_path = os.path.join('configs', 'config.py')
    else:
        ## import config here
        spec = importlib.util.spec_from_file_location('*', resume_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    for optional_arg, arg_type in optional_args:
        if args.__dict__.get(optional_arg, None) is not None:
            locals()['config'].__setattr__(optional_arg, args.__dict__.get(optional_arg, None))


    ### resume
    assert config.auto_encoder_config_path is not None and os.path.exists(config.auto_encoder_config_path) and \
           config.auto_encoder_resume_path is not None and os.path.exists(config.auto_encoder_resume_path)
    auto_spec = importlib.util.spec_from_file_location('*', config.auto_encoder_config_path)
    auto_config = importlib.util.module_from_spec(auto_spec)
    auto_spec.loader.exec_module(auto_config)


    model_type = f"AutoEncoder-{config.encoder_type}-{config.decoder_type}" if config.network_type == 'AutoEncoder' else f"AutoDecoder-{config.decoder_type}"
    debugger = MyDebugger(
        f'Image-Training-experiment-{os.path.basename(config.data_folder)}-{model_type}{config.special_symbol}', 
        is_save_print_to_file = True, 
        config_path = resume_path,
        config=config,
    )
    trainer = ImageTrainer(config = config, debugger = debugger, auto_encoder_config = auto_config)
    trainer.train_network()