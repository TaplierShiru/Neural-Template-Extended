import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import argparse

try:
    from models.network import ImageAutoEncoderEndToEnd
except ModuleNotFoundError:
    # Append base path with all needed code
    import pathlib
    import sys
    base_path, _ = os.path.split(pathlib.Path(__file__).parent.resolve())
    sys.path.append(base_path)
    # Try again
    from models.network import ImageAutoEncoderEndToEnd

from train.implicit_trainer import Trainer
from data.data import ImNetAllDataSamples
from torch.utils.data import DataLoader
from utils.debugger import MyDebugger


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ImageTrainer(object):

    def __init__(self, config, debugger, auto_encoder_cofig):
        self.debugger = debugger
        self.config = config
        self.auto_encoder_cofig = auto_encoder_cofig

    def train_network(self):


        ### Network
        network = ImageAutoEncoderEndToEnd(
            config=self.config, 
            auto_encoder_config=self.auto_encoder_cofig
        )
        network = network.to(device)
        self.config.auto_encoder_resume_path = None

        ### create dataset
        train_samples = ImNetAllDataSamples(data_path=self.config.data_path,
                                          auto_encoder = network.auto_encoder,
                                          sample_class=hasattr(config, 'sample_class') and config.sample_class,
                                          use_depth=hasattr(config, 'use_depth') and config.use_depth,
                                          image_preferred_color_space=config.image_preferred_color_space if hasattr(config, 'image_preferred_color_space') else 1)

        train_data_loader = DataLoader(dataset=train_samples,
                                       batch_size=self.config.batch_size,
                                       num_workers=self.config.data_worker,
                                       shuffle=True,
                                       drop_last=False)

        if hasattr(self.config, 'use_testing') and self.config.use_testing:
            test_samples = ImNetAllDataSamples(data_path=self.config.data_path[:-10] + 'test.hdf5',
                                        auto_encoder=auto_encoder,
                                        sample_class=hasattr(config, 'sample_class') and config.sample_class)
            test_data_loader = DataLoader(dataset=test_samples,
                                          batch_size=self.config.batch_size,
                                          num_workers=config.data_worker,
                                          shuffle=True,
                                          drop_last=False)

        ### set up network
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUS!")
            network = nn.DataParallel(network)

        loss_fn = torch.nn.MSELoss()

        ## reload the network if needed
        if self.config.network_resume_path is not None:
            network_state_dict = torch.load(self.config.network_resume_path)
            network_state_dict = Trainer.process_state_dict(network_state_dict)
            network.load_state_dict(network_state_dict)
            network.train()
            print(f"Reloaded the network from {self.config.network_resume_path}")
            self.config.network_resume_path = None

        if torch.cuda.device_count() > 1:
            optimizer = torch.optim.Adam(params=network.module.image_encoder.parameters(), lr=self.config.lr,
                                         betas=(self.config.beta1, 0.999))
        else:
            optimizer = torch.optim.Adam(params=network.image_encoder.parameters(), lr=self.config.lr,
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
                losses = self.evaluate_one_epoch(
                    is_training, loss_fn, losses, 
                    network, optimizer, tepoch, 
                    use_additional_losses=idx >= self.config.use_additional_losses_after_epochs
                )
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


    def evaluate_one_epoch(self, is_training, loss_fn, losses, network, optimizer, tepoch, use_additional_losses):
        ###
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ###
        for inputs, samples_indices in tepoch:
            if hasattr(config, 'sample_class') and self.config.sample_class:
                voxels_inputs, coordinate_inputs, occupancy_ground_truth, input_images, latent_vector_gt, class_ground_truth = inputs
                # Keep class info for eval and future loss updates
                class_ground_truth = class_ground_truth.to(device)
            else:
                voxels_inputs, coordinate_inputs, occupancy_ground_truth, input_images, latent_vector_gt = inputs
            input_images, latent_vector_gt = input_images.to(device), latent_vector_gt.to(device)
            voxels_inputs, coordinate_inputs, occupancy_ground_truth = voxels_inputs.to(
                device), coordinate_inputs.to(device), occupancy_ground_truth.to(
                device)

            if is_training:
                network.train()
                optimizer.zero_grad()
            else:
                network.eval()

            if not use_additional_losses:
                # TODO: Handle different architecture network, 
                #       otherwise keep interface same for every new network
                if torch.cuda.device_count() > 1:
                    pred_latent_vector = network.module.image_encoder(input_images)
                else:
                    pred_latent_vector = network.image_encoder(input_images)
                ## output results
                loss = loss_fn(pred_latent_vector, latent_vector_gt)
                losses.append(loss.item())
            else:
                # TODO: Handle different architecture network, 
                #       otherwise keep interface same for every new network
                prediction, pred_latent_vector = network(input_images, coordinate_inputs)

                if hasattr(self.config, 'sample_class') and self.config.sample_class:
                    convex_prediction, prediction, exist, convex_layer_weights, class_prediction = (
                        self.extract_prediction(prediction)
                    )
                else:
                    class_prediction = None
                    convex_prediction, prediction, exist, convex_layer_weights = self.extract_prediction(prediction)

                ## output results
                loss = loss_fn(pred_latent_vector, latent_vector_gt)
                losses.append(loss.item())
                ode_loss = loss = self.config.loss_fn(torch.clamp(prediction, min=0, max=1), occupancy_ground_truth)
                loss = loss + ode_loss
                losses.append(ode_loss.item())

                if hasattr(self.config, 'sample_class') and self.config.sample_class and class_prediction is not None:
                    class_loss = self.config.class_loss_fn(class_prediction, class_ground_truth.long())
                    # (N, n_class)
                    class_accuracy = (
                        F.softmax(class_prediction, dim=0).argmax(dim=-1).int() == class_ground_truth
                    ).float().mean()
                    if avg_accuracy is None:
                        avg_accuracy = class_accuracy
                    else:
                        avg_accuracy = 0.9 * class_accuracy + (1 - 0.9) * class_accuracy
                    print_dict['recognition'] = '{:.4f}'.format(class_loss)
                    print_dict['acc'] = '{:.2%}'.format(avg_accuracy)
                    
                    loss = loss + class_loss
                    losses.append(class_loss.item())

                ### loss function to be refactor
                
                if (self.config.decoder_type == 'Flow' and self.config.flow_use_bsp_field == True) or self.config.decoder_type == 'MVP':
                    loss, losses = self.flow_bsp_loss(loss, losses, network,
                                                    occupancy_ground_truth,
                                                    prediction, convex_layer_weights)
                else:
                    raise Exception("Unknown Network Type....")
            
            if is_training:
                loss.backward()
                optimizer.step()

            tepoch.set_postfix(loss=f'{np.mean(losses)}')

        return losses

    def flow_bsp_loss(self, loss, losses, network, occupancy_ground_truth, prediction, convex_layer_weights):

        bsp_thershold = self.config.bsp_thershold if hasattr(self.config, 'bsp_thershold') else 0.01
        if self.config.bsp_phase == 0:
            concave_layer_weights = network.auto_encoder.decoder.bsp_field.concave_layer_weights \
                if torch.cuda.device_count() <= 1 \
                    else network.module.auto_encoder.decoder.bsp_field.concave_layer_weights
            losses.append(loss.detach().item())
            loss = loss + torch.sum(
                torch.abs(concave_layer_weights - 1))  ### convex layer weight close to 1
            loss = loss + torch.sum(
                torch.clamp(convex_layer_weights - 1, min=0) - torch.clamp(convex_layer_weights,
                                                                           max=0))
        elif self.config.bsp_phase == 1:
            loss = torch.mean((1 - occupancy_ground_truth) * (
                    1 - torch.clamp(prediction, max=1)) + occupancy_ground_truth * torch.clamp(
                prediction, min=0))
            losses.append(loss.detach().item())
            loss = loss + torch.sum(
                (convex_layer_weights < bsp_thershold).float() * torch.abs(
                    convex_layer_weights)) + torch.sum(
                (convex_layer_weights >= bsp_thershold).float() * torch.abs(convex_layer_weights - 1))
        else:
            raise Exception("Unknown Phase.....")


        return loss, losses

    def extract_prediction(self, predictions_packed):
        assert self.config.decoder_type == 'Flow' and self.config.flow_use_bsp_field == True
        convex_prediction, prediction, exist, convex_layer_weights = predictions_packed[:4]
        if hasattr(self.config, 'sample_class') and self.config.sample_class:
            predicted_class = predictions_packed[-1] 
            return convex_prediction, prediction, exist, convex_layer_weights, predicted_class
        return convex_prediction, prediction, exist, convex_layer_weights


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
    trainer = ImageTrainer(config = config, debugger = debugger, auto_encoder_cofig = auto_config)
    trainer.train_network()