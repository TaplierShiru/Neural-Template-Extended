import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

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

from data.data import ImNetSamples
from torch.utils.data import DataLoader
from utils.debugger import MyDebugger

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.backends.cudnn.benchmark = True

class Trainer:

    def __init__(self, config, debugger):
        self.debugger = debugger
        self.config = config

    def train_network(self):

        phases = range(self.config.starting_phase, 3) if hasattr(self.config, 'starting_phase') else [
            int(np.log2(self.config.sample_voxel_size // 16))]
        
        if isinstance(self.config.batch_size, list) and max(phases) >= len(self.config.batch_size):
            raise Exception(f'Current list of batch sizes does not have value for indx (phase) {max(phases)}. ')

        for phase in phases:
            print(f"Start Phase {phase}")
            sample_voxel_size = 16 * (2 ** (phase))

            if isinstance(self.config.batch_size, list):
                batch_size = self.config.batch_size[phase]
            else:
                batch_size = self.config.batch_size 

            if phase == 2:
                if not hasattr(config, 'half_batch_size_when_phase_2') or self.config.half_batch_size_when_phase_2:
                    batch_size = batch_size // 2
                self.config.training_epochs = self.config.training_epochs * 2

            ### create dataset
            train_samples = ImNetSamples(data_path=self.config.data_path,
                                         sample_voxel_size=sample_voxel_size, 
                                         sample_class=hasattr(config, 'sample_class') and config.sample_class)

            train_data_loader = DataLoader(dataset=train_samples,
                                     batch_size=batch_size,
                                     num_workers=config.data_worker,
                                     shuffle=True,
                                     drop_last=False)

            if hasattr(self.config, 'use_testing') and self.config.use_testing:
                test_samples = ImNetSamples(data_path=self.config.data_test_path,
                                            sample_voxel_size=sample_voxel_size,
                                            interval=self.config.testing_interval, 
                                            sample_class=hasattr(config, 'sample_class') and config.sample_class)

                test_data_loader = DataLoader(dataset=test_samples,
                                               batch_size=batch_size,
                                               num_workers=config.data_worker,
                                               shuffle=True,
                                               drop_last=False)

            if self.config.network_type == 'AutoEncoder':
                network = AutoEncoder(config=self.config)
            else:
                raise Exception("Unknown Network type!")

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                print(f"Use {torch.cuda.device_count()} GPUS!")
                network = nn.DataParallel(network)
            network = network.to(device)

            ## reload the network if needed
            if self.config.network_resume_path is not None:
                network_state_dict = torch.load(self.config.network_resume_path)
                network_state_dict = Trainer.process_state_dict(network_state_dict)
                network.load_state_dict(network_state_dict)
                print(f"Reloaded the network from {self.config.network_resume_path}")
                self.config.network_resume_path = None
            optimizer = torch.optim.Adam(params=network.parameters(), lr=self.config.lr,
                                         betas=(self.config.beta1, 0.999))
            if self.config.optimizer_resume_path is not None:
                optimizer_state_dict = torch.load(self.config.optimizer_resume_path)
                optimizer.load_state_dict(optimizer_state_dict)
                print(f"Reloaded the optimizer from {self.config.optimizer_resume_path}")
                self.config.optimizer_resume_path = None

            for idx in range(self.config.starting_epoch, self.config.training_epochs + 1):
                with tqdm(train_data_loader, unit='batch') as tepoch:
                    tepoch.set_description(f'Epoch {idx}')
                    losses = self.evaluate_one_epoch(network, optimizer, tepoch, is_training = True)

                    print(f"Test Loss for epoch {idx} : {np.mean(losses)}")


                    ## saving the models
                    if idx % self.config.saving_intervals == 0:
                        # save
                        save_model_path = self.debugger.file_path(f'model_epoch_{phase}_{idx}.pth')
                        save_optimizer_path = self.debugger.file_path(f'optimizer_epoch_{phase}_{idx}.pth')
                        torch.save(network.state_dict(), save_model_path)
                        torch.save(optimizer.state_dict(), save_optimizer_path)
                        print(f"Epoch {idx} model saved at {save_model_path}")
                        print(f"Epoch {idx} optimizer saved at {save_optimizer_path}")
                        self.config.network_resume_path = save_model_path  ## add this resume after the whole things are compelete

                        if hasattr(self.config, 'use_testing') and self.config.use_testing:
                            with tqdm(test_data_loader, unit='batch') as tepoch:
                                losses, avg_accuracy = self.evaluate_one_epoch(
                                    network, optimizer = None, 
                                    tepoch = tepoch, is_training=False,
                                    return_avg_accuracy=True
                                )
                                final_print = 'Test epoch {} | Loss : {:.4f} '.format(idx, np.mean(losses))
                                if hasattr(self.config, 'sample_class') and self.config.sample_class and avg_accuracy is not None:
                                    final_print += '; avg accuracy : {:.2%}'.format(avg_accuracy)
                                print(final_print)
                                

            ## when done the phase
            self.config.starting_epoch = 0

    def evaluate_one_epoch(self, network, optimizer, tepoch, is_training = True, return_avg_accuracy=False):
        if is_training:
            network.train()
            return self._evaluate_loop(
                network, optimizer, 
                tepoch, is_training=is_training, 
                return_avg_accuracy=return_avg_accuracy
            )

        network.eval()
        with torch.no_grad():
            return self._evaluate_loop(
                network, optimizer, 
                tepoch, is_training=is_training, 
                return_avg_accuracy=return_avg_accuracy
            )

    def _evaluate_loop(self, network, optimizer, tepoch, is_training = True, return_avg_accuracy=False):
        losses = []
        ## main training loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        avg_accuracy = None

        for inputs, samples_indices in tepoch:
            print_dict = {}
            ## get voxel_inputs
            if hasattr(self.config, 'sample_class') and self.config.sample_class:
                voxels_inputs, coordinate_inputs, occupancy_ground_truth, class_ground_truth = inputs
                class_ground_truth = class_ground_truth.to(device)
            else:
                voxels_inputs, coordinate_inputs, occupancy_ground_truth = inputs
            # TODO: Is it used for Normal Consistency metric here?
            normals_gt = None

            ## remove gradient
            if is_training:
                optimizer.zero_grad()
                network.zero_grad()

            voxels_inputs, coordinate_inputs, occupancy_ground_truth, samples_indices = voxels_inputs.to(
                device), coordinate_inputs.to(device), occupancy_ground_truth.to(
                device), samples_indices.to(device)

            if self.config.network_type == 'AutoEncoder':
                prediction = network(voxels_inputs, coordinate_inputs)
            else:
                raise Exception("Unknown Network Type....")

            if hasattr(self.config, 'sample_class') and self.config.sample_class:
                convex_prediction, prediction, exist, convex_layer_weights, class_prediction = (
                    self.extract_prediction(prediction)
                )
            else:
                class_prediction = None
                convex_prediction, prediction, exist, convex_layer_weights = self.extract_prediction(prediction)

            loss = self.config.loss_fn(torch.clamp(prediction, min=0, max=1), occupancy_ground_truth)

            ### loss function to be refactor
            if (self.config.decoder_type == 'Flow' and self.config.flow_use_bsp_field == True) or self.config.decoder_type == 'MVP':
                loss, losses = self.flow_bsp_loss(loss, losses, network,
                                                  occupancy_ground_truth,
                                                  prediction, convex_layer_weights)
            else:
                raise Exception("Unknown Network Type....")

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

            if is_training:
                loss.backward()
                optimizer.step()

            print_dict['loss'] = '{:.4f}'.format(np.mean(losses))
            tepoch.set_postfix(**print_dict)
        if return_avg_accuracy:
            return losses, avg_accuracy
        return losses

    def flow_bsp_loss(self, loss, losses, network, occupancy_ground_truth, prediction, convex_layer_weights):

        bsp_thershold = self.config.bsp_thershold if hasattr(self.config, 'bsp_thershold') else 0.01
        if self.config.bsp_phase == 0:
            concave_layer_weights = network.decoder.bsp_field.concave_layer_weights if torch.cuda.device_count() <= 1 else network.module.decoder.bsp_field.concave_layer_weights
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

    @staticmethod
    def process_state_dict(network_state_dict, type = 0):

        if torch.cuda.device_count() >= 2 and type == 0:
            for key, item in list(network_state_dict.items()):
                if key[:7] != 'module.':
                    new_key = 'module.' + key
                    network_state_dict[new_key] = item
                    del network_state_dict[key]
        else:
            for key, item in list(network_state_dict.items()):
                if key[:7] == 'module.':
                    new_key = key[7:]
                    network_state_dict[new_key] = item
                    del network_state_dict[key]

        return network_state_dict


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


    model_type = f"AutoEncoder-{config.encoder_type}-{config.decoder_type}" if config.network_type == 'AutoEncoder' else f"AutoDecoder-{config.decoder_type}"
    debugger = MyDebugger(
        f'IM-Net-Training-experiment-{os.path.basename(config.data_folder)}-{model_type}{config.special_symbol}', 
        is_save_print_to_file = True, 
        config_path = resume_path,
        config=config
    )
    trainer = Trainer(config = config, debugger = debugger)
    trainer.train_network()
