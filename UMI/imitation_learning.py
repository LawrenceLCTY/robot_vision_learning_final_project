"""
Training a save a policy network and a RGBD network based on collected trajectories
"""
from neural_networks import PolicyNetwork, RGBDNetwork
from dataCollector import get_imitation_training_data
from data import append_values_to_file
from variables import ACTIONS, LOSS_FILENAME, RGBD_WEIGHT_FILENAME, POLICY_WEIGHT_FILENAME

import os
import torch
import torch.nn as nn
import numpy as np

from torch.optim.adam import Adam
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def imitation_learning(pNet: PolicyNetwork, pNet_opt, rgbdNet: RGBDNetwork, rgbNet_opt, loss_criterion, epoch_num=500,
                       folder_path="./IL_training", batch_sze=64, interval=1):
    """
    just use MSE for loss, try to generate policy as close to the collected trajectories as possible
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for epoch in tqdm(range(epoch_num), desc="imitiation learning"):
        # collect_data(env, total_trajectory_num=200)
        success, batch = get_imitation_training_data("./data/", batch_size=batch_size, interval=interval)

        gt_action_inputs_tensor = torch.tensor(np.array([np.array(transition[ACTIONS]) for transition in batch])
                                               .astype(np.float32), device=device)

        # rgbd net processed batch transitions into inputs for policy network
        batch_policy_inputs = rgbdNet.process_inputs(batch)



        # pass processed batch inputs to pNet to generate the actions
        output_actions_tensor, log_probs = pNet.sample_or_likelihood(batch_policy_inputs)

        # get the loss value
        mse_loss = loss_criterion(gt_action_inputs_tensor, output_actions_tensor)

        # backpropagation
        pNet_opt.zero_grad()
        rgbNet_opt.zero_grad()

        mse_loss.backward()

        pNet_opt.step()
        rgbNet_opt.step()

        # save the loss for observation
        append_values_to_file(mse_loss.item(), os.path.join(folder_path, LOSS_FILENAME))

    rgbdNet.save_weights(os.path.join(folder_path, RGBD_WEIGHT_FILENAME))
    pNet.save_weights(os.path.join(folder_path, POLICY_WEIGHT_FILENAME))


if __name__ == "__main__":

    # define variables
    input_height = 224
    input_width = 600
    output_dim = 512
    action_dim = 12
    epoch_num = 1

    action_bounds = [-1, 1]

    lr = 3e-4
    batch_size = 64
    interval = 1


    # Create MSE loss function
    criterion = nn.MSELoss()

    rgbd_network = RGBDNetwork(input_height, input_width, output_dim)
    policy_network = PolicyNetwork(n_states=output_dim, n_actions=action_dim, action_bounds=action_bounds)

    # initialise optimizers
    policy_opt = Adam(policy_network.parameters(), lr=lr)
    rgbd_opt = Adam(policy_network.parameters(), lr=lr)

    # start learning
    imitation_learning(policy_network, policy_opt, rgbd_network, rgbd_opt, criterion, epoch_num,
                       batch_sze=batch_size, interval=interval, folder_path="IL_training")


