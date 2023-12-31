"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLP_policy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            # the order of input and ouput are different than the build_mlp class
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)


        # add support of log_net
        self.log_net = build_mlp(
            # the order of input and ouput are different than the build_mlp class
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.log_net.to(ptu.device)

        # original logstd parameter
        self.logstd = nn.Parameter(
            torch.randn(self.ac_dim, dtype=torch.float32, device=ptu.device),
            requires_grad=True
        )
        self.logstd.to(ptu.device)

        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        # actions = self.mean_net(observation)

        # TODO: can change to distribution analysis

        mean = self.mean_net(observation)
        # mean = mean.squeeze(1)  # Assuming you still want to squeeze the singleton dimension
        # log_std = self.log_net(observation)
        # Create a normal distribution with the mean and learned log standard deviation
        # action_distribution = distributions.Normal(mean, torch.exp(logstd))
        action_distribution = distributions.Normal(mean, self.logstd.exp())
        # action_distribution = distributions.Normal(mean, log_std.exp())
        # Sample an action from the distribution
        # action = action_distribution.sample()
        # # breakpoint()
        # result = action_distribution.log_prob(action)
        # breakpoint()
        # print(result)
        return action_distribution

        raise NotImplementedError

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss

        # criterion = nn.MSELoss(reduction='mean')
        self.optimizer.zero_grad() 
        observations = observations.to(ptu.device)
        actions = actions.to(ptu.device)
        # dataset = TensorDataset(observations, actions)
        # loader = DataLoader(dataset, batch_size=64, shuffle=True) # gives you data in batches
        # epoch_loss = 0
        # # for curr_states, curr_actions in loader:
        #     action_distribution = self.forward(curr_states)
        #     loss = -action_distribution.log_prob(curr_actions).sum()
        #     loss.backward()
        #     self.optimizer.step()
        #     epoch_loss += loss.detach().cpu().numpy().squeeze()

        action_distribution = self.forward(observations)
        # put minus sign for minimizing the loss as the minimum negative log likelihood
        loss = -action_distribution.log_prob(actions).sum()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach().cpu().numpy().squeeze()
        return {
            # You can add extra logging information here, but keep this line
            # 'Training Loss': ptu.to_numpy(loss),
            'Training Loss': loss,
        }
    
    def get_action(self, states):
        return self.forward(states).sample()
