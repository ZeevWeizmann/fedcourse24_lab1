import time
import random

from abc import ABC, abstractmethod

import numpy as np
import torch

from utils.torch_utils import *

from tqdm import tqdm


class Aggregator(ABC):
    def __init__(
            self,
            clients,
            clients_weights,
            global_learner,
            logger,
            verbose=0,
            seed=None
    ):
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.clients = clients
        self.n_clients = len(clients)

        self.clients_weights = clients_weights

        self.global_learner = global_learner
        self.device = self.global_learner.device

        self.verbose = verbose
        self.logger = logger

        self.model_dim = self.global_learner.model_dim

        self.c_round = 0

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def write_logs(self):
        global_train_loss = 0.
        global_train_metric = 0.
        global_test_loss = 0.
        global_test_metric = 0.

        for client_id, client in enumerate(self.clients):

            train_loss, train_metric, test_loss, test_metric = client.write_logs(counter=self.c_round)

            if self.verbose > 1:
                tqdm.write("*" * 30)
                tqdm.write(f"Client {client_id}..")
                tqdm.write(f"Train Loss: {train_loss:.3f} | Train Metric: {train_metric :.3f}|", end="")
                tqdm.write(f"Test Loss: {test_loss:.3f} | Test Metric: {test_metric:.3f} |")
                tqdm.write("*" * 30)

            global_train_loss += self.clients_weights[client_id] * train_loss
            global_train_metric += self.clients_weights[client_id] * train_metric
            global_test_loss += self.clients_weights[client_id] * test_loss
            global_test_metric += self.clients_weights[client_id] * test_metric

        if self.verbose > 0:
            tqdm.write("+" * 50)
            tqdm.write(f"Global | Round {self.c_round}..")
            tqdm.write(f"Train Loss: {global_train_loss:.3f} | Train Metric: {global_train_metric:.3f} |", end="")
            tqdm.write(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_metric:.3f} |")
            tqdm.write("+" * 50)

        self.logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
        self.logger.add_scalar("Train/Metric", global_train_metric, self.c_round)
        self.logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
        self.logger.add_scalar("Test/Metric", global_test_metric, self.c_round)
        self.logger.flush()


class NoCommunicationAggregator(Aggregator):
    def mix(self):
        for idx in range(self.n_clients):
            self.clients[idx].step()

        self.c_round += 1

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
    def mix(self):
        clients_weights = torch.tensor(self.clients_weights, dtype=torch.float32)

        print(f"[DEBUG][Aggregator] Start round {self.c_round}, clients = {self.n_clients}")

        for idx in range(self.n_clients):
            print(f"[DEBUG][Aggregator] Client {idx} step() start")
            self.clients[idx].step()
            print(f"[DEBUG][Aggregator] Client {idx} step() done")

        client_params = []
        for idx in range(self.n_clients):
            params = self.clients[idx].learner.get_param_tensor()
            if params is None:
                print(f"[ERROR][Aggregator] Client {idx} returned None params!")
            else:
                print(f"[DEBUG][Aggregator] Client {idx} params shape = {params.shape}")
                client_params.append(params)

        if len(client_params) == 0:
            print("[ERROR][Aggregator] All clients returned None!")
            return

        avg_params = torch.zeros_like(client_params[0])
        for idx in range(self.n_clients):
            avg_params += clients_weights[idx] * client_params[idx]

        print(f"[DEBUG][Aggregator] Averaged params shape = {avg_params.shape}")

        self.global_learner.set_param_tensor(avg_params)
        print("[DEBUG][Aggregator] Global model updated")

        self.update_clients()

        self.c_round += 1

    def update_clients(self):
        global_params = self.global_learner.get_param_tensor()
        print(f"[DEBUG][Aggregator] update_clients: global_params shape = {global_params.shape}")
        for idx, client in enumerate(self.clients):
            client.learner.set_param_tensor(global_params)
            print(f"[DEBUG][Aggregator] Client {idx} received global params")
