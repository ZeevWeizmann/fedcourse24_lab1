from models import *
from learner import *

from datasets.mnist import *

from client import *

from aggregator import *

from .optim import *
from .metrics import *
from .constants import *

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

import torch.nn as nn
from torchvision import datasets, transforms, models
from datasets.cifar10 import NPYDataset
from torchvision.models import MobileNet_V2_Weights


def experiment_not_implemented_message(experiment_name):
    error = f"{experiment_name} is not available! " \
            f"Possible are: 'mnist', 'cifar10'."

    return error


def get_data_dir(experiment_name):
    """
    Returns the path where to find/store the dataset for the experiment.
    """
    if experiment_name == "cifar10":
        data_dir = os.path.join("data", "cifar10", "all_data")
    else:
        data_dir = os.path.join("data", experiment_name, "all_data")
    return data_dir


def get_model(experiment_name, device):
    """
    constructs model for an experiment
    Parameters
    ----------
    experiment_name: str
        name of the experiment to be used;
        possible are {"mnist", "cifar10"}
    device: str
        used device; possible `cpu` and `cuda`
    Returns
    -------
        nn.Module
    """


    if experiment_name == "mnist":
        model = LinearLayer(input_dim=784, output_dim=10, bias=True)
    elif experiment_name == "cifar10":
        # MobileNet v2  CIFAR10
        # model = models.mobilenet_v2(weights=None) #Train from scratch
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT) #Download pretrained weights
        # Freeze feature extractor
        for param in model.features.parameters():
            param.requires_grad = False   
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 10) # Change the output layer to match CIFAR-10 classes
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )
    print(f"[DEBUG][get_model] Number of params: {sum(p.numel() for p in model.parameters())}")
    print(f"[DEBUG] Built model for {experiment_name}: {model.__class__.__name__}")
    print(f"[DEBUG] Number of parameters: {sum(p.numel() for p in model.parameters())}")

    model = model.to(device)

    return model


def get_learner(experiment_name, device, optimizer_name, lr, seed):
    """
    constructs learner for an experiment for a given seed

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be used;
        possible are {"mnist", "cifar10"}

    device: str
        used device; possible `cpu` and `cuda`

    optimizer_name: str

    lr: float
        learning rate

    seed: int

    Returns
    -------
        Learner

    """
    torch.manual_seed(seed)

    if experiment_name in ["mnist", "cifar10"]:
        criterion = nn.CrossEntropyLoss().to(device)
        metric = accuracy
        is_binary_classification = False
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = \
        get_model(experiment_name=experiment_name, device=device)

    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr=lr,
        )

    return Learner(
        model=model,
        criterion=criterion,
        metric=metric,
        device=device,
        optimizer=optimizer,
        is_binary_classification=is_binary_classification
    )


def get_loader(experiment_name, client_data_path, batch_size, train):
    """

    Parameters
    ----------
    experiment_name: str

    client_data_path: str

    batch_size: int

    train: bool

    Returns
    -------
        * torch.utils.data.DataLoader

    """

    if experiment_name == "mnist":
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        dataset = MNIST(root=client_data_path, train=train, transform=transform)
    elif experiment_name == "cifar10":
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
        dataset = NPYDataset(data_path=client_data_path, train=train, transform=transform)
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def init_client(args, client_id, client_dir, logger):
    """initialize one client


    Parameters
    ----------
    args:

    client_id: int

    client_dir: str

    logger:

    Returns
    -------
        * Client

    """
    train_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=client_dir,
        batch_size=args.bz,
        train=True,
    )

    val_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=client_dir,
        batch_size=args.bz,
        train=False,
    )

    test_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=client_dir,
        batch_size=args.bz,
        train=False,
    )

    learner = \
        get_learner(
            experiment_name=args.experiment,
            device=args.device,
            optimizer_name=args.local_optimizer,
            lr=args.local_lr,
            seed=args.seed
        )

    client = Client(
        client_id=client_id,
        learner=learner,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        local_steps=args.local_steps,
        logger=logger
    )
    print(f"[DEBUG] Initializing client {client_id}")
    print(f"[DEBUG] Learner model: {learner.model.__class__.__name__}")
    print(f"[DEBUG] Number of params: {sum(p.numel() for p in learner.model.parameters())}")
    return client


def get_aggregator(aggregator_type, clients, clients_weights, global_learner, logger, verbose, seed):
    """
    Parameters
    ----------
    aggregator_type: str
        possible are {"centralized", "no_communication"}

    clients: Dict[int: Client]

    clients_weights: Dict[int: Client]

    global_learner: Learner

    logger: torch.utils.tensorboard.SummaryWriter

    verbose: int

    seed: int


    Returns
    -------
        * Aggregator
    """
    if aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            clients_weights=clients_weights,
            global_learner=global_learner,
            logger=logger,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            clients_weights=clients_weights,
            global_learner=global_learner,
            logger=logger,
            verbose=verbose,
            seed=seed,
        )
    else:
        error_message = f"{aggregator_type} is not a possible aggregator type, possible are: "
        for type_ in AGGREGATOR_TYPES:
            error_message += f" {type_}"


def get_clients_weights(clients, objective_type):
    """Compute the weights to be associated with every client.

    If objective_type is "average", clients receive the same weight.
    If objective_type is "weighted", clients receive weight proportional to the number of samples.

    Parameters
    ----------
    clients: List[Client]
    objective_type: str
        Type of the objective function; possible are: {"average", "weighted"}

    Returns
    -------
    clients_weights: List[float]
    """
    n_clients = len(clients)
    clients_weights = []

    total_num_samples = 0
    for client in clients:
        total_num_samples += client.num_samples

    for client in clients:

        if objective_type == "average":
            weight = 1 / n_clients

        elif objective_type == "weighted":
            weight = client.num_samples / total_num_samples

        else:
            raise NotImplementedError(
                f"{objective_type} is not an available objective type. Possible are 'average' and 'weighted'.")

        clients_weights.append(weight)

    return clients_weights
