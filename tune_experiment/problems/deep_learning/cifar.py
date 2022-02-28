from tune_experiment.problems.problem import Problem
from typing import Callable, Dict
from ray import tune
from ray.tune.sample import Domain
from ray.tune.utils.placement_groups import PlacementGroupFactory

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from filelock import FileLock
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

import ray.train as train
from ray import tune
from ray.train import Trainer
from ray.util.sgd.torch.resnet import ResNet18


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) // train.world_size()
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset) // train.world_size()
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    return {"loss": test_loss}


def train_func(config):
    epochs = config.pop("epochs", 25)
    checkpoint = train.load_checkpoint() or {}
    model = checkpoint.get("model", ResNet18(config))
    model = train.torch.prepare_model(model)

    # Create optimizer.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.1),
        momentum=1-config.get("1_minus_momentum", 0.1),
        weight_decay=config.get("weight_decay", 0.001),
    )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load in training and validation data.
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )  # meanstd transformation

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    with FileLock(".ray.lock"):
        train_dataset = CIFAR10(
            root="~/data", train=True, download=True, transform=transform_train
        )
        validation_dataset = CIFAR10(
            root="~/data", train=False, download=False, transform=transform_test
        )

    if config.get("test_mode"):
        train_dataset = Subset(train_dataset, list(range(64)))
        validation_dataset = Subset(validation_dataset, list(range(64)))

    worker_batch_size = 2**config["batch_size_pow"] // train.world_size()

    train_loader = DataLoader(train_dataset, batch_size=worker_batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=worker_batch_size)

    train_loader = train.torch.prepare_data_loader(train_loader)
    validation_loader = train.torch.prepare_data_loader(validation_loader)

    # Create loss.
    criterion = nn.CrossEntropyLoss()

    if "criterion" in checkpoint:
        criterion.load_state_dict(checkpoint["criterion"])

    results = []

    start_epoch = checkpoint.get("epoch", -1) + 1

    for epoch in range(start_epoch, epochs):
        train_epoch(train_loader, model, criterion, optimizer)
        result = validate_epoch(validation_loader, model, criterion)
        train.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer.state_dict(),
            criterion=criterion.state_dict(),
        )
        train.report(**result)
        results.append(result)

    return results

class CIFARProblem(Problem):
    def __init__(self, n_gpus: int):
        self.n_gpus = n_gpus

    @property
    def config(self) -> Dict[str, Domain]:
        return {
            "lr": tune.loguniform(1e-4, 1e-2),
            "weight_decay": tune.uniform(0.0001, 0.005),
            "1_minus_momentum": tune.loguniform(0.003, 0.1),
            "batch_size_pow": tune.randint(6, 8),
            "epochs": 25
        }

    @property
    def init_config(self) -> Dict[str, Domain]:
        return {
            "batch_size_pow": 7,
            "lr": 0.01
        }

    @property
    def early_stopping_key(self) -> str:
        return "training_iteration"

    def get_trainable(self) -> Callable:
        trainer = Trainer("torch", num_workers=self.n_gpus, use_gpu=True)
        return trainer.to_tune_trainable(train_func)

    @property
    def resource_requirements(self) -> PlacementGroupFactory:
        return None

    @property
    def metric_name(self) -> str:
        return "loss"

    @property
    def metric_mode(self) -> str:
        return "min"

    def trainable_with_parameters(self):
        return tune.with_parameters(self.get_trainable())