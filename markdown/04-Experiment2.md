::: {.cell .markdown}
## Experiment 2:
 
In this experiment, we compare two methods of weight initialization: warm-starting and random initialization, for two models: **ResNet18** and **3-layer MLP** with tanh activation. We also compare two optimizers: **SGD** and **Adam**, for updating the weights based on the gradients. We use three image classification datasets:  **CIFAR-10**, **CIFAR-100** and **SVHN**, and report the test accuracy of each model on each dataset. All models are trained using a mini-batch size of 128 and a learning rate of 0.001.

We use the same components as in experiment one: the `get_loaders` function to get the required dataset's train and test loaders, the [**ResNet18**](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) model from `torchvision.models`, and the [**SGD**](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) optimizer from `torch.optim`. We also introduce some new components:

1. The **CIFAR-100** dataset, which has 60,000 color images of 100 classes. To get the CIFAR-100 dataset, we pass the string `cifar100` as the dataset name argument to the `get_loaders` function.
2. The **SVHN** dataset, which has 73,257 color images of 10 classes of street view house numbers. To get the SVHN dataset, we pass the string `svhn` as the dataset name argument to the `get_loaders` function.
3. The `MLP` class that defines a 3-layer MLP model with a tanh activation function and a bias term.
4. The `torch.optim.Adam` class that implements the [**Adam**](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer, which is an adaptive learning rate method.
5. The `train_to_threshold` function that trains the model until it reaches 99% training accuracy, following the original paper.
:::

::: {.cell .markdown}
***
We import the required packages as before.
:::

::: {.cell .code}
``` python
import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms, datasets, models
```
:::

::: {.cell .markdown}
***
This is the same `get_loaders` function from Experiment 1
:::

::: {.cell .code}
``` python
def get_loaders(dataset="cifar10", use_half_train=False, batch_size=128, dataset_portion=None):
    """
    This loads the whole CIFAR-10 into memory and returns train and test data according to params
    @param use_half_train (bool): return half the data or the whole train data
    @param batch_size (int): batch size for training and testing
    @param dataset_portion (double): portion of train data

    @returns dict() with train and test data loaders with keys `train_loader`, `test_loader`
    """
    
    # Normalization using channel means
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Creating transform function
    train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
        
    # Test transformation function    
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    
    # Check which dataset is required and load data from torchvision datasets
    if dataset == 'cifar10':
        original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=True, transform=train_transform, download=True)
        original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        original_train_dataset = datasets.CIFAR100(root=os.path.join('data', 'cifar100_data'),
                                             train=True, transform=train_transform, download=True)
        original_test_dataset = datasets.CIFAR100(root=os.path.join('data', 'cifar100_data'),
                                             train=False, transform=test_transform, download=True)
    else:
        original_train_dataset = datasets.SVHN(root=os.path.join('data', 'svhn_data'),
                                             split='train', transform=train_transform, download=True)
        original_test_dataset = datasets.SVHN(root=os.path.join('data', 'svhn_data'),
                                             split='test', transform=test_transform, download=True)
    
    # Check half data flag
    if use_half_train:
        print('Using Half Data')
        dataset_portion = 0.5
        
    # Check if only a portion is required
    if dataset_portion:
        dataset_size = len(original_train_dataset)
        split = int(np.floor((1 - dataset_portion) * dataset_size))
        original_train_dataset, _ = random_split(original_train_dataset, [dataset_size - split, split])
    
    # Creating data loaders
    loader_args = {
        "batch_size": batch_size,
    }

    train_loader = torch.utils.data.DataLoader(
        dataset=original_train_dataset,
        shuffle=True,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loader": train_loader,
            "test_loader": test_loader}
```
:::


::: {.cell .markdown}
***

The following is a class for a **multilayer perceptron** (MLP) model with several fully connected (fc) layers and a final fully connected layer for the logits output. The arguments are:

- `input_dim`: the input feature dimension.
- `num_classes`: the output class number.
- `hidden_units`: the hidden unit number for each fc layer we set the default as 100 dimension as mentioned in the appendix.
- `activation`: the activation function, either `tanh` or `relu`.
- `bias`: whether to use bias terms in the fc layers.

The function returns an MLP model object that can be trained or tested. The forward method takes an input tensor x and returns an output tensor x with the logits values. The output tensor does not have a final activation function. This will be used to create the **3-layer MLP** model.
:::

::: {.cell .code}
``` python
# Define a class for a multilayer perceptron (MLP) model
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=10, hidden_units=[100, 100, 100], activation='tanh', bias=True):
        super().__init__()

        # Check that the activation argument is valid
        assert activation in ['tanh', 'relu'], "Activation must be tanh or relu"

        # Assign the activation function based on the argument
        if activation == 'tanh':
            self.activation_function = torch.tanh
        if activation == 'relu':
            self.activation_function = torch.relu
        
        # Store num_classes and input_dim to be used in forward function
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Initialize a variable to keep track of the last dimension of the layers
        last_dim = input_dim
        
        # Initialize an empty list to store the fully connected (fc) layers
        self.fcs = []
        
        # Loop through the hidden units argument and create fc layers with the given dimensions and bias
        for i, n_h in enumerate(hidden_units):
            self.fcs.append(nn.Linear(last_dim, n_h, bias=bias))
            # Register the fc layer as a submodule with a name
            self.add_module(f"hidden_layer_{i}", self.fcs[-1])
            # Update the last dimension to match the output dimension of the fc layer
            last_dim = n_h
            
        # Create a final fc layer for the logits output with the number of classes and bias
        self.logit_fc = nn.Linear(last_dim, self.num_classes, bias=bias)

        
    def forward(self, x):
        # Reshape the input x to have a batch size and an input dimension
        x = x.view(-1, self.input_dim)
        
        # Loop through the fc layers and apply them to x with the activation function
        for fc in self.fcs:
            x = fc(x)
            x = self.activation_function(x)
            
        # Apply the final fc layer to x and return it as the output
        x = self.logit_fc(x)
        
        # x is returned without adding the final activation
        return x
```
:::

::: {.cell .markdown}
***

The following cell defines two functions that perform one epoch of training or evaluation on a data loader. They return the average loss and accuracy of the model. These functions will be used in the training function to run epoch by epoch.
:::

::: {.cell .code}
``` python
# Function takes predictions and true values to return accuracies
def get_accuracy(logit, true_y):
    pred_y = torch.argmax(logit, dim=1)
    return (pred_y == true_y).float().mean()

def eval_on_dataloader(device, criterion, model, dataloader):
    """
    Evaluate the model on a given data loader and return the average loss and accuracy.

    Parameters:
    device: the device (cpu or gpu) to use for computation
    criterion: the loss function to use
    model: the model to evaluate
    dataloader: the data loader to iterate over the data

    Returns:
    loss: the average loss over the data loader
    accuracy: the average accuracy over the data loader
    """
    # Lists to store accuracy and loss
    accuracies = []
    losses = []
    
    for batch_idx, (data_x, data_y) in enumerate(dataloader): 
        data_x = data_x.to(device) 
        data_y = data_y.to(device)
        
        # get the model output for the input data
        model_y = model(data_x) 
        
        # compute the loss and accuracy
        loss = criterion(model_y, data_y)
        batch_accuracy = get_accuracy(model_y, data_y)
        
        # append accuracy and loss to lists
        accuracies.append(batch_accuracy.item()) 
        losses.append(loss.item())

    # compute average loss and accuracy
    loss = np.mean(losses) 
    accuracy = np.mean(accuracies) 
    return loss, accuracy 


def train_one_epoch(device, model, optimizer, criterion, dataloader):
    """
    Train the model for one epoch on a given training data loader and return the average loss and accuracy.

    Parameters:
    device: the device (cpu or gpu) to use for computation
    model: the model to train
    optimizer: the optimizer to use for updating the weights
    criterion: the loss function to use
    train_dataloader: the training data loader to iterate over the training data

    Returns:
    train_loss: the average loss over the training data loader
    train_accuracy: the average accuracy over the training data loader
    """
    # Lists to store accuracy and loss
    accuracies = []
    losses = [] 
    
    for batch_idx, (data_x, data_y) in enumerate(dataloader):
        data_x = data_x.to(device) 
        data_y = data_y.to(device) 
        
         # reset the gradients of the optimizer
        optimizer.zero_grad()
        
        # get the model output for the input data
        model_y = model(data_x)
        
        # compute the loss and accuracy
        loss = criterion(model_y, data_y)
        batch_accuracy = get_accuracy(model_y, data_y)
        
        # compute the gradients and update model parameters
        loss.backward()
        optimizer.step()

        # append accuracy and loss to lists
        accuracies.append(batch_accuracy.item()) 
        losses.append(loss.item())

    # compute average loss and accuracy
    loss = np.mean(losses) 
    accuracy = np.mean(accuracies) 
    return loss, accuracy 

```
:::

::: {.cell .markdown}
***

The `train_to_threshold` function is the same as `train_model_exp1` functions except that it trains the model until a certain `training_threshold` or until the change in the training accuracy doesn't exceed `convergence_change_threshold` for certain number of epochs. The new parameter introduced are:

- `train_threshold`: The training accuracy at which the model stops training.
- `convergence_change_threshold`: The minimum accuracy change to continue training.
- `convergence_epochs`: The maximum number of epochs allowed with insufficient change in training accuracy before stopping the training.
:::

::: {.cell .code}
``` python
def train_to_threshold(title='warm', lr=0.001, checkpoint=None, use_half_data=False, convergence_epochs=4,
                       train_threshold=0.5, convergence_change_threshold=0.002, random_seed=42):
    # use gpu if available ( change device id if needed )
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Get the dataset
    loaders = get_loaders(dataset="cifar10", use_half_train=use_half_data)

    # Get the model
    model = models.resnet18(num_classes=10).to(device)

    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Get model from checkpoint
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device)['model'])

    print(f"Training {'warm-starting' if checkpoint is not None else 'random initialized'} " \
        f"ResNet-18 model with SGD optimizer on {'50%' if use_half_data else '100%'} of cifar10 dataset")

    # initialize training varaibles
    train_accuracies = []
    stop_indicator = False
    model_name = 'resnet18'
    epoch = 0
    # Train until convergence or stop condition is met
    while(not stop_indicator):
        if epoch % 5 == 0:
            print(f"\t Training in epoch {epoch + 1} \t")
        # Train for one epoch and get loss and accuracy
        train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion, loaders['train_loader'])

        # Append training accuracy to list
        train_accuracies.append(train_accuracy)
        epoch += 1
        # Check if training accuracy is above a threshold
        if train_accuracy >= train_threshold:
            print(f"Convergence codition met. Training accuracy > {train_threshold}")
            stop_indicator = True

        # Check if training accuracy has stopped improving for a number of epochs
        if len(train_accuracies) >= convergence_epochs:
            if np.std(train_accuracies[-convergence_epochs:]) < convergence_change_threshold:
                print(f"\tConvergence codition met. Training accuracy = {train_accuracy} stopped improving")
                stop_indicator = True

    # Evaluate on test set and get loss and accuracy
    test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
    print(f"\tTest accuracy = {test_accuracy}")

    # Save the model if will be used for warm-starting
    if use_half_data:
        # Create directory in exp with experiment title
        experiment_dir = os.path.join('experiments/exp2', title)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # save the model
        model_name = model_name+'-sgd'
        model_directory =  os.path.join(experiment_dir, f'{model_name}.pt')
        torch.save({
            'model': model.state_dict()
        }, model_directory)

        print(f"Model saved to checkpoint: {model_directory}")

    return test_accuracy * 100 
```
:::

::: {.cell .markdown}
***

The ResNet-18 model is trained on the CIFAR-10 dataset using an SGD optimizer and the `train_to_threshold` function. It is trained with warm-start and random initialization.
:::

::: {.cell .code}
``` python
# Dictionary to save all results
overal_result = {}

# train on full data with random initialization
random_init = train_to_threshold(title='resnet-sgd-cifar10', train_threshold=0.99)
```
:::


::: {.cell .code}
``` python
# train on half data
_ = train_to_threshold(title='resnet-sgd-cifar10', train_threshold=0.99, use_half_data=True)

# train on full data with warm-starting
warm_start = train_to_threshold(title='resnet-sgd-cifar10', train_threshold=0.99,
                                     checkpoint='experiments/exp2/resnet-sgd-cifar10/resnet18-sgd.pt')
```
:::

::: {.cell .code}
``` python
# get the difference between random and warm-start models using sgd optimizer on CIFAR-10
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['resnet-sgd-cifar10'] = [random_init, warm_start, diff]
```
:::


::: {.cell .markdown}
***

We extend this experiment by training the same model with the Adam optimizer instead of SGD. We add a new parameter `optimizer_name` to select the optimizer for the model.
:::

::: {.cell .code}
``` python
def train_to_threshold(title='warm', lr=0.001, checkpoint=None, 
                       use_half_data=False, optimizer_name='adam', convergence_epochs=4,
                       train_threshold=0.5, convergence_change_threshold=0.002, random_seed=42):
    # use gpu if available ( change device id if needed )
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Get the dataset
    loaders = get_loaders(dataset="cifar10", use_half_train=use_half_data)

    # Get the model
    model = models.resnet18(num_classes=10).to(device)

    # Create the optimizer
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Get model from checkpoint
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device)['model'])

    print(f"Training {'warm-starting' if checkpoint is not None else 'random initialized'} ResNet-18 model " \
            f"with {optimizer_name.upper()} optimizer on {'50%' if use_half_data else '100%'} of cifar10 dataset")

    # initialize training varaibles
    train_accuracies = []
    stop_indicator = False
    model_name = 'resnet18'
    epoch = 0
    # Train until convergence or stop condition is met
    while(not stop_indicator):
        if epoch % 5 == 0:
            print(f"\t Training in epoch {epoch + 1} \t")
        # Train for one epoch and get loss and accuracy
        train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion, loaders['train_loader'])

        # Append training accuracy to list
        train_accuracies.append(train_accuracy)
        epoch += 1
        # Check if training accuracy is above a threshold
        if train_accuracy >= train_threshold:
            print(f"Convergence codition met. Training accuracy > {train_threshold}")
            stop_indicator = True

        # Check if training accuracy has stopped improving for a number of epochs
        if len(train_accuracies) >= convergence_epochs:
            if np.std(train_accuracies[-convergence_epochs:]) < convergence_change_threshold:
                print(f"\tConvergence codition met. Training accuracy = {train_accuracy} stopped improving")
                stop_indicator = True

    # Evaluate on test set and get loss and accuracy
    test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
    print(f"\tTest accuracy = {test_accuracy}")

    # Save the model if will be used for warm-starting
    if use_half_data:
        # Create directory in exp with experiment title
        experiment_dir = os.path.join('experiments/exp2', title)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # save the model
        model_name = model_name+'-'+optimizer_name
        model_directory =  os.path.join(experiment_dir, f'{model_name}.pt')
        torch.save({
            'model': model.state_dict()
        }, model_directory)

        print(f"Model saved to checkpoint: {model_directory}")

    return test_accuracy * 100
```
:::

::: {.cell .markdown}
***

We repeat the training of the same models with the Adam optimizer instead of SGD.
:::

::: {.cell .code}
``` python
# train on full data with random initialization but with Adam
random_init = train_to_threshold(title='resnet-adam-cifar10', train_threshold=0.99, optimizer_name='adam')

# train on half data
_ = train_to_threshold(title='resnet-adam-cifar10', train_threshold=0.99,
                                             optimizer_name='adam', use_half_data=True)

# train on full data with warm-starting but with Adam
warm_start = train_to_threshold(title='resnet-adam-cifar10', train_threshold=0.99, optimizer_name='adam',
                                        checkpoint='experiments/exp2/resnet-adam-cifar10/resnet18-adam.pt')

# get the difference between random and warm-start models using Adam optimizer on CIFAR-10
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['resnet-adam-cifar10'] = [random_init, warm_start, diff]
```
:::

::: {.cell .markdown}
***

We also experiment with the MLP model instead of the ResNet. We introduce a parameter `model_name` to select the model for the training.
:::

::: {.cell .code}
``` python
def train_to_threshold(title='warm', lr=0.001, checkpoint=None, use_half_data=False,
                        optimizer_name='adam', model_name='resnet18', convergence_epochs=4,
                        train_threshold=0.5, convergence_change_threshold=0.002, random_seed=42):
    # use gpu if available ( change device id if needed )
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Get the dataset
    loaders = get_loaders(dataset="cifar10", use_half_train=use_half_data)

    # Get the model
    if model_name == 'resnet18':
        model = models.resnet18(num_classes=10).to(device)
    else:
        model = MLP( input_dim = 32 * 32 * 3, num_classes=10).to(device)

    # Create the optimizer
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Get model from checkpoint
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device)['model'])

    print(f"Training {'warm-starting' if checkpoint is not None else 'random initialized'} {model_name} model " \
            f"with {optimizer_name.upper()} optimizer on {'50%' if use_half_data else '100%'} of cifar10 dataset")

    # initialize training varaibles
    train_accuracies = []
    stop_indicator = False
    epoch = 0
    # Train until convergence or stop condition is met
    while(not stop_indicator):
        if epoch % 5 == 0:
            print(f"\t Training in epoch {epoch + 1} \t")
        # Train for one epoch and get loss and accuracy
        train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion, loaders['train_loader'])

        # Append training accuracy to list
        train_accuracies.append(train_accuracy)
        epoch += 1
        # Check if training accuracy is above a threshold
        if train_accuracy >= train_threshold:
            print(f"Convergence codition met. Training accuracy > {train_threshold}")
            stop_indicator = True

        # Check if training accuracy has stopped improving for a number of epochs
        if len(train_accuracies) >= convergence_epochs:
            if np.std(train_accuracies[-convergence_epochs:]) < convergence_change_threshold:
                print(f"\tConvergence codition met. Training accuracy = {train_accuracy} stopped improving")
                stop_indicator = True

    # Evaluate on test set and get loss and accuracy
    test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
    print(f"\tTest accuracy = {test_accuracy}")

    # Save model if it is needed for warm-starting
    if use_half_data:
        # Create directory in exp with experiment title
        experiment_dir = os.path.join('experiments/exp2', title)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # save the model
        model_name = model_name+'-'+optimizer_name
        model_directory =  os.path.join(experiment_dir, f'{model_name}.pt')
        torch.save({
            'model': model.state_dict()
        }, model_directory)

        print(f"Model saved to checkpoint: {model_directory}")

    return test_accuracy * 100
```
:::


::: {.cell .markdown}
***

We use warm-starting and the SGD optimizer to train the MLP model in the next cell.
:::

::: {.cell .code}
``` python
# train MLP model on full data with random initialization and SGD optimizer
random_init = train_to_threshold(title='mlp-sgd-cifar10', train_threshold=0.99, 
                                 optimizer_name='sgd', model_name='mlp')

# train MLP model on half data
_ = train_to_threshold(title='mlp-sgd-cifar10', train_threshold=0.99, 
                       optimizer_name='sgd', model_name='mlp', use_half_data=True)

# train MLP on full data with warm-starting and SGD optimizer
warm_start = train_to_threshold(title='mlp-sgd-cifar10', train_threshold=0.99, optimizer_name='sgd', 
                                model_name='mlp', checkpoint='experiments/exp2/mlp-sgd-cifar10/mlp-sgd.pt')

# get the difference between random and warm-start MLP models using SGD optimizer on CIFAR-10
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['mlp-sgd-cifar10'] = [random_init, warm_start, diff]
```
:::

::: {.cell .markdown}
***

In the next cell, we train the MLP model with warm-starting and the Adam optimizer.
:::

::: {.cell .code}
``` python
# train MLP model on full data with random initialization
random_init= train_to_threshold(title='mlp-adam-cifar10', train_threshold=0.99, 
                                optimizer_name='adam', model_name='mlp')

# train MLP mode on half data
_ = train_to_threshold(title='mlp-adam-cifar10', train_threshold=0.99, 
                       optimizer_name='adam', model_name='mlp', use_half_data=True)

# train MLP on full data with warm-starting
warm_start = train_to_threshold(title='mlp-adam-cifar10', train_threshold=0.99, optimizer_name='adam',
                                model_name='mlp', checkpoint='experiments/exp2/mlp-adam-cifar10/mlp-adam.pt')

# get the difference between random and warm-start MLP models using Adam optimizer on CIFAR-10
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['mlp-adam-cifar10'] = [random_init, warm_start, diff]
```
:::

::: {.cell .markdown}
***

Finaly we extend this to allow using the SVHN and CIFAR-100 datasets by adding a parameter `dataset` to specify the dataset we would like to use.
:::

::: {.cell .code}
``` python
def train_to_threshold(title='warm', dataset='cifar10', lr=0.001, checkpoint=None, use_half_data=False,
                       optimizer_name='adam', model_name='resnet18', convergence_epochs=4,
                       train_threshold=0.5, convergence_change_threshold=0.002, random_seed=42):
    # use gpu if available ( change device id if needed )
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Get the dataset
    loaders = get_loaders(dataset=dataset, use_half_train=use_half_data)

    # Define the number of classes
    num_classes = 10
    if dataset == 'cifar100':
        num_classes=100

    # Get the model
    if model_name == 'resnet18':
        model = models.resnet18(num_classes=num_classes).to(device)
    else:
        model = MLP( input_dim = 32 * 32 * 3, num_classes=num_classes).to(device)

    # Create the optimizer
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Get model from checkpoint
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device)['model'])

    print(f"Training {'random initialized' if checkpoint is None else 'warm-starting'} {model_name} model wtih " \
            f"{optimizer_name.upper()} optimizer on {'50%' if use_half_data else '100%'} of {dataset} dataset")

    # initialize training variables
    train_accuracies = []
    stop_indicator = False
    epoch = 0
    # Train until convergence or stop condition is met
    while(not stop_indicator):
        if epoch % 5 == 0:
            print(f"\t Training in epoch {epoch + 1} \t")
        # Train for one epoch and get loss and accuracy
        train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion, loaders['train_loader'])

        # Append training accuracy to list
        train_accuracies.append(train_accuracy)
        epoch += 1
        # Check if training accuracy is above a threshold
        if train_accuracy >= train_threshold:
            print(f"Convergence codition met. Training accuracy > {train_threshold}")
            stop_indicator = True

        # Check if training accuracy has stopped improving for a number of epochs
        if len(train_accuracies) >= convergence_epochs:
            if np.std(train_accuracies[-convergence_epochs:]) < convergence_change_threshold:
                print(f"\tConvergence codition met. Training accuracy = {train_accuracy} stopped improving")
                stop_indicator = True

    # Evaluate on test set and get loss and accuracy
    test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
    print(f"\tTest accuracy = {test_accuracy}")

    # Save the model if it will be used for warm-starting
    if use_half_data:
        # Create directory in exp with experiment title
        experiment_dir = os.path.join('experiments/exp2', title)
        os.makedirs(experiment_dir, exist_ok=True)

        # save the model
        model_name = model_name+'-'+optimizer_name
        model_directory =  os.path.join(experiment_dir, f'{model_name}.pt')
        torch.save({
            'model': model.state_dict()
        }, model_directory)

        print(f"Model saved to checkpoint: {model_directory}")

    return test_accuracy * 100
```
:::

::: {.cell .markdown}
***

We repeat all the previous for the CIFAR-100 dataset using the ResNet model with different optimizers.
:::

::: {.cell .code}
``` python
# train Resnet model on full CIFAR-100 data with random initialization and SGD optimizer
random_init = train_to_threshold(title='resnet-sgd-cifar100', dataset='cifar100', train_threshold=0.99,
                                       optimizer_name='sgd', model_name='resnet18')

# train on half CIFAR-100 data
_ = train_to_threshold(title='resnet-sgd-cifar100', dataset='cifar100', train_threshold=0.99, optimizer_name='sgd',
                   model_name='resnet18', use_half_data=True)

# train Resnet model on full CIFAR-100 data with warm-starting and SGD optimizer
warm_start = train_to_threshold(title='resnet-sgd-cifar100', dataset='cifar100', train_threshold=0.99, 
                                optimizer_name='sgd', model_name='resnet18', 
                                checkpoint='experiments/exp2/resnet-sgd-cifar100/resnet18-sgd.pt')

# get the difference between random and warm-start ResNet models using SGD optimizer on CIFAR-100
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['resnet-sgd-cifar100'] = [random_init, warm_start, diff]

# train Resnet model on full CIFAR-100 data with random initialization and Adam optimizer
random_init = train_to_threshold(title='resnet-adam-cifar100', dataset='cifar100', train_threshold=0.99,
                                            optimizer_name='adam', model_name='resnet18')

# train on half CIFAR-100 data
_ = train_to_threshold(title='resnet-adam-cifar100', dataset='cifar100', train_threshold=0.99, 
                       optimizer_name='adam', model_name='resnet18', use_half_data=True)

# train Resnet model on full CIFAR-100 data with warm-starting and Adam optimizer
warm_start = train_to_threshold(title='resnet-adam-cifar100', dataset='cifar100', train_threshold=0.99, 
                                optimizer_name='adam', model_name='resnet18', 
                                checkpoint='experiments/exp2/resnet-adam-cifar100/resnet18-adam.pt')

# get the difference between random and warm-start ResNet models using Adam optimizer on CIFAR-100
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['resnet-adam-cifar100'] = [random_init, warm_start, diff]
```
:::

::: {.cell .markdown}
***

We repeat all the previous for the CIFAR-100 dataset using the MLP model with different optimizers.
:::

::: {.cell .code}
``` python
# train MLP model on full CIFAR-100 data with random initialization and SGD optimizer
random_init = train_to_threshold(title='mlp-sgd-cifar100', dataset='cifar100', train_threshold=0.99,
                                             optimizer_name='sgd', model_name='mlp')

# train on half CIFAR-100 data
_ = train_to_threshold(title='mlp-sgd-cifar100', dataset='cifar100', train_threshold=0.99,
                       optimizer_name='sgd', model_name='mlp', use_half_data=True)

# train MLP model on full CIFAR-100 data with warm-starting and SGD optimizer
warm_start = train_to_threshold(title='mlp-sgd-cifar100', dataset='cifar100', train_threshold=0.99, 
                                optimizer_name='sgd', model_name='mlp', 
                                checkpoint='experiments/exp2/mlp-sgd-cifar100/mlp-sgd.pt')

# get the difference between random and warm-start MLP models using SGD optimizer on CIFAR-100
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['mlp-sgd-cifar100'] = [random_init, warm_start, diff]

# train MLP model on full CIFAR-100 data with random initialization and Adam optimizer
random_init = train_to_threshold(title='mlp-adam-cifar100', dataset='cifar100',train_threshold=0.99,
                                              optimizer_name='adam', model_name='mlp')

# train on half CIFAR-100 data
_ = train_to_threshold(title='mlp-adam-cifar100', dataset='cifar100', train_threshold=0.99, 
                       optimizer_name='adam', model_name='mlp', use_half_data=True)

# train MLP model on full CIFAR-100 data with warm-starting and Adam optimizer
warm_start = train_to_threshold(title='mlp-adam-cifar100', dataset='cifar100', train_threshold=0.99, 
                                optimizer_name='adam', model_name='mlp', 
                                checkpoint='experiments/exp2/mlp-adam-cifar100/mlp-adam.pt')

# get the difference between random and warm-start MLP models using Adam optimizer on CIFAR-100
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['mlp-adam-cifar100'] = [random_init, warm_start, diff]
```
:::

::: {.cell .markdown}
***

We create the previous for the SVHN dataset using the ResNet model with different optimizers.
:::

::: {.cell .code}
``` python
# train ResNet model on full SVHN data with random initialization and SGD optimizer
random_init = train_to_threshold(title='resnet-sgd-svhn', dataset='svhn', train_threshold=0.99,
                                       optimizer_name='sgd', model_name='resnet18')

# train on half SVHN data
_ = train_to_threshold(title='resnet-sgd-svhn', dataset='svhn', train_threshold=0.99, 
                       optimizer_name='sgd', model_name='resnet18', use_half_data=True)

# train ResNet model on full SVHN data with warm-starting and SGD optimizer
warm_start = train_to_threshold(title='resnet-sgd-svhn', dataset='svhn', train_threshold=0.99, 
                                optimizer_name='sgd', model_name='resnet18',
                                checkpoint='experiments/exp2/resnet-sgd-svhn/resnet18-sgd.pt')

# store the difference between random and warm-start ResNet models using SGD optimizer on SVHN
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['resnet-sgd-svhn'] = [random_init, warm_start, diff]

# train ResNet model on full SVHN data with random initialization and Adam optimizer
random_init = train_to_threshold(title='resnet-adam-svhn', dataset='svhn', train_threshold=0.99,
                                        optimizer_name='adam', model_name='resnet18')

# train on half SVHN data
_ = train_to_threshold(title='resnet-adam-svhn', dataset='svhn', train_threshold=0.99, 
                       optimizer_name='adam', model_name='resnet18', use_half_data=True)

# train ResNet model on full SVHN data with warm-starting and Adam optimizer
warm_start = train_to_threshold(title='resnet-adam-svhn', dataset='svhn', train_threshold=0.99, 
                                optimizer_name='adam', model_name='resnet18', 
                                checkpoint='experiments/exp2/resnet-adam-svhn/resnet18-adam.pt')

# store the difference between random and warm-start ResNet models using Adam optimizer on SVHN
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['resnet-adam-svhn'] = [random_init, warm_start, diff]
```
:::

::: {.cell .markdown}
***

We create the previous for the SVHN dataset using the MLP model with different optimizers.
:::

::: {.cell .code}
``` python
# train MLP model on full SVHN data with random initialization and SGD optimizer
random_init = train_to_threshold(title='mlp-sgd-svhn', dataset='svhn', train_threshold=0.99,
                                             optimizer_name='sgd', model_name='mlp')

# train on half SVHN data
_ = train_to_threshold(title='mlp-sgd-svhn', dataset='svhn', train_threshold=0.99, 
                       optimizer_name='sgd', model_name='mlp', use_half_data=True)

# train MLP model on full SVHN data with warm-starting and SGD optimizer
warm_start = train_to_threshold(title='mlp-sgd-svhn', dataset='svhn', train_threshold=0.99, 
                                optimizer_name='sgd', model_name='mlp',
                                checkpoint='experiments/exp2/mlp-sgd-svhn/mlp-sgd.pt')

# get the difference between random and warm-start MLP models using SGD optimizer on SVHN
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['mlp-sgd-svhn'] = [random_init, warm_start, diff]

# train MLP model on full SVHN data with random initialization and Adam optimizer
random_init = train_to_threshold(title='mlp-adam-svhn', dataset='svhn', train_threshold=0.99,
                                              optimizer_name='adam', model_name='mlp')

# train on half SVHN data
_ = train_to_threshold(title='mlp-adam-svhn', dataset='svhn', train_threshold=0.99, 
                       optimizer_name='adam', model_name='mlp', use_half_data=True)

# train MLP model on full SVHN data with warm-starting and Adam optimizer
warm_start = train_to_threshold(title='mlp-adam-svhn', dataset='svhn', train_threshold=0.99, 
                                optimizer_name='adam', model_name='mlp',
                                checkpoint='experiments/exp2/mlp-adam-svhn/mlp-adam.pt')

# get the difference between random and warm-start MLP models using Adam optimizer on SVHN
diff = random_init - warm_start

# Store the results in the dictionary
overal_result['mlp-adam-svhn'] = [random_init, warm_start, diff]
```
:::

::: {.cell .markdown}
***

We save all the previous results in the `overal_result` dictionary and save it in `overal_result.json` to be loaded for table creation.
:::

::: {.cell .code}
``` python
# Save the outputs in a json file
with open("experiments/exp2/overal_result.json", "w") as f:
    json.dump(overal_result, f)
```
:::

::: {.cell .markdown}
***

The table is created in the next cell so we can compare our results with the table from second claims.
:::

::: {.cell .code}
``` python
# Read from json file
with open("experiments/exp2/overal_result.json", "r") as f:
    overal_result = json.load(f)

# Create a dataframe with the result to be in a table form
df = pd.DataFrame.from_dict(overal_result).rename(index={0: "Random Init", 1: "Warm Start", 2: "Difference"})

# Display the dataframe
display(df.style.set_properties(**{'text-align': 'center', 'border': '1px solid black', 'padding': '5px'}))
```
:::

::: {.cell .markdown}
***

<p style="color: crimson;font-size: 16px;">Did the experiment description provide all the parameter values or did we make any assumptions? If so, what criteria do you think was used to make those assumptions?</p>
:::

::: {.cell .markdown}
<p style="color: green; font-size: 16px;"> Answer: </p>

***
:::

::: {.cell .markdown}
### Things to try:
In this experiment you can:

- Change the learning rate by setting `lr=0.0001` as an argument in the `train_to_threshold` function
- Experiment with different `train_threshold` values and see how they affect the training time and the generalization gap

***
:::

::: {.cell .markdown}
If you are using colab click on this link to go to the next notebook: [Open in Colab](https://colab.research.google.com/github/mohammed183/ml-reproducibility-p1/blob/main/notebooks/05-Experiment3.ipynb)
:::