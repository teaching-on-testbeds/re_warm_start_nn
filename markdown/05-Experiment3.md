::: {.cell .markdown}
## Experiment 3:
In this experiment we conduct an experiment to compare the effects of random initialization and warm-starting on online training, which is a common scenario in real time setting. We divide the **CIFAR-10** dataset into splits of 1000 samples each, and train a **ResNet18** model on each split until it reaches *99% training accuracy*. We incrementally add more splits to the training data until we exhaust the whole dataset. We record the training time and test accuracy for each split and analyze the differences between the two initialization methods.

We reuse the same components from the previous experiment, except for the `get_cifar10_online_loaders` function, which returns a list train loaders with a variable number of samples. We also use the ResNet18 model from torchvision and the Adam optimizer from `torch.optim`, with cross entropy loss as the loss function.
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
import matplotlib.pyplot as plt
from torch.utils.data import random_split, ConcatDataset
from torchvision import transforms, datasets, models
```
:::

::: {.cell .markdown}
***
The `get_cifar10_online_loaders` function accepts a `split_size` parameter and returns train, test and validation loaders. The train loader is a list of loaders with increasing number of samples, where each loader adds `split_size` more samples to the previous one.
:::

::: {.cell .code}
```python
def get_cifar10_online_loaders(split_size):
    """
    This loads the whole CIFAR-10 into memory and returns train, validation and test data according to params
    train is returned into several data loaders
    @param split_size (int): size of train loaders

    @returns dict() with train, validation and
                test data loaders with keys `train_loaders`, `val_loader`, `test_loader`
    """
    
    # Normalization and transformation functions
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # Create train and test transforms
    train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    
    # Loading datasets from torch vision
    original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                              train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)
    
    # Validation split as 33% of train data
    val_dataset_size = int(len(original_train_dataset) / 3)
    train_dataset_size = len(original_train_dataset) - val_dataset_size
    original_train_dataset, val_dataset = random_split(original_train_dataset,
                                                       [train_dataset_size, val_dataset_size])

    train_datasets = random_split(original_train_dataset,
                                  [split_size for _ in range(train_dataset_size // split_size)] + [
                                      train_dataset_size % split_size])
    # Creating data loaders
    loader_args = {
        "batch_size": 128,
    }
    
    # train loader for every train dataset
    train_loaders = []
    active_datasets = []
    for train_dataset in train_datasets:
        active_datasets.append(train_dataset)
        train_loaders.append(torch.utils.data.DataLoader(
            dataset=ConcatDataset(active_datasets),
            shuffle=True,
            **loader_args
        ))

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loaders": train_loaders,
            "test_loader": test_loader,
            "val_loader": val_loader}
```
:::

::: {.cell .markdown}
***
We use some functions that were defined in previous experiments.
:::

::: {.cell .code}
```python
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
The `train_exp3` function performs an online learning experiment on the CIFAR-10 dataset, using a ResNet18 model and an Adam optimizer. The function uses the following new parameters:

- `init_type`: string indicating the type of initialization for the model. It can be either *random* or *warm*. If *random*, the model is reset for each split of data. If *warm*, the model is reused with the previous parameters.
- `split_size`: integer indicating the number of samples to add to the training data in each iteration.
:::

::: {.cell .code}
```python
from datetime import datetime

def train_exp3(init_type='random', lr=0.001, split_size=1000, acc_threshold=0.99, random_seed=42):
    experiment_dir = 'experiments/exp3'
    # make experiment directory
    os.makedirs(experiment_dir, exist_ok=True)

    # Set the seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    # Get the dataset
    loaders = get_cifar10_online_loaders(split_size=split_size)

    # Get the model
    model = models.resnet18(num_classes=10).to(device)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dictionary to hold results
    results = {}
    
    # Online learning setup
    number_of_samples_online = [0]
    test_accuracies_online = [0]
    training_times_online = [0]
    
    # Loop on all train loaders
    for i, train_loader in enumerate(loaders['train_loaders']):
        t_start = datetime.now()
        n_train = (i + 1) * split_size
        number_of_samples_online.append(n_train)
        
        # Reset model if training random initialization
        if init_type == 'random':
            model = models.resnet18(num_classes=10).to(device)
        
        # Reset the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print(f"{init_type.capitalize()}-Start training with {n_train} data.")
        
        # Train model until convergence
        stop_indicator = False
        epoch = 0
        while(not stop_indicator):
            if epoch % 5 == 0:
                print(f"Starting training in epoch {epoch + 1}")
            train_loss, train_accuracy = train_one_epoch(device, model, optimizer, criterion,
                                                         train_loader)
            epoch += 1
            
            if train_accuracy >= acc_threshold:
                print(f"Convergence condition met. Training accuracy > {100 * acc_threshold}")
                stop_indicator = True
                    
        # Calculate train time time
        t_end = datetime.now()
        training_time = (t_end - t_start).total_seconds()
        
        # Get test accuracy and append results
        test_loss, test_accuracy =  eval_on_dataloader(device, criterion, model, loaders['test_loader'])
        test_accuracies_online.append(test_accuracy * 100)
        training_times_online.append(training_time)        

    # Add to results
    results["test_accuracies_online"] = test_accuracies_online
    results["training_times_online"] = training_times_online
    results["number_of_samples_online"] = number_of_samples_online
        
    return results
```
:::

::: {.cell .markdown}
***
We start by training with **random** initialization using the previous function and store the results.
:::

::: {.cell .code}
```python
# Initialize dictionary to hold results
results={}

# Train on cifar10 for threshold 0.99
results['random'] = train_exp3(init_type='random', lr=0.001, split_size=1000, acc_threshold=0.99, random_seed=42)

# Save the outputs in a json file
with open("experiments/exp3/results.json", "w") as f:
    json.dump(results, f)
```
:::

::: {.cell .markdown}
***
Next, we train with **warm-starting** models and store the results.
:::

::: {.cell .code}
```python
# Train on cifar10 for threshold 0.99
results['warm'] = train_exp3(init_type='warm', lr=0.001, split_size=1000, acc_threshold=0.99, random_seed=42)

# Save the outputs in a json file
with open("experiments/exp3/results.json", "w") as f:
    json.dump(results, f)
```
:::

::: {.cell .markdown}
***
We plot the figure from the original paper using the data we obtained earlier and compare the results.
:::

::: {.cell .code}
```python
# Read from json file
with open("experiments/exp3/results.json", "r") as f:
    results = json.load(f)

# Create figure 
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# divide the number of samples to decrease them to two integers
number_of_samples_online = np.array(results['warm']['number_of_samples_online']) / 1000
number_of_samples_offline = np.array(results['random']['number_of_samples_online']) / 1000

# Left plot accuracy comparison
axs[0].plot(number_of_samples_online, results['warm']['test_accuracies_online'], label='warm start', color='C0')
axs[0].plot(number_of_samples_offline, results['random']['test_accuracies_online'], label='random', color='C1')
axs[0].set_ylabel("Test Accuracy")
axs[0].set_xlabel("Number of Samples (thousands)")

# Right plot time comparison
axs[1].plot(number_of_samples_online, results['warm']['training_times_online'], label='warm start', color='C0')
axs[1].plot(number_of_samples_offline, results['random']['training_times_online'], label='random', color='C1')
axs[1].set_ylabel("Train Time (seconds)")
axs[1].set_xlabel("Number of Samples (thousands)")

# Plot and save
plt.legend()
plt.show()
plt.savefig(f"experiments/exp3/cifar10-99.png")
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

- Change the learning rate and observe what effects it have on the experiment.
- Experiment other `split_size` values and see if it affects the results of the generalization gap
- Try different `acc_threshold` values and see how they affect the training time and the generalization gap

***
:::

::: {.cell .markdown}
If you are using colab click on this link to go to the next notebook: [Open in Colab](https://colab.research.google.com/github/mohammed183/ml-reproducibility-p1/blob/main/notebooks/06-Explanation.ipynb)
:::