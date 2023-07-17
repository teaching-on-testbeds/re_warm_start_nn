::: {.cell .markdown}
# Experiments

In this section, we will test the claims made by the authors. You will come across sections in the code marked with `#TODO`, where you need to fill an argument as described in the experiment description.

<p style="color: crimson;font-size: 16px;"> Note: The parameter values given in the paper are specified in the description of each experiment. Otherwise, we need to assume them.</p>

***
:::

::: {.cell .markdown}
## Experiment 1:
In this experiment we want to compare two ways of training a ResNet-18 model, which is a type of deep neural network that can classify images. The CIFAR-10 dataset is a collection of 60,000 color images of 10 classes, such as airplanes, cars, and dogs. The experiment splits the dataset into two parts: a training set and a test set. The training set is used to update the model weights, and the test set is used to evaluate the model performance.

The experiment uses two models: a warm-starting model and a randomly initialized model. The warm-starting model starts with some pre-trained weights that are learned training on 50% of the training data for 350 epochs. The randomly initialized model starts with random weights that are not learned from any data. Both models train on the full training data for 350 epochs, where one epoch means one pass over the entire data. The experiment will measure the accuracy of the models on both the training and test sets, which is the percentage of correctly classified images.

To run this experiment we will need to:

1. Create `get_loaders` function to load the CIFAR-10 dataset and split it into training and test sets.
2. Define a function that takes a model, a data loader, an optimizer, and a loss function, and trains the model for a given number of epochs, saving the model weights after each epoch.
3. Create a ResNet-18 model and train it for 350 epochs on 50% of the training data, using stochastic gradient descent as the optimizer and cross entropy loss as the loss function. Save the final model weights as `half_cifar.pt`.
4. Create another ResNet-18 model and load the weights from `half_cifar.pt`. Train this model for another 350 epochs on the full training data, using the same optimizer and loss function. Save the final model weights as `warm_start_full.pt`.
5. Create a third ResNet-18 model with random weights. Train this model for 350 epochs on the full training data, using the same optimizer and loss function. Save the final model weights as `random_full.pt`.
6. Evaluate the test accuracy of all three models using the test data loader. Plot the accuracy curves of the models over time. Compare the results with those reported in the paper and analyze the differences.
::: 

::: {.cell .code}
``` python
import os 
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms, datasets, models
```
:::

::: {.cell .markdown}
***

The following function is `get_loaders` which we use to load the CIFAR-10 dataset, which consists of 60,000 color images of 10 classes, and returns data loaders for training and testing. The function takes four parameters:

- `dataset`: A string that specifies the desired dataset. For this experiment, we use the `cifar10` key to access the CIFAR-10 dataset from the `dataset_factories` dictionary.
- `use_half_train`: a boolean flag that indicates whether to use only half of the training data or the whole dataset. If this is set to `True`, then the parameter `dataset_portion` is automatically set to 0.5.
- `batch_size`: an integer that specifies the number of images to process in each batch. A larger batch size may speed up the training but also require more memory.
- `dataset_portion`: a double value between 0 and 1 that indicates the portion of the training data to use. For example, if this is set to 0.8, then only 80% of the training data will be used and the rest will be discarded.

The function returns a dictionary with two keys: `train_loader` and `test_loader` which can be used to iterate over the training and testing data respectively. The function also downloads the dataset from torchvision datasets if it is not already present in the specified directory.
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

The following function is the `train_model_exp1` which trains a ResNet-18 model on the CIFAR-10 dataset and returns the train and test accuracies. The function takes six parameters:

- `title`: a string that specifies the name of the experiment. This is used to create a subdirectory under the `experiments/exp1` directory where the model checkpoints and final weights will be saved.
- `experiment_dir`: a string that specifies the path of the experiment directory. If this is `None`, then the function will use the title parameter to create a default directory name.
- `use_half_data`: a boolean flag that indicates whether to use half of the training data or the whole dataset. This is passed to the `get_loaders` function that loads the data loaders.
- `lr`: a float value that specifies the learning rate for the stochastic gradient descent optimizer.
- `checkpoint`: a string that specifies the path of a model checkpoint file. If this is not `None`, then the function will load the model weights from the checkpoint file and resume training from there.
- `epochs`: an integer that specifies the number of epochs to train the model for.

The function returns a tuple of two lists: `train_acc` and `test_acc`. These are lists that contain the train and test accuracies for each epoch, respectively. The function uses the [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) model from the torchvision models. The function also sets the random seeds for reproducibility. The function uses cross entropy loss as the loss function and SGD as an optimizer. The function also uses a helper function called get_accuracy to compute the accuracy of the model predictions.
:::

::: {.cell .code}
``` python
# Function takes predictions and true values to return accuracies
def get_accuracy(logit, true_y):
    pred_y = torch.argmax(logit, dim=1)
    return (pred_y == true_y).float().mean()

# Function to train the model and return train and test accuracies
def train_model_exp1(title='', experiment_dir=None, use_half_data=False, lr=0.001, checkpoint=None, epochs=10):
    # Create experiment directory name if none
    if experiment_dir is None:
        experiment_dir = os.path.join('experiments/exp1', title)

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
    loaders = get_loaders(dataset="cifar10", use_half_train=use_half_data)
    num_classes = 10

    # Get the model
    model = models.resnet18(num_classes=10).to(device)

    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Get model from checkpoint
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device)['model'])

    # Arrays to hold accuracies
    test_acc = [0]
    train_acc = [0]

    # Train the model
    for epoch in range(1, epochs + 1):
        model.train()
        print(f"Epoch {epoch}")
        accuracies = []
        losses = []

        for batch_idx, (data_x, data_y) in enumerate(loaders["train_loader"]):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()
            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)
            loss.backward()
            optimizer.step()

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())


        train_loss = np.mean(losses)
        train_accuracy = np.mean(accuracies)
        train_acc.append(train_accuracy*100)

        print("Train accuracy: {} Train loss: {}".format(train_accuracy, train_loss))

        accuracies = []
        losses = []
        model.eval()
        for batch_idx, (data_x, data_y) in enumerate(loaders["test_loader"]):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())

        test_loss = np.mean(losses)
        test_accuracy = np.mean(accuracies)
        test_acc.append(test_accuracy*100)
        print("Test accuracy: {} Test loss: {}".format(test_accuracy, test_loss))


    torch.save({
        'model': model.state_dict()
    }, os.path.join(experiment_dir, 'final.pt'))
    
    # return the accuracies
    return train_acc, test_acc
```
:::

::: {.cell .markdown}
***

To be used warm-starting a model later, we first train a model for 350 epochs on 50% of the CIFAR-10 dataset. We keep track of the train and test accuracies at each epoch, which will form the blue line on the left half of figure 1.

We set `use_half_data` to `True` to train on only half of the CIFAR-10 dataset. We don’t need a `checkpoint` since we start from scratch. We use `lr = 0.001` for all models, following the original paper.
:::

::: {.cell .code}
``` python
# initialize runs dictionary to hold runs outputs
runs = {}

# Run the train_model_exp1 function to get train and test accuracies for the first model (trained on half the data)
half_cifar_train_acc, half_cifar_test_acc = train_model_exp1( title="half_cifar",
                                                                use_half_data=True,
                                                                lr=0.001,
                                                                checkpoint=None,
                                                                epochs=350 )

# Put the results in the runs dictionary
runs["half_cifar"] = [half_cifar_train_acc, half_cifar_test_acc, 0]
```
:::

::: {.cell .markdown}
***

Now we use the previous model to train a warm-starting model for 350 epochs on 100% of the CIFAR-10 dataset. We keep track of the train and test accuracies at each epoch, which will form the blue line on the right half of figure 1.

We set `use_half_data` to `False` to train on the full CIFAR-10 dataset. We add a `checkpoint` to the model trained on 50% of the data.
:::

::: {.cell .code}
``` python
# Run the train_model_exp1 function to get train and test accuracies for the Second model ( warm starting )
warm_start_train_acc, warm_start_test_acc = train_model_exp1( title="warm_start",
                                                                use_half_data=False,
                                                                lr=0.001,
                                                                checkpoint='experiments/exp1/half_cifar/final.pt',
                                                                epochs=350 )

# Put the results in the runs dictionary
runs["warm_start"] = [warm_start_train_acc, warm_start_test_acc, 1]
```
:::

::: {.cell .markdown}
***

Finaly, we train a model for 350 epochs on 100% of the CIFAR-10 dataset. We keep track of the train and test accuracies at each epoch, which will form the orange line on the right half of figure 1.

We set `use_half_data` to `False` to train on the full CIFAR-10 dataset. We don’t need a `checkpoint` since we start from scratch.
:::

::: {.cell .code}
``` python
# Run the train_model_exp1 function to get train and test accuracies for the Last model ( randomly initialized )
full_cifar_train_acc, full_cifar_test_acc = train_model_exp1( title="full_cifar",
                                                                use_half_data=False,
                                                                lr=0.001,
                                                                checkpoint=None,
                                                                epochs=350 )

# Put the results in the runs dictionary
runs["full_cifar"] = [full_cifar_train_acc, full_cifar_test_acc, 1]
```
:::

::: {.cell .markdown}
***

Now we save the training and test accuracies in the runs dictionary in `runs.json`.
:::

::: {.cell .code}
``` python
# Save the outputs in a json file
with open("experiments/exp1/runs.json", "w") as f:
    json.dump(runs, f)
```
:::

::: {.cell .markdown}
***

Let’s visualize the accuracies and analyze the outcomes! Run the next cell to plot the accuracies.
:::

::: {.cell .code}
``` python
# Read from json file
with open("experiments/exp1/runs.json", "r") as f:
    runs = json.load(f)

# Get number of epochs
epochs = len(list(runs.items())[0][1][0])

# Select colors
colors = {'half_cifar' : 'C0',
          'warm_start' : 'C0',
          'full_cifar' : 'C1',
         }

# Plot train Figure
plt.figure()
for title, vals in runs.items():
    offset = epochs * vals[2]
    x = np.arange(offset, offset + len(vals[0]))
    y = vals[0]
    plt.plot(x, y, label=title, c=colors[title])
plt.legend()

plt.ylabel(" Train accuracy ")
plt.ylim(0, 100)
plt.plot([epochs, epochs], plt.gca().get_ylim(), '--', c='black')
plt.savefig("experiments/exp1/fig1_train.png")

# Plot test Figure
plt.figure()
for title, vals in runs.items():
    offset = epochs * vals[2]
    x = np.arange(offset, offset + len(vals[1]))
    y = vals[1]
    plt.plot(x, y, label=title, c=colors[title])
plt.legend()

plt.ylabel(" Test accuracy ")
plt.ylim(0, 100)
plt.plot([epochs, epochs], plt.gca().get_ylim(), '--', c='black')
plt.savefig("experiments/exp1/fig1_test.png")
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
This experiment uses a specific model and optimizer. Exploring different combinations might be beneficial but costly in terms of computation. A simple way to further examine the first claim is:

- Use a lower learning rate since the model achieves 99% training accuracy quickly
- Use number of epochs at which validation accuracies of both models are maximized

***
:::

::: {.cell .markdown}
If you are using colab click on this link to go to the next notebook: [Open in Colab](https://colab.research.google.com/github/mohammed183/ml-reproducibility-p1/blob/main/notebooks/04-Experiment2.ipynb)
:::