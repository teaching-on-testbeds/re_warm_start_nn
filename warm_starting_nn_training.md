::: {.cell .markdown}
# On Warm-Starting Neural Network Training

The paper is available on [arXiv](https://arxiv.org/abs/1910.08475). In creating the interactive material for this notebook, we utilized the code from this reproducibility challenge: [Re: Warm-Starting Neural Network Training](https://rescience.github.io/bibliography/Kireev_2021.html).
:::

::: {.cell .markdown}
## Goals

At the end of this notebook, you will:

- learn to identify specific claims, both qualitative and quantitative, in a machine learning research paper
- learn to identify the specific experiments they would need to run to validate each claim
- learn to identify the data, code, and hyperparameters needed to run each experiment, and to make appropriate choices when these are not available
- understand the computational cost associated with reproducing a result, and the effect of missing information on that cost.
:::

::: {.cell .markdown}
### While experimenting we need to answer some questions to understand the level of reproducibility of this paper:

-   Is there code available for training? for inference?
-   Is it author code, or written by someone else? Are there multiple implementations available?
-   What framework and version was used? Are all the functions are still available or should you make some changes?
-   Did the author compare to other models that are not implemented in the code? Are these models available?
-   Are all hyperparameters for all experiments available? If not, what is the sensitivity of each hyperparameter?
-   Was the initial values set at random?
-   Are the datasets used available? Are there any modifications done to the data?
-   Did our results match that of the original paper?
:::

::: {.cell .markdown}
## Introduction

Warm starting and cold starting are two different ways of initializing the weights of a neural network before training. <b>Cold starting</b> means starting with random weights, while <b>warm starting</b> means starting with weights copied from a previously trained model. In our context the model is previously trained on a subset of the same dataset.

Updating datasets over time can be a costly endeavor, making it impractical to retrain models from scratch each time. Therefore, warm-starting becomes crucial as it allows leveraging pre-trained weights on a subset of the data, significantly reducing the time and resources required for training. By utilizing warm-starting, models can be efficiently adapted to incorporate new data without incurring the high computational expenses associated with starting from scratch.
:::

::: {.cell .markdown}
## Primary Claims:
The original paper makes several claims that can be classified as either quantitative or qualitative. Read the following claims carefully to be able to conduct the right experiment.

***
:::

::: {.cell .markdown}
<p style="color: crimson;font-size: 16px;"> Which model do you expect to have a better generalization performance (Test accuracy): <br>
    <p style="display:inline-block;margin-top:0.5em;margin-left: 2em; color:crimson;font-size: 16px;">
        1- Model that uses the weights from a previous model trained on a subset of the data (Warm-starting). <br>
        2- Model that starts with random weights (Cold-starting). 
    </p>
</p>
:::

::: {.cell .markdown}
<p style="color: green; font-size: 16px;"> Answer: </p>

***
:::

::: {.cell .markdown}
### Claim 1: Warm-starting neural network training may result in lower validation accuracy than random initialized models, despite having similar final training accuracy.
![Figure](assets/claim1.png) 
*We compare a warm-starting ResNet-18 model (Blue) and a randomly initialized ResNet-18 model (Orange) on the CIFAR-10 dataset. The warm-starting model first trains on 50% of the data for 350 epochs, then both models train on the full dataset for another 350 epochs. The figure shows that both models overfit the training data, but the randomly initialized model achieves higher test accuracy.*

- Excerpt: 

> "However, warm-starting seems to hurt generalization in deep neural networks. This is particularly troubling because warm-starting does not damage training accuracy."

- Type: This claim is qualitative because it states that the warm-start model has worse generalization performance than the fresh-start model, without stating a clear numerical evidence.
- Experiment: A possible way to evaluate this claim is to use some unseen validation data and compare the performance of the two models using different metrics, such as accuracy, precision, recall, or others. You can also try different model architectures and datasets to test the claim’s robustness.

***
:::

::: {.cell .markdown}
### Claim 2: Warm-started models had worse test accuracies than randomly initialized models on CIFAR-10, SVHN, and CIFAR-100, using ResNet-18 and MLP.
[comment1]: <> (![Figure2](assets/claim3.png))

| CIFAR-10    | ResNet-SGD | ResNet-Adam | MLP-SGD | MLP-Adam | CIFAR-100   | ResNet-SGD | ResNet-Adam | MLP-SGD | MLP-Adam |    SVHN     | ResNet-SGD | ResNet-Adam | MLP-SGD | MLP-Adam |
| :---------: | :--------: | :---------: |:------: |:-------: | :---------: | :--------: | :---------: |:------: |:-------: | :---------: | :--------: | :---------: |:------: |:-------: |
| Random init |     56.2    |     78.0     |   39.0   |   39.4    |  |    18.2     |     41.4     |   10.3   |    11.6   |   |  89.4      |   93.6      |    76.5 |  76.7    |
| Warm-Start  |    51.7     |    74.4      |  37.4    |    36.1   |   |     15.5    |     35.0     |   9.4   |    9.9   |   |  87.5      |     93.5    |   75.4  |   69.4   |
| Difference  |      4.5   |     3.6     |    1.6  |   3.3    |   |    2.7     |     6.4     |    0.9  |  1.7     |   |      1.9   |    0.1      |   1.1   |    7.3   |


- Excerpt: 

> "Our results (Table 1) indicate that generalization performance is damaged consistently and significantly for both ResNets and MLPs. This effect is more dramatic for CIFAR-10, which is considered relatively challenging to model (requiring, e.g., data augmentation), than for SVHN, which is considered easier."

- Type: This is a quantitative claim, as it uses numerical values to compare the performance of different models on different datasets.
- Experiment: To verify this claim, you will need to follow the authors’ details and train the models mentioned. Then, you will need to compare their test accuracies. However, some of the accuracy differences are very small, especially for the SVHN dataset. Therefore, reproducing these results may be difficult without the authors’ hyperparameters.

***
:::

::: {.cell .markdown}
<p style="color: darkblue; font-size: 16px;"> In most cases it is a reasonable strategy to warm-start the model and potentially achieve quicker convergence. As it would be inefficient to discard the old model that has already learned something.</p>

***
:::

::: {.cell .markdown}
<p style="color: crimson;font-size: 16px;"> Do you think warm-starting models will take more or less training time than random initialized models?</p>
:::

::: {.cell .markdown}
<p style="color: green; font-size: 16px;"> Answer: </p>

***
:::

::: {.cell .markdown}
### Claim 3: Warm-starting neural networks saves resources and time, but lowers accuracy by 10% compared to fresh models.
![Figure1](assets/claim2.png)\
*The data is divided into 1000-sample batches for online training. The warm-started (Blue) and randomly initialized (Orange) models train until 99% training accuracy. The plots show how training time and test accuracy vary with the number of samples.*

- Excerpt:

> “Nevertheless, it is highly desirable to be able to warm-start neural network training, as it would dramatically reduce the resource usage associated with the construction of performant deep learning systems.”

- Type: This claim is quantitative because it compares relation between the training time and mechanism used for initialization of weights. The figure also shows that there is more than a 10% difference in test accuracy given a certain training accuracy threshold ( 99% ).
- Experiment: A possible way to test this claim is to run an online training experiment with model; one of them should initialized using the old version and the other with random initialization each time and compare there test accuracies at the end.

***
:::

::: {.cell .markdown}
**Based on your understanding of the previous claims, answer the following question.**
:::

::: {.cell .markdown}
<p style="color: crimson;font-size: 16px;"> You have a trained model for a classification project. The dataset has 10,000 new samples. Your team wants to use the new data. One option is to retrain the model from scratch. Another option is to warm-start the model with the current weights. Which option do you prefer? Give a brief reason for your choice. </p>
:::

::: {.cell .markdown}
<p style="color: green; font-size: 16px;"> Answer: </p>

***
:::

::: {.cell .markdown}
## Experiments

In this section, we will test the claims made by the authors. You will come across sections in the code marked with `#TODO`, where you need to fill an argument as described in the experiment description.

***
:::

::: {.cell .markdown}
### Experiment 1:
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
import glob
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import random_split, ConcatDataset
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

# Plot train Figure
plt.figure()
for title, vals in runs.items():
    offset = epochs * vals[2]
    x = np.arange(offset, offset + len(vals[0]))
    y = vals[0]
    plt.plot(x, y, label=title)
plt.legend()

plt.ylabel(" Train accuracy ")
plt.ylim(0, 100)
plt.plot([epochs, epochs], plt.gca().get_ylim(), '--', c='black')
plt.savefig("fig1_train.pdf")

# Plot test Figure
plt.figure()
for title, vals in runs.items():
    offset = epochs * vals[2]
    x = np.arange(offset, offset + len(vals[1]))
    y = vals[1]
    plt.plot(x, y, label=title)
plt.legend()

plt.ylabel(" Test accuracy ")
plt.ylim(0, 100)
plt.plot([epochs, epochs], plt.gca().get_ylim(), '--', c='black')
plt.savefig("fig1_test.pdf")
```
:::

::: {.cell .markdown}
Now compare the the Figure you got with the one from the claim

***
:::


::: {.cell .markdown}
### Experiment 2:

In this experiment, we compare two methods of weight initialization: warm-starting and random initialization, for two models: **ResNet18** and **3-layer MLP**. We also compare two optimizers: **SGD** and **Adam**, for updating the weights based on the gradients. We use three image classification datasets:  **CIFAR-10**, **CIFAR-100** and **SVHN**, and report the test accuracy of each model on each dataset.

We use the same components as in experiment one: the `get_loaders` function to get the required dataset's train and test loaders, the [**ResNet18**](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) model from `torchvision.models`, and the [**SGD**](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) optimizer from `torch.optim`. We also introduce some new components:

1. The **CIFAR-100** dataset, which has 60,000 color images of 100 classes. To get the CIFAR-100 dataset, we pass the string `cifar100` as the dataset name argument to the `get_loaders` function.
2. The **SVHN** dataset, which has 73,257 color images of 10 classes of street view house numbers. To get the SVHN dataset, we pass the string `svhn` as the dataset name argument to the `get_loaders` function.
3. The `MLP` class that defines a 3-layer MLP model with a tanh activation function and a bias term.
4. The `torch.optim.Adam` class that implements the [**Adam**](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer, which is an adaptive learning rate method.
:::

::: {.cell .markdown}
***

The following is a class for a **multilayer perceptron** (MLP) model with several fully connected (fc) layers and a final fully connected layer for the logits output. The arguments are:

- `input_dim`: the input feature dimension.
- `num_classes`: the output class number.
- `hidden_units`: the hidden unit number for each fc layer.
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
def train_to_threshold(title='warm', lr=0.001, checkpoint=None, use_half_data=False, convergence_epochs=3,
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

    return test_accuracy
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
```
:::

::: {.cell .code}
``` python
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
                       use_half_data=False, optimizer_name='adam', convergence_epochs=3,
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

    return test_accuracy
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
                        optimizer_name='adam', model_name='resnet18', convergence_epochs=3,
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

    return test_accuracy
```
:::

::: {.cell .markdown}
***

We use warm-starting and the Adam optimizer to train the MLP model in the next cell.
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

In the next cell, we train the MLP model with warm-starting and the SGD optimizer.
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

Finaly we extend this to allow using the SVHN and CIFAR-100 datasets by adding a parameter `dataset` to specify the dataset we would like to use.
:::

::: {.cell .code}
``` python
def train_to_threshold(title='warm', dataset='cifar10', lr=0.001, checkpoint=None, use_half_data=False,
                       optimizer_name='adam', model_name='resnet18', convergence_epochs=3,
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

    return test_accuracy
```
:::

::: {.cell .markdown}
***

We repeat all the previous for the CIFAR-100 dataset using the ResNet model with different optimizers.
:::

::: {.cell .code}
``` python
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
```
:::

::: {.cell .markdown}
***

We repeat all the previous for the CIFAR-100 dataset using the MLP model with different optimizers.
:::

::: {.cell .code}
``` python
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
```
:::

::: {.cell .markdown}
***

We create the previous for the SVHN dataset using the ResNet model with different optimizers.
:::

::: {.cell .code}
``` python
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
```
:::

::: {.cell .markdown}
***

We create the previous for the SVHN dataset using the MLP model with different optimizers.
:::

::: {.cell .code}
``` python
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
:::

::: {.cell .markdown}
### Experiment 3:
In this experiment we conduct an experiment to compare the effects of random initialization and warm-starting on online training, which is a common scenario in real time setting. We divide the **CIFAR-10** dataset into splits of 1000 samples each, and train a **ResNet18** model on each split until it reaches *99% training accuracy*. We incrementally add more splits to the training data until we exhaust the whole dataset. We record the training time and test accuracy for each split and analyze the differences between the two initialization methods.

We reuse the same components from the previous experiment, except for the `get_cifar10_online_loaders` function, which returns a list train loaders with a variable number of samples. We also use the ResNet18 model from torchvision and the Adam optimizer from `torch.optim`, with cross entropy loss as the loss function.
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
        test_accuracies_online.append(test_accuracy)
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
plt.savefig(f"experiments/exp3/cifar10-99.pdf")
```
:::


::: {.cell .markdown}
## Explaining the results

In this section we will answer the questions in the begining of the notebook and maybe leave some room for the student to add his answers
:::
