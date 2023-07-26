::: {.cell .markdown}
# Experiments

In this section, we will evaluate the claims made by the authors. You should already know the general steps for each experiment from the previous section. We will now implement these experiments following the author description of each experiment and try to identify what was clear and what was vague due to incomplete information from the authors. 

***
:::

::: {.cell .markdown}
## Experiment 1:
This experiment will test the claim that *"Warm-starting neural network training can lead to lower test accuracy than random initialized models, even if they have similar final training accuracy"*. We anticipate that there will be a generalization gap between the two models trained with the two initialization methods.

We compare two ways of training a ResNet-18 model, which is a type of deep neural network that can classify images. The CIFAR-10 dataset is a collection of 60,000 color images of 10 classes, such as airplanes, cars, and dogs. The experiment splits the dataset into two parts: a training set and a test set. The training set is used to update the model weights, and the test set is used to evaluate the model performance.

The experiment uses two models: 

- The warm-starting model starts with some pre-trained weights that are learned by training the model on 50% of the training data. 
- The randomly initialized model starts with random weights that are not learned from any data.

Both models train on the full training data for 350 epochs, where one epoch means one pass over the entire data. The experiment will measure the accuracy of the models on both the training and test sets, which is the percentage of correctly classified images.
::: 

::: {.cell .code}
``` python
import os 
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms, datasets, models
```
:::

::: {.cell .markdown}
***

The following function is `get_loaders` which we use to load the CIFAR-10 dataset, which consists of 60,000 color images of 10 classes, and returns data loaders for training and testing. The function has three parameters, you will need to know the following about them:

- `use_half_train`: a boolean flag that indicates whether to use only half of the training data or the whole dataset. If this is set to `True`, then the parameter `dataset_portion` is automatically set to 0.5.
-   `dataset_portion`: a double value between 0 and 1 that indicates the portion of the training data to use. For example, if this is set to 0.8, then only 80% of the training data will be used and the rest will be discarded.

The function returns a dictionary with two keys: `train_loader` and `test_loader` which can be used to iterate over the training and testing data respectively. The function also downloads the dataset from torchvision datasets if it is not already present in the specified directory.
:::

::: {.cell .code}
``` python
def get_loaders(use_half_train=False, batch_size=128, dataset_portion=None):
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
    
    # Load data from torchvision datasets
    original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                         train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                         train=False, transform=test_transform, download=True)
    
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

The following function is the `train_model_epochs` which trains a ResNet-18 model on the CIFAR-10 dataset and returns the train and test accuracies. The function takes six parameters:

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
def train_model_epochs(title='', experiment_dir=None, use_half_data=False,
                        lr=0.001, checkpoint=None, epochs=10, random_seed=42):
    
    # Create experiment directory name if none
    if experiment_dir is None:
        experiment_dir = os.path.join('experiments/exp1', title)

    # make experiment directory
    os.makedirs(experiment_dir, exist_ok=True)

    # Set the seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    # Get the dataset
    loaders = get_loaders(use_half_train=use_half_data)

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

    # Iterate over the number of epochs
    for epoch in range(1, epochs + 1):
        model.train()
        print(f"Epoch {epoch}")
        accuracies = []
        losses = []
        
        # Calculate loss and gradients for models on every training batch
        for batch_idx, (data_x, data_y) in enumerate(loaders["train_loader"]):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()
            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)
            
            # Perform back propagation
            loss.backward()
            optimizer.step()

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())

        # Store training accuracy for plotting
        train_loss = np.mean(losses)
        train_accuracy = np.mean(accuracies)
        train_acc.append(train_accuracy*100)

        print("Train accuracy: {} Train loss: {}".format(train_accuracy, train_loss))

        # Evaluate the model on all the test batches
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

        # Store test accuracy for plotting
        test_loss = np.mean(losses)
        test_accuracy = np.mean(accuracies)
        test_acc.append(test_accuracy*100)
        print("Test accuracy: {} Test loss: {}".format(test_accuracy, test_loss))

    # Save the final model
    torch.save({
        'model': model.state_dict()
    }, os.path.join(experiment_dir, 'final.pt'))
    
    # return the accuracies
    return train_acc, test_acc
```
:::

::: {.cell .markdown}
***

Before running the experiment we create a parameter table to store the parameter values from the paper that we will use in the upcoming cells.

| Model                 | Learning rate | Epochs | use half data | Checkpoint           | Optimizer |
| :-------------------: | :-----------: | :----: | :-----------: | :------------------: | :-------: |
| Trained on half data  |  0.0001       |  350   |      True     | No checkpoint        |     SGD   |
| Warm-Starting         |  0.0001       |  350   |      False    | Trained on half data |     SGD   |
| Random initialized    |  0.0001       |  350   |      False    | No checkpoint        |     SGD   |
:::

::: {.cell .markdown}
***

To be used warm-starting a model later, we first train a model for 350 epochs on 50% of the CIFAR-10 dataset. We keep track of the train and test accuracies at each epoch, which will form the blue line on the left half of figure 1.

We set `use_half_data` to `True` to train on only half of the CIFAR-10 dataset. We don’t need a `checkpoint` since we start from scratch.
:::

::: {.cell .code}
``` python
# initialize runs dictionary to hold runs outputs
runs = {}

# Run the train_model_epochs function to get train and test accuracies for the first model 
# Random initialized model trained on half the data
half_train_acc, half_test_acc = train_model_epochs( title="half_cifar",
                                                    use_half_data=True,
                                                    lr=0.001,
                                                    checkpoint=None,
                                                    epochs=350 )

# Put the results in the runs dictionary
runs["half_cifar"] = { 'training_accuracy' : half_train_acc,
                       'test_accuracy' : half_test_acc,
                       'offset' : 0
                     }
```
:::

::: {.cell .markdown}
***

Now we use the previous model to train a warm-starting model for 350 epochs on 100% of the CIFAR-10 dataset. We keep track of the train and test accuracies at each epoch, which will form the blue line on the right half of figure 1.

We set `use_half_data` to `False` to train on the full CIFAR-10 dataset. We specify a `checkpoint` to the model trained on 50% of the data.
:::

::: {.cell .code}
``` python
# Run the train_model_epochs function to get train and test accuracies for the Second model
# Warm starting model using the first model
ws_train_acc, ws_test_acc = train_model_epochs( title="warm_start",
                                                use_half_data=False,
                                                lr=0.001,
                                                checkpoint='experiments/exp1/half_cifar/final.pt',
                                                epochs=350 )

# Put the results in the runs dictionary
runs["warm_start"] = { 'training_accuracy' : ws_train_acc,
                       'test_accuracy' : ws_test_acc,
                       'offset' : len(ws_test_acc)
                     }
```
:::

::: {.cell .markdown}
***

Finaly, we train a model for 350 epochs on 100% of the CIFAR-10 dataset. We keep track of the train and test accuracies at each epoch, which will form the orange line on the right half of figure 1.

We set `use_half_data` to `False` to train on the full CIFAR-10 dataset. We don’t need a `checkpoint` since we start from scratch.
:::

::: {.cell .code}
``` python
# Run the train_model_epochs function to get train and test accuracies for the Last model
# Model with random initialization
ri_train_acc, ri_test_acc = train_model_epochs( title="random_init",
                                                use_half_data=False,
                                                lr=0.001,
                                                checkpoint=None,
                                                epochs=350 )

# Put the results in the runs dictionary
runs["random_init"] = { 'training_accuracy' : ri_train_acc,
                       'test_accuracy' : ri_test_acc,
                       'offset' : len(ri_test_acc)
                     }
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
epochs = len(runs['half_cifar']['training_accuracy'])

# Select colors
colors = {'half_cifar' : 'C0',
          'warm_start' : 'C0',
          'random_init' : 'C1',
         }

# Plot train Figure
plt.figure()
for title, dictionary in runs.items():
    offset = dictionary['offset']
    x = np.arange(offset, offset + len(dictionary['training_accuracy']))
    y = dictionary['training_accuracy']
    plt.plot(x, y, label=title, c=colors[title])
plt.legend()

plt.ylabel(" Train accuracy ")
plt.ylim(0, 100)
plt.plot([epochs, epochs], plt.gca().get_ylim(), '--', c='black')
plt.savefig("experiments/exp1/fig1_train.png")

# Plot test Figure
plt.figure()
for title, dictionary in runs.items():
    offset = dictionary['offset']
    x = np.arange(offset, offset + len(dictionary['test_accuracy']))
    y = dictionary['test_accuracy']
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
**Did we validate the qualitative claim? Numerically, are the results consistent with the original paper? 🤔**

**In the parameter table we speicified the parameter values that we used in the experiment. Can you find these values in the paper text?  🔍**

***
:::

::: {.cell .markdown}
### Things to try: 🧪
This experiment uses a specific model and optimizer. Exploring different combinations might be beneficial but costly in terms of computation. A simple way to further examine the first claim is:

- Use a lower learning rate since the model achieves 99% training accuracy quickly
- Use number of epochs at which validation accuracies of both models are maximized
- Check the sensitivity of the model to the random seed by changing it

***
:::