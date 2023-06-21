::: {.cell .markdown}
# On Warm-Starting Neural Network Training

The paper is available on [arXiv](https://arxiv.org/abs/1910.08475). In creating the interactive material for this notebook, we utilized the code from this reproducibility challenge: [Re: Warm-Starting Neural Network Training](https://rescience.github.io/bibliography/Kireev_2021.html).
:::

::: {.cell .markdown}
## Introduction

Retraining neural networks with new data added to the training set is a time and energy-consuming task. To speed up this process, the technique of warm-starting can be used. Warm-starting involves using the weights of a pre-trained model, trained on a subset of the data, as the starting point for training the complete dataset.

The paper examines the impact of warm-starting on the final model's accuracy and highlights the presence of a generalization gap in warm-started models. The authors propose a method to address this gap by shrinking the pre-trained weights and introducing a random perturbation.

The warm-starting technique is an effective way to accelerate the training process of large neural network models. However, the paper emphasizes the importance of mitigating the generalization gap in warm-started models, and proposes a method to achieve better accuracy.

Updating datasets over time can be a costly endeavor, making it impractical to retrain models from scratch each time. Therefore, warm-starting becomes crucial as it allows leveraging pre-trained weights on a subset of the data, significantly reducing the time and resources required for training. By utilizing warm-starting, models can be efficiently adapted to incorporate new data without incurring the high computational expenses associated with starting from scratch.
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
## Claims by the authors:

-   Claim 1: Warm-starting neural network training can reduce the resource usage associated with the construction of performant deep learning systems.
-   Claim 2: Warm-starting neural network training can yield poorer generalization performance than models that have fresh random initializations, even though the final training losses are similar.
-   Claim 3: Warm-starting neural network training can be improved by using a simple trick that involves resetting the batch normalization statistics after copying the weights from a previous model.
-   Claim 4: The simple trick can close the generalization gap between warm-starting and random initialization in several important situations, such as when data arrive piecemeal, when data are actively selected, or when data are noisy or corrupted.
-   Claim 5: The simple trick can also improve the convergence speed and stability of warm-starting, especially when the learning rate is large or the batch size is small.
-   Claim 6: The reason why warm-starting can hurt generalization is that it can cause a mismatch between the batch normalization statistics and the data distribution, which can lead to suboptimal feature representations and gradient directions.
-   Claim 7: The reason why resetting the batch normalization statistics can mitigate this effect is that it can restore the alignment between the batch normalization statistics and the data distribution, which can lead to better feature representations and gradient directions.

Additional points:

-   Training a model initialized with the weights trained on a part of the same dataset leads to loss of generality in the deep neural network, identified as the warm-starting gap. A model trained on 100% of the data at once takes more time to train but yields better results.
-   The warm-starting gap is independent of batch size and learning rate.
-   Only a little training with a warm-starting model can lead to a loss of generality.
-   Regularization doesn't resolve the generalization gap.
-   Shrinking the weights doesn't significantly affect models without bias or batch normalization, but extreme shrinking can impact the performance of more sophisticated architectures.
-   Adding perturbation (noise) after shrinking improves both training time and generalization performance.
-   Utilizing the shrink-perturb trick can close the generalization gap and provide similar results to a newly randomly initialized model in less training time.
:::

::: {.cell .markdown}
### Conducting Experiments to Test Previous Claims

In this notebook, we will perform experiments to validate the claims mentioned earlier. Please note that some parts of the experiments will be incomplete, requiring you to fill in the missing functions with the correct parameter values as mentioned in the paper.

For the values that are not explicitly provided by the authors, you can try different values to assess the sensitivity of the hyperparameters used.

The missing code for the experiments can be found in the `solution.ipynb` notebook.
:::

::: {.cell .markdown}
## Functions

The following part contains data loaders, models, and training functions that will be used. Please note that you are not required to implement any of these functions, but rather read the notes on how to use them and understand their purpose.
:::

::: {.cell .markdown}
### Data Loaders

The paper utilizes publicly available datasets, including CIFAR10, CIFAR100, and SVHN. We will create data loaders that match the specifications of the data used in the paper.
:::

::: {.cell .code}
``` python
## The next few cell should contain the code with explaination on how to use the dataloaders. 
## The only functions the students will need to run are the training functions at the end however I will be adding
## explaination to everything.
```
:::

::: {.cell .markdown}
### Models:

The model used in the paper are Logistic Regression, 3-Layer Resnet-18 and Multi-layer perceptron.
:::

::: {.cell .code}
``` python
## The next few cells will contain the model implementations as function to be called in the training. 
## Explaination on the architecture and maybe some references will be added just in case anyone is curious to 
## understand the model architecture.
```
:::

::: {.cell .markdown}
### Training:

The training function will be added in this section, there are different training functions for different experiments.

I will add explaination on to use these fucntions and give some examples as the students will have to use these functions.
:::

::: {.cell .code}
``` python
## The next few cells will contain the function with detailed explaination on how to use them
```
:::

::: {.cell .markdown}
## Experiments

In this section, we will utilize the previously defined functions to test the claims made by the authors. You will come across sections in the code marked with `#TODO`, where you need to add one or two functions.

Please use the hyperparameter values provided in the paper. If certain hyperparameters are not provided, feel free to use your own values.
:::

::: {.cell .markdown}
### Experiment n: Testing Claim m

In this experiment, we aim to validate the m'th claim mentioned in the paper. The authors have provided a specific section and a corresponding figure to support their claim.

Our objective is to reproduce the figure and then compare it with the original one.

To accomplish this, we need to identify the hyperparameter values required for this experiment and include them in the training function.

Hint:

-   Utilize the `train1` function for this particular experiment.
-   Run the `plot_fig#` function in the subsequent cell.

(I will add assertions to ensure the proper utilization of resources and to detect any potential issues in the process.)
:::

::: {.cell .code}
``` python
## Example on the student should write
# TODO
params = get_params(the hyperparameter values he chooses)
returned_values_if_any = train(params)
# End
```
:::

::: {.cell .code}
``` python
plot_fig#()
```
:::

::: {.cell .markdown}
After the figure I will the figure that I got when doing this myself so they can make sure the one they got is right.
I will also add some notes about the results, whether they match or not and so.

#### The previous is repeated for all claims
:::

::: {.cell .markdown}
## Conclusion

In this section we will answer the questions in the begining of the notebook and maybe leave some room for the student to add his answers
:::
