::: {.cell .markdown}
# On Warm-Starting Neural Network Training 

The paper is available on [arXiv](https://arxiv.org/abs/1910.08475). In creating the interactive material for this notebook, we utilized the code from this reproducibility challenge: [Re: Warm-Starting Neural Network Training](https://rescience.github.io/bibliography/Kireev_2021.html).

***
:::

::: {.cell .markdown}
## Introduction

The paper *"On Warm-Starting Neural Network Training"* by Jordan T. Ash and Ryan P. Adams addresses a common challenge in machine learning scenarios where the neural network model needs to be updated with new data that comes in continuously. For instance, Netflix’s movie recommendation system needs to constantly update its model based on the user ratings and views, and the movie features and genres. This means the model has to train on the new data that is added to the old data each time there is new data available. Each time they train the model on the new dataset they can choose from different retraining strategies.

The paper investigates the trade-offs between two retraining strategies: 

- Starting from scratch with random weight initialization
- Using the weights of the model trained on the dataset before the new data was added

**What are your thoughts on each of these initializtions? Do you think they should yield similar results? Do you think they require the same training time?**  🤔
:::

:::{.cell .markdown}
Training a new model from scratch using the old and new data, ignoring the existing model weights and biases is called **Cold-starting**. Training a new model using the old and new data, but initializing the model weights and biases from the existing model is called **Warm-starting**. In the following notebooks We will evaluate the author’s claims and verify them with experiments to answer the previous question.

***
:::

::: {.cell .markdown} 
## Goals

The main purpose of this notebook is to:

- develop the skills to critically analyze specific claims, both qualitative and quantitative, that are made in the research paper
- learn to identify the specific experiments they would need to run to validate each claim
- learn to identify the data, code, and hyperparameters that are necessary to run each experiment, and to make reasonable choices when these are not provided
- understand the computational cost associated with reproducing a result, and the impact of missing information on that cost.
:::

::: {.cell .markdown}
**To assess the reproducibility level of this paper, we need to answer some questions while experimenting:**

-   Is there code available for both training and inference stages?
-   Is the code written by the authors themselves, or by someone else? Are there multiple implementations available for comparison?
-   What framework and version was used by the authors? Are all the functions still available or do we need to make some modifications?
-   Did the authors compare their model to other models that are not implemented in the code? Are these models available elsewhere?
-   Are all the hyperparameters for all the experiments clearly specified? If not, how sensitive is each hyperparameter to the performance?
-   Were the initial values set randomly or deterministically?
-   Are the datasets used by the authors accessible? Are there any preprocessing steps or modifications done to the data?
-   Did we obtain the same results as reported in the original paper?
:::