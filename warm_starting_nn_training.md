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

### Claim 1: Warm-starting neural network training can reduce the resource usage and training time associated with the construction of performant deep learning models.

- Excerpt: "Nevertheless, it is highly desirable to be able to warm-start neural network training, as it would dramatically reduce the resource usage associated with the construction of performant deep learning systems."
- Type: This claim is qualitative because it compares the resource usage of warm-starting and cold-starting neural network training without giveing difference in percentage of resource usage. 
- Experiment: A possible way to test this claim is to train the same model with different percentages of the dataset: 50% for warm-starting and 0% for cold-starting. Then, measure how many epochs each model needs to achieve a certain level of training loss or accuracy.

### Claim 2: Warm-starting neural network training can yield poorer generalization performance than models that have fresh random initializations, even though the final training losses are similar.
![Figure1](assets/claim2.png)
*This is a comparison of warm-starting model (Blue) and randomly initialized model (Orange) in Figure 1 of the paper*

- Excerpt: "However, in practice this warm-starting seems to yield poorer generalization performance than models that have fresh random initializations, even though the final training losses are similar."
- Type: This claim is qualitative because it states that the warm-start model has worse generalization performance than the fresh-start model, without giving any numerical evidence or comparison but shows this in several figures in the paper.
- Experiment: A possible way to evaluate this claim is to use some unseen validation data and compare the performance of the models produced from verifying the first claim using different metrics, such as accuracy, precision, recall, or others. You can also try different model architectures and datasets to test the claim’s robustness.

### Claim 3: Compared to the random initialized models, the warm-started models achieved lower test accuracies on three datasets: CIFAR-10, SVHN, and CIFAR-100. The accuracy drops for ResNet-SGD, ResNet-Adam, MLP-SGD, and MLP-Adam were 4.5%, 3.6%, 1.6%, and 3.3% on CIFAR-10; 1.9%, 0.1%, 1.1%, and 7.3% on SVHN; and 2.7%, 6.4%, 0.9%, and 1.7% on CIFAR-100 respectively.
![Figure2](assets/claim3.png)\

- Excerpt: "Our results (Table 1) indicate that generalization performance is damaged consistently and significantly for both ResNets and MLPs. This effect is more dramatic for CIFAR-10, which is considered relatively challenging to model (requiring, e.g., data augmentation), than for SVHN, which is considered easier."
- Type: This is a quantitative claim, as it uses numerical values to compare the performance of different models on different datasets.
- Experiment: To verify this claim, you will need to follow the authors’ details and train the models mentioned. Then, you will need to compare their test accuracies. However, some of the accuracy differences are very small, especially for the SVHN dataset. Therefore, reproducing these results may be difficult without the authors’ hyperparameters.

### Claim 4: The accuracies of random initialized and warm-started LR-SGD and LR-Adam were similar on different datasets as the accuracy differences between them were 0.9% and 0.5% on CIFAR-10; 0.0% and 0.2% on SVHN; and 0.6% and 0.3% on CIFAR-100 respectively.
![Figure3](assets/claim4.png)\

- Excerpt: "Logistic regression, which enjoys a convex loss surface, is not significantly damaged by warm starting for any datasets."
- Type: This is a quantitative claim, as it uses numerical values to show that the accuracies of warm-started and cold-started logistic regression models are very similar.
- Experiment: You can verify this by training the models mentioned and compare their test accuracies.

### Claim 5: Warm-starting neural network training can achieve comparable generalization performance to randomly initialized models by tuning the batch size and learning rates, but without any benifit in training time.
![Figure4](assets/claim5.png)
*This is Figure 3 from the paper, warm-starting models (Blue) with randomly initialized models (Orange)*

- Excerpt: "Interestingly, we do find warm-started models that perform as well as randomly-initialized models, but they are unable to do so while benefiting from their warm-started initialization. The training time for warm-started ResNet models that generalize as well as randomly-initialized models is roughly the same as those randomly-initialized models."
- Type: This claim is qualitative because it specifies that the generalization performance is comparable but not how comparable it is, or how much resources are used.
- Experiment: You can verify this claim by trying different combinations of batch sizes and learning rates then plot the performance - training time relation for warm-start and cold-start models.

### Claim 6: A little training with a warm-start model can lead to loss of generality.
![Figure5](assets/claim6.png)
*This is Figure 4 from the paper, left is validation accuracy in 50% training while right is percentage of damage when training on 100%*

- Excerpt: "One surprising result in our investigation is that only a small amount of training is necessary to damage the validation performance of the warm-started model."
- Type: This claim is qualitative.
- Experiment: A possible way to verify this claim is to train the warm-start model on a subset of the data for a few epochs and measure its performance. Then, use the full data and observe how the performance drops.

### Claim 7:  For the regularization values of 0.1, 0.001, 0.0001, and 0.00001 respectively, the performance gap remained even after applying regularization. The gap for L2 regularization was 8.8, 4.2, 4.1, and 4.7; for adversarial training, it was 2.4, 2.5, 2.6, and 5.4; and for confidence-penalized training, it was 2.8, 5.8, 4.2, and 6.6.
![Figure6](assets/claim7.png)\

- Excerpt: "We apply regularization in both rounds of training, and while it is helpful, regularization does not resolve the generalization gap induced by warm starting."
- Type: This claim is a quantitative as it compares the test accuracies of warm-start and cold-start models after using different regularization methods.
- Experiment: Evaluate the effect of different regularization methods on the generalization gap for warm-start and cold-start models. The authors used weight decay, confidence penalized training and adversarial training as regularization methods.

### Claim 8: The shrink-and-perturb trick can overcome the generalization gap between warm-starting and cold-starting in several important situations.
![Figure7](assets/claim8.png)
*This is the new update equation where λ is the shrink factor and p is the perturbation value*

- Excerpt: "We describe a simple trick that overcomes this pathology, and report on experiments that give insights into its behavior in batch online learning and pre-training scenarios."
- Type: This claim is a qualitative observation that the shrink-perturb mechanism can reduce the generalization gap.
- Experiment: To test this claim, you can experiment with different values of shrinkage and perturbation on warm-started weights, and measure the performance of the models with and without this trick. You should cover different situations as explained in the paper.

### Claim 9: The shrink-and-perturb trick can reduce the generalization gap by eliminating the average gradient discrepancy between the first and second training.
![Figure8](assets/claim9.png)
*This is a visualization of the shrink-and-perturb trick’s effect on the gradient difference from Figure 5 of the original paper*

- Excerpt: "The success of the shrink and perturb trick may lie in its ability to standardize gradients while preserving a model’s learned hypothesis."
- Type: This is a qualitative claim.
- Experiment: Testing this claim will be by trying different shrink and perturbation value and check how it affect the average gradients in training.

### Claim 10: The generalization performance of pretrained models can be enhanced by using the shrink-and-perturb trick when the datasets are small or limited.
![Figure9](assets/claim10.png)
*This is figure 9 in the original paper, showing how the shrink-and-perturb trick can be used to pretrain on similar datasets*

- Excerpt: "We find that shrink-perturb initialization, however, allows us to avoid having to make such a prediction: shrink-perturbed models perform at least as well as warm-started models when pre-training is the most performant strategy and as well as randomly-initialized models when it is better to learn from scratch."
- Type: This claim is qualitative.
- Experiment: Testing this claim will be by using shrink-and-perturb trick to transfer learn models pretrained on different datasets.

__To summarize the previous claims:__

-   Training a model initialized with the weights trained on a part of the same dataset leads to loss of generality in the deep neural network, identified as the warm-starting gap. A model trained on 100% of the data at once takes more time to train but yields better results.
-   The warm-starting gap is independent of batch size and learning rate.
-   Only a little training with a warm-starting model can lead to a loss of generality.
-   Regularization doesn't resolve the generalization gap.
-   Shrinking the weights doesn't significantly affect models without bias or batch normalization, but extreme shrinking can impact the performance of more sophisticated architectures.
-   Adding perturbation (noise) after shrinking improves both training time and generalization performance.
-   Utilizing the shrink-perturb trick can close the generalization gap and provide similar results to a newly randomly initialized model in less training time.
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
