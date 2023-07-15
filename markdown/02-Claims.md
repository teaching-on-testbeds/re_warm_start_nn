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
If you are using colab click on this link to go to the next notebook: [Open in Colab](https://colab.research.google.com/github/mohammed183/ml-reproducibility-p1/blob/main/notebooks/03-Experiment1.ipynb)
:::