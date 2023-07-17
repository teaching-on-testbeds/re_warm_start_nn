::: {.cell .markdown}
# Primary Claims ☑️:

The original paper makes several claims that can be classified as either quantitative or qualitative. Read the following claims carefully to be able to understand the corresponding experiment. Intuitively, warm-starting should be faster and more effective than cold-starting, since it leverages the previous knowledge of the model. However, the paper shows that warm-starting often leads to worse generalization performance than cold-starting, even though the final training losses are similar.

The paper conducts several comprehensive experiments to compare the performance of warm-starting and cold-starting, demonstrating the trade-off between generalization and training time. They also vary the models, optimizers, and datasets used in the experiments to validate their findings.

The paper contributes to the understanding of how to efficiently utilize deep learning resources in dynamic environments, where data is constantly changing and models need to be updated frequently. The paper also provides some insights into the optimization landscape of neural networks, and how different initialization strategies affect the learning dynamics.

***
:::

::: {.cell .markdown}
## Claim 1: Warm-starting neural network training may result in lower test accuracy than random initialized models, despite having similar final training accuracy. 

The authors support their claim by comparing the test accuracy of two models trained with the two initialization strategies for **350** epochs each. The difference in generalization between the warm-starting model and the randomly initialized model that the authors trained is provided in the figure below.
![](assets/claim1.png) 
*Figure1: Model trained on half the data is represented by blue line from 0 to 350 epochs. Warm-starting is the blue line from 350-700 model while Random initialized model is the orange line.*

We will conduct an experiment that compares the two initialization strategies using the CIFAR-10 dataset and a ResNet-18 model. The warm-starting model is first trained on half of the data for 350 epochs, then both models are trained on the full data for another 350 epochs. We verify the overall qualitative claim by measuring the generalization gap between the warm-starting and the randomly initialized model.

We also compare our figure with figure1 that the authors provided to verify the quantitative claim.

***
:::

::: {.cell .markdown}
## Claim 2: The test accuracy of the warm-started model is lower than that of the randomly initialized model across various datasets, optimizers and model architectures.

The authors support their claim using different model architectures and datasets. They train these models with SGD and Adam optimizers until they reach 99% training accuracy or stop improving for a certain number of epochs. Their results are presented in the table below.

| CIFAR-10    | ResNet-SGD | ResNet-Adam | MLP-SGD | MLP-Adam | CIFAR-100   | ResNet-SGD | ResNet-Adam | MLP-SGD | MLP-Adam |    SVHN     | ResNet-SGD | ResNet-Adam | MLP-SGD | MLP-Adam |
| :---------: | :--------: | :---------: |:------: |:-------: | :---------: | :--------: | :---------: |:------: |:-------: | :---------: | :--------: | :---------: |:------: |:-------: |
| Random init |     56.2    |     78.0     |   39.0   |   39.4    |  |    18.2     |     41.4     |   10.3   |    11.6   |   |  89.4      |   93.6      |    76.5 |  76.7    |
| Warm-Start  |    51.7     |    74.4      |  37.4    |    36.1   |   |     15.5    |     35.0     |   9.4   |    9.9   |   |  87.5      |     93.5    |   75.4  |   69.4   |
| Difference  |      4.5   |     3.6     |    1.6  |   3.3    |   |    2.7     |     6.4     |    0.9  |  1.7     |   |      1.9   |    0.1      |   1.1   |    7.3   |


To test the qualitative claim, we will train ResNet-18 and MLP models on CIFAR-10, CIFAR-100 and SVHN dataset using SGD and Adam optimizers. We stop the training when the models reach 99% training accuracy and measure their test accuracy to examine the generalization gap as claimed by the authors.

To verify the quantitative claim, we compare our numerical results with the previous table.

***
:::

::: {.cell .markdown}
## Claim 3: Warm-starting neural networks saves resources and time, but lowers test accuracy compared to random initialized models.

To support their claim, the authors conduct an online learning experiment. They train a model on a split dataset that starts with 1000 samples and grows by 1000 new samples at each iteration. They stop the training when the model achieves *99%* training accuracy and record the test accuracy and training time. They conduct this experiment for warm-starting and random initialization at each iteration. Their results are shown in the figure below.

![](assets/claim3.png)\

To evaluate the overall qualitative claim we conduct an online learning experiment. We use a ResNet-18 model on splited CIFAR-10 dataset. The dataset start with 1000 samples and is updated with 1000 new samples of data at each iteration. We train the model until it reaches 99% training accuracy and measure the test accuracy and training time.

We also compare the our results with the figure to verify the quantitative claim.

***
:::

::: {.cell .markdown}
The paper proposes more claims but we will focus on the mentioned claims. The other claims include a solution to the generalization gap due to warm-starting.

**Can you identify one of these claims and provide the experiment they did to support their claim?** 🧐
:::