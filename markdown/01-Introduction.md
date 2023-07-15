::: {.cell .markdown}
## Introduction

Warm starting and cold starting are two different ways of initializing the weights of a neural network before training. <b>Cold starting</b> means starting with random weights, while <b>warm starting</b> means starting with weights copied from a previously trained model. In our context the model is previously trained on a subset of the same dataset.

Updating datasets over time can be a costly endeavor, making it impractical to retrain models from scratch each time. Therefore, warm-starting becomes crucial as it allows leveraging pre-trained weights on a subset of the data, significantly reducing the time and resources required for training. By utilizing warm-starting, models can be efficiently adapted to incorporate new data without incurring the high computational expenses associated with starting from scratch.
:::

::: {.cell .markdown}
If you are using colab click on this link to go to the next notebook: [Open in Colab](https://colab.research.google.com/github/mohammed183/ml-reproducibility-p1/blob/main/notebooks/02-Claims.ipynb)
:::