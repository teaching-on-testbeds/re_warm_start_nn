::: {.cell .markdown}
# On Warm-Starting Neural Network Training

The paper is available on [arXiv](https://arxiv.org/abs/1910.08475). In creating the interactive material for this notebook, we utilized the code from this reproducibility challenge: [Re: Warm-Starting Neural Network Training](https://rescience.github.io/bibliography/Kireev_2021.html).
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
### To assess the reproducibility level of this paper, we need to answer some questions while experimenting:

-   Is there code available for both training and inference stages?
-   Is the code written by the authors themselves, or by someone else? Are there multiple implementations available for comparison?
-   What framework and version was used by the authors? Are all the functions still available or do we need to make some modifications?
-   Did the authors compare their model to other models that are not implemented in the code? Are these models available elsewhere?
-   Are all the hyperparameters for all the experiments clearly specified? If not, how sensitive is each hyperparameter to the performance?
-   Were the initial values set randomly or deterministically?
-   Are the datasets used by the authors accessible? Are there any preprocessing steps or modifications done to the data?
-   Did we obtain the same results as reported in the original paper?
:::


::: {.cell .markdown}
If you are using colab click on this link to go to the next notebook: 
<a target="_blank" href="https://colab.research.google.com/github/mohammed183/ml-reproducibility-p1/blob/main/notebooks/01-Introduction.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
:::