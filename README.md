# [Re] Warm-Starting Neural Network Training

This project is part of the UCSC OSPO summer of reproducibility fellowship and aims to create an interactive notebook that can be used to teach undergraduate or graduate students different levels of reproducibility in machine learning education.

The project is based on the paper "[On Warm-Starting Neural Network Training](https://arxiv.org/abs/1910.08475)" by Jordan T. Ash and Ryan P. Adams, which was successfully replicated and published on [ReScience C](https://rescience.github.io/bibliography/Kireev_2021.html).

The paper investigates the problem of training neural networks on incremental data and shows that warm-starting, i.e., initializing the network with the weights from the previous training on a subset of the data, often leads to worse generalization performance than random initialization, even though the training losses are similar. The paper proposes a simple trick to overcome this problem, which involves shrinking and perturbing the weights before retraining. The paper demonstrates that this trick can close the generalization gap and reduce the training time in several scenarios.

The notebook will guide the students through the following steps:

- Introduce and explain the problem and the claims of the original paper
- Determine which claims can be tested using the available data
- Use the code and functions provided by the replication authors to test each claim
- Compare their results with the original paper and the replication paper
- Evaluate the reproducibility of the research

The notebook will use Python and PyTorch as the main tools and will require some basic knowledge of machine learning and neural networks.

## Installation

To run the notebook, you will need to install the following dependencies:

- Python 3.6 or higher
- PyTorch 1.7 or higher
- NumPy
- Matplotlib
- Jupyter Notebook or Jupyter Lab

You can install them using pip or conda, for example:

```
pip install -r requirements.txt
```

or

```
conda install --file requirements.txt
```
or to avoid being prompt for yes for every package
```
conda install --yes --file requirements.txt
```

## Usage

To run the notebook, you can clone this repository and launch Jupyter notebook from the project directory:

```
git clone https://github.com/mohammed183/ml-reproducibility-p1.git
cd ml-reproducibility-p1
make
jupyter notebook
```

Then, open the notebook file `warm_starting_nn_training.ipynb` and follow the instructions.

Alternatively, you can use Google Colab to run the notebook online without installing anything. Just click on this link: [Open in Colab](https://colab.research.google.com/github/mohammed183/ml-reproducibility-p1/blob/main/warm_starting_nn_training.ipynb)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
