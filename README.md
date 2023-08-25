# [Re] Warm-Starting Neural Network Training

This project is part of the [UCSC OSPO](https://ospo.ucsc.edu/) summer of reproducibility fellowship and aims to create an interactive notebook that can be used to teach undergraduate or graduate students different levels of reproducibility in machine learning education.

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
- torch
- torchvision

There are three options for installing and running the notebook:

### Option 1: Run Locally on Your Device

To run the notebook locally on your device, you'll need to install Python and Jupyter Notebook. Once you have those installed, follow these steps:

1. Clone the repository and navigate to the `re_warm_start_nn` directory by running the following command:
```
$ git clone https://github.com/mohammed183/re_warm_start_nn.git && cd re_warm_start_nn
```

2. Install the required packages by running this command:
```
$ pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Launch Jupyter Notebook by running this command:
```
$ jupyter notebook
```

4. In Jupyter Notebook, open the `Start_Here.ipynb` file located in the `notebooks` folder and follow the instructions.

### Option 2: Run on Chameleon Cloud

You can run the notebook on Chameleon Cloud using either a Colab frontend or a Jupyter Lab frontend. Both options are available in the `Reserve.ipynb` notebook. Follow the steps in that notebook to reserve an instance on Chameleon Cloud and run it with your desired frontend.

### Option 3: Run on Google Colab

You can also run the notebook on Google Colab. To open the `Start_Here.ipynb` file on Colab and navigate through the notebooks, click this button:<a target="_blank" href="https://colab.research.google.com/github/mohammed183/re_warm_start_nn/blob/main/Start_Here.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
