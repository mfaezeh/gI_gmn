# Pairwise Functional Connectivity Estimation in Spinocerebellar Ataxia Type 3 Using Sparse Gaussian Markov Network: Integrating Group and Individual Analyses of rs-fMR 

## Abstract

Functional Connectivity (FC) patterns using rs-fMRI data are essential in understanding brain function. In this research, we present a representation for FC pattern of rs-fMRI data in the form of a sparse graphical model that characterizes the functional connectivity of brain networks. It is reasonable to introduce sparsity into a graphical model, as neurological processes typically involve specific brain regions interacting with only a few other regions. Current approaches enforce sparsity into activity patterns based on group-shared characteristics. These approaches consider the shared neural activity that exhibits similar FC patterns across all groups, known as shared FC outside the group. However, these approaches do not adjust the sparsity pattern based on the difference in neural activity between groups. We propose a three-step novel approach that simultaneously considers different sparsity patterns both outside the group and similar sparsity patterns inside each group with the consideration of individual variability.
We evaluate our approach by training a Sparse Gaussian Markov Network model for distinguishing between spinocerebellar ataxia patients and healthy controls. Our results show that our proposed method outperforms other state-of-the-art methods that use resting state functional connectivity. Moreover, as a product, our approach gives an explainable subset of FC patterns, which includes information on regions and connections that are conditionally independent and can be used for future studies of SCA3 disease.


## Contents

* The **data** folder contains the model parameters and architecture specifications to reconstruct the models for each language (this is created after running *download_data.py*).
* The **evaluate** folder contains the scripts to reproduce the evaluation results from the paper.
* The **lib** folder contains the code to use the sequence-to-sequence models to correct very long strings of characters, to compute the metrics used in the paper and the source code of the sequence-to-sequence models.
* The **notebooks** folder contains the Jupyter Notebooks to build the datasets required to train the sequence-to-sequence models, as well as the exploratory data analysis of the data from the [ICDAR 2019 competition](https://sites.google.com/view/icdar2019-postcorrectionocr).
* The **tests** folder contains scripts to test the installation of the repository.
* The **train** folder contains the scripts with hyper-parameters to train the models shown in the paper.
* The **tutorials** folder contains use cases on how to use the library.

## Installation

    git clone https://github.com/mfaezeh/gI_gmn.git
    cd gl_gmn
    pip install .
    
To download the datasets and models

    python download_data.py
    
To reproduce the results from the paper

    pip install -r requirements.txt
    cd notebooks

To install the Python package

    pip install gl_gmn


## License

The project is licensed under the MIT License.