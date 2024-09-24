# Pairwise Functional Connectivity Estimation in Spinocerebellar Ataxia Type 3 Using Sparse Gaussian Markov Network: Integrating Group and Individual Analyses of rs-fMR 

## Abstract

Functional Connectivity (FC) patterns using rs-fMRI data are essential in understanding brain function. In this research, we present a representation for FC pattern of rs-fMRI data in the form of a sparse graphical model that characterizes the functional connectivity of brain networks. It is reasonable to introduce sparsity into a graphical model, as neurological processes typically involve specific brain regions interacting with only a few other regions. Current approaches enforce sparsity into activity patterns based on group-shared characteristics. These approaches consider the shared neural activity that exhibits similar FC patterns across all groups, known as shared FC outside the group. However, these approaches do not adjust the sparsity pattern based on the difference in neural activity between groups. We propose a three-step novel approach that simultaneously considers different sparsity patterns both outside the group and similar sparsity patterns inside each group with the consideration of individual variability.
We evaluate our approach by training a Sparse Gaussian Markov Network model for distinguishing between spinocerebellar ataxia patients and healthy controls. Our results show that our proposed method outperforms other state-of-the-art methods that use resting state functional connectivity. Moreover, as a product, our approach gives an explainable subset of FC patterns, which includes information on regions and connections that are conditionally independent and can be used for future studies of SCA3 disease.


## Contents

* The **src** folder contains the code to build both group-level and individual-level GMN of the dataset. 
* The **notebooks** folder contains the Jupyter Notebooks to build the GMN model of the dataset, as well as the exploratory data analysis of the data from the results.

## Installation

    git clone https://github.com/mfaezeh/gI_gmn.git
    cd gl_gmn
    pip install .    

To install the Python package

    pip install gl_gmn
## Full paper access
https://ieeexplore.ieee.org/document/10596860

## Cite This

F. Moradi, J. Faber and C. R. Hernandez-Castillo, "Pairwise Functional Connectivity Estimation in Spinocerebellar Ataxia Type 3 Using Sparse Gaussian Markov Network: Integrating Group and Individual Analyses of rs-fMRI," 2024 IEEE International Symposium on Medical Measurements and Applications (MeMeA), Eindhoven, Netherlands, 2024, pp. 1-6, doi: 10.1109/MeMeA60663.2024.10596860. keywords: {Training;Measurement;Graphics;Graphical models;Estimation;Machine learning;Brain modeling;Functional connectivity;Graphical LASSO;rs-fMRI;Gaussian Markov Network},



## License

The project is licensed under the MIT License.
