# Naive Bayes Classifier

A naive Bayes classifier example with different prior probabilities.

## Table of Contents

- [Background](#background)
- [Usage](#usage)
- [License](#license)

## Background

This example use 3 datasets from UCI datasets, which are:
- Glass Identification 
- Hepatitis
- Image Segmentation

And 3 different prior probabilities, which are:
- Lapalce's Estimate 
- Dirichlet Distribution
- Generalized Dirichlet Distribution

The expected accuracy of the 3 priors is estimated by 5-fold cross validation. Continuous attributes are discretized with ten-bin discretization (i.e. equal-width).

## Usage

- Files start with 'snb': Naive Bayes classifier with Laplace's estimate and selective naive Bayes to rank the atrributes. The ranking will be stored as txt file in the ranked_attr directory.
- Files start with 'dirichlet': Naive Bayes classifier with Dirichlet priors. Parameters tested from 1 to 50 and the testing sequence of attibutes is order by the snb result.
- Files start with 'gdirichlet': Naive Bayes classifier with generalDirichlet priors. Parameters tested from 1 to 50 and the testing sequence of attibutes is order by the snb result.
- algorithms_evaluation: Tests if the performance results with different algorithms are statistically significant. Use the matched sample method from Wong, T. T. ,2015 (Performance evaluation of classification algorithms by k-fold and leave-one-out cross validation). Generally speaking, with a confidence interval of 95%, the performances are significantly defferent only if the t-value is greater 2.776.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)