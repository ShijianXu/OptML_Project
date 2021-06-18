# A Comparison of First-Order and Second-Order Optimization Methods
This repository contains the code and results for the OptML course project.


## Files included:
1. ``` run.py ```: The main function to conduct the experiments
2. ``` model.py ```: The implementation of Logistic Regression and MLP
3. ``` lbfgsnew.py ```: An improved LBFGS optimizer for PyTorch [here](https://github.com/nlesc-dirac/pytorch/blob/master/lbfgsnew.py)
4. ``` plot.py ```: Plot the results curves
5. ``` results/ ```: JSON files stroing the experiment results
6. ``` plots/  ```: Expereiment results figures
7. ``` report.pdf ```: Project report

## Requirement
1. Python 3.5+
2. PyTorch
3. Torchvision

## Reproduce
To reproduce our results, you can simply using the following command:

``` python run.py ```