# Exploring common machine learning models 

## Models 

Simple module that takes data and shows performance of some common machine learning models. Useful for initial data exploration and to see which models could be candidates for finetuning. Models included are: 

1. Logistic regression. 
2. Support vector machines with linear, radial basis function and sigmoid kernels. 
3. Random forest. 
4. XGBoost. 

An example of a result is shown below. 

<img align="center" src="https://raw.githubusercontent.com/NielsBongers/model-overviews/main/Figures/Classification%20example.png" width="500"> 

---

The function returns a dataframe with the different models and associated metrics. It turns out Pandas also supports outputting formated LaTeX (that could have saved me a lot of time...), so there is an additional flag to print this directly. 

## PCA 



## Build and installation 

The package can be built and installed using 

    py -m build --wheel
    py -m pip install model-overviews/dist/model_exploration-0.1.4-py3-none-any.whl
