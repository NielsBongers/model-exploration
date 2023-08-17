# Exploring common machine learning models 

I found myself re-implementing some code in several projects, so I decided to make it into an easy package, for model comparison and some limited exploratory data analysis. 

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

I've added a simple PCA feature, where you can pass a Pandas DataFrame, a list of features to consider (by default, it takes all of them, but throws an exception if there are strings etc.), the number of components for the PCA, and a feature to split the others by, if you want to plot them separately. 

```python
get_PCA(
    df=df,
    selected_features=["weight", "height", "width", "thickness"],
    n_components=2,
    split_feature="success",
    title="Example PCA",
    show_figure=True,
    save_figure=True,
)
```

The output is the explained variance ratio for the selected number of components, and for two components, a scatterplot of the selected split features, here "success". 

<img align="center" src="https://raw.githubusercontent.com/NielsBongers/model-overviews/main/Figures/PCA%20example.png" width="500"> 

## Build and installation 

The package can be built and installed using 

    py -m build --wheel
    py -m pip install model-overviews/dist/model_exploration-0.1.4-py3-none-any.whl
