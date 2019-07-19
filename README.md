# Approximating Wasserstein distances with PyTorch

Repository for the blog post on [Wasserstein distances](https://dfdazac.github.io/sinkhorn.html).

***Update (July, 2019):*** I'm glad to see many people have found this post useful. Its main purpose is to introduce and illustrate the problem. To apply these ideas to large datasets and train on GPU, I highly recommend the <a href="http://www.kernel-operations.io/geomloss/index.html" target="_blank">GeomLoss</a> library, which is optimized for this.

**Instructions**

Create a conda environment with all the requirements (edit `environment.yml` if you want to change the name of the environment):

```sh
conda env create -f environment.yml
```

Activate the environment

```sh
source activate pytorch
```

Open the notebook to reproduce the results:


```sh
jupyter notebook
```
