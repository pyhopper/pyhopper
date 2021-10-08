<div align="center"><img src="https://raw.githubusercontent.com/pyhopper/pyhopper/main/docs/img/banner_gray.png" width="800"/></div>

# PyHopper - Optimizing high-dimensional hyperparameters

![ci_badge](https://github.com/PyHopper/PyHopper/actions/workflows/continuous_integration.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/pyhopper/badge/?version=latest)](https://pyhopper.readthedocs.io/en/latest/?badge=latest) ![pyversion](docs/img/pybadge.svg)
![PyPI version](https://img.shields.io/pypi/v/pyhopper)
![downloads](https://img.shields.io/pypi/dm/pyhopper)

[**Website**](https://pyhopper.io)
| [**Docs**](https://pyhopper.readthedocs.io/)
| [**Quickstart**](https://pyhopper.readthedocs.io/en/latest/quickstart.html)
| [![Colab Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UPzhfCNCagh4OjI0VQyq87TpFbGoFBBl?usp=sharing)


PyHopper is a hyperparameter optimizer, made specifically for high-dimensional problems arising in machine learning research and businesses.

```bash
pip3 install -U pyhopper
```

PyHopper is lightweight, rich in features, and requires minimal changes to existing code

```python
import pyhopper

def my_objective(params: dict) -> float:
    model = build_model(params["hidden_size"],...)
    # .... train and evaluate the model
    return val_accuracy

search = pyhopper.Search(
    {
        "hidden_size": pyhopper.int(100,500),
        "dropout_rate": pyhopper.float(0,0.4),
        "matrix": pyhopper.float(-1,1,shape=(20,20)),
        "opt": pyhopper.choice(["adam","rmsprop","sgd"]),
    }
)
best_params = search.run(my_objective, "maximize", "1h 30min", n_jobs="per-gpu")
```

Its most important features are

- 1-line multi GPU parallelization
- native NumPy array hyperparameter support
- automatically focuses its search space based on the remaining runtime

Under its hood, PyHopper uses an efficient 2-stage Markov chain Monte Carlo (MCMC) optimization algorithm.

![alt](docs/img/sampling.webp)

For more info, check out [PyHopper's documentation](https://pyhopper.readthedocs.io/)





Copyright ©2018-2021. Mathias Lechner  
Copyright ©2018-2021. Institute of Science and Technology Austria (IST Austria)  
Copyright ©2021. Simple-AI  