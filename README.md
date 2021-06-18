# PyHopper - High-dim hyperparameter tuning made easy 

![ci_badge](https://github.com/PyHopper/PyHopper/actions/workflows/continuous_integration.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/pyhopper/badge/?version=latest)](https://pyhopper.readthedocs.io/en/latest/?badge=latest) ![pyversion](docs/img/pybadge.svg)
![PyPI version](https://img.shields.io/pypi/v/pyhopper)
![downloads](https://img.shields.io/pypi/dm/pyhopper)


PyHopper is a hyperparameter optimizer, made specifically for high-dimensional problems arising in deep learning.

```bash
pip3 install -U pyhopper
```

PyHopper is lightweight, rich in features, and requires minimal changes to existing code

```python
import pyhopper

def objective(params: dict) -> float:
    model = build_model(params["hidden_size"],...)
    # .... train and evaluate the model
    return val_accuracy

search = pyhopper.Search(
    {
        "hidden_size": pyhopper.int(100,500),
        "dropout_rate": pyhopper.float(0,0.4),
        "opt": pyhopper.choice(["adam","rmsprop","sgd"]),
    }
)
best_params = search.run(objective, "maximize", "1h 30min", n_jobs="per-gpu")
```

Its most important features are

- runs parallel on multiple GPUs
- natively supports NumPy array parameters with millions of dimensions
- is highly customizable (e.g. you can directly tune entire ```torch.Tensor``` hyperparameters)

Under its hood, PyHopper uses an efficient 2-stage Markov chain Monte Carlo (MCMC) optimization algorithm.

![alt](docs/img/sampling.webp)

For more info, check out [PyHopper's documentation](https://pyhopper.readthedocs.io/)