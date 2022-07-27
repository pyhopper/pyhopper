import gzip
import pickle
import numpy as np


def store(filename, obj):
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f)


def load(filename):
    with gzip.open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def abx(x):
    print("hello: ", x)


if __name__ == "__main__":
    test = {"a": 1, "b": np.array((0.1, 0.2))}
    print(test)

    store("example.zip", test)

    obj = load("example.zip")
    print(obj)

    store("example.zip", {"x": abx})
    obj = load("example.zip")
    print("abx=", obj["x"]("new"))

    store("example.zip", {"f": lambda x: 2 * x})
    obj = load("example.zip")
    print("3*2=", obj["f"](3))