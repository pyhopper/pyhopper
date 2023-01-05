import numpy as np
import sys
import os

sys.path.insert(0, ".")
import pyhopper


def of(param):
    print(param)
    return np.random.default_rng().uniform(0,1)

# space = hp.choice('a', [ ('case 1', 1 + hp.lognormal('c1', 0, 1)), ('case 2', hp.uniform('c2', -10, 10)) ])


class ConditionalParameter(pyhopper.Parameter):
    def __init__(self, **kwargs):
        super().__init__()
        if len(kwargs) == 0:
            raise ValueError("Must pass at least one case in the form of a kw argument!")
        self.cases = list(kwargs.keys())
        self.values = kwargs
        self.initial_value = self.sample()

    def sample(self):
        k = np.random.default_rng().choice(self.cases)
        v = self.values[k]
        if isinstance(v,pyhopper.Parameter):
            v = v.sample()
        return k,v

    def mutate(self, value, temperature: float):
        switch_case = np.random.default_rng().choice([False, True])
        if switch_case:
            return self.sample()
        k,v = value
        if isinstance(self.values[k],pyhopper.Parameter):
            v = self.values[k].mutate(v,temperature)
        return k, v

if __name__ == "__main__":

    search = pyhopper.Search(
        {
            "lr": pyhopper.float(1e-5, 1e-2, fmt="0.1g"),
            # "case": ConditionalParameter(case1="abs",other_case=pyhopper.int(0,10))
            # "choice": pyhopper.choice("abs",pyhopper.int(0,10)),
            "cond": pyhopper.cases(case1="abc",other_case=pyhopper.int(0,10),third_case=["xyz",pyhopper.int(-10,0)])
        }
    )
    search.run(of, "max", steps=10, seeding_steps=5)
    search = pyhopper.Search(
        {
            "cond": pyhopper.cases(case1="abc",other_case=pyhopper.int(0,10),third_case=["xyz",pyhopper.int(-10,0)])
        }
    )
    search.run(of, "max", steps=10, seeding_steps=5)

    # try:
    #
    #     of(None)
    # except:
    #     etype, value, tb = sys.exc_info()
    #     e = "".join(format_exception(etype, value, tb, 4096))
    #     print(e)