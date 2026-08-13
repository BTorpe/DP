import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 50_000

# Education
p_e = [0.40, 0.35, 0.25]
S_e = [1, 3, 5]
h_e0 = [1.00, 1.20, 1.55]
delta_e = [0.010, 0.020, 0.030]

# Human capital
delta = 0.06
sigma_psi = 0.10

# Labour market
lambda_ = 0.60
sigma = 0.05

# Income
y_SU = 0.45
rho = 0.60
y_floor = 0.35

rng = np.random.default_rng(123)

education = rng.choice(
    [0, 1, 2],
    size=N,
    p=p_e)

education[0] = 2

ages = np.arange(18, 66)
print(education[:10])
print(ages)
