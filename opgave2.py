import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# Opgave 2.1 - The model
# ============================================================
N = 50000

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

# Random number generator
rng = np.random.default_rng(123)

# Education
education = rng.choice(
    [0, 1, 2],
    size=N,
    p=p_e)

# Ages
ages = np.arange(18, 66)

# Education length for each individual
education_years = np.array(S_e)[education]




































# ============================================================
# Opgave 2.2 - Simulate the income distribution
# ============================================================


# ============================================================
# Opgave 2.3 - Compute the Gini coefficient
# ============================================================

# ============================================================
# Opgave 2.4 - What drives inequality?
# ============================================================

# ============================================================
# Opgave 2.5 - Extension: More risk
# ============================================================