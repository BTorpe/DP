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

# Simulation
T = len(ages)

state = np.zeros((N, T), dtype=int)
human_capital = np.zeros((N, T))
income = np.zeros((N, T))
last_job_income = np.zeros(N)

for t, age in enumerate(ages):

    # Individuals still in education
    in_education = age < 18 + education_years

    state[in_education, t] = 0
    income[in_education, t] = y_SU

    # First year after education: unemployed
    first_labour_year = age == 18 + education_years

    state[first_labour_year, t] = 1
    income[first_labour_year, t] = y_floor

# Initial human capital after education
    human_capital[first_labour_year, t] = np.array(h_e0)[
        education[first_labour_year]]

 # Individuals already in the labour market
    if t > 0:

        # Identify individuals already in the labour market
        in_labour_market = ~in_education & ~first_labour_year

        # Previous labour market status
        previous_state = state[in_labour_market, t - 1]

        # Random numbers for labour market transitions
        random_numbers = rng.random(in_labour_market.sum())

        # Unemployed people finding a job
        find_job = (
            (previous_state == 1) &
            (random_numbers < lambda_))

        # Employed people losing their job
        lose_job = (
            (previous_state == 2) &
            (random_numbers < sigma))

        # Update labour market status
        current_state = previous_state.copy()

        current_state[find_job] = 2
        current_state[lose_job] = 1

        state[in_labour_market, t] = current_state

        # Human capital shock
        psi = rng.lognormal(
            -0.5 * sigma_psi**2,
            sigma_psi,
            size=in_labour_market.sum())

        # Update human capital
        previous_human_capital = human_capital[
            in_labour_market, t - 1]

        human_capital[in_labour_market, t] = np.where(
            current_state == 2,
            previous_human_capital
            * (1 + np.array(delta_e)[education[in_labour_market]])
            * psi,
            previous_human_capital
            * (1 - delta)
            * psi)

        # Income
        employed = current_state == 2

        income[in_labour_market, t] = np.where(
            employed,
            human_capital[in_labour_market, t],
            np.maximum(
                rho * last_job_income[in_labour_market],
                y_floor))

        # Update last job income
        last_job_income[in_labour_market] = np.where(
            employed,
            income[in_labour_market, t],
            last_job_income[in_labour_market])

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