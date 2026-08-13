import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# Opgave 2.1 - Simulating an Income Distribution
# ============================================================

# ============================================================
# Parameters
# ============================================================

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


# ============================================================
# Random number generator
# ============================================================
rng = np.random.default_rng(123)


# ============================================================
# Education
# ============================================================
education = rng.choice([0, 1, 2], size=N,p=p_e)


# Education length for each individual
education_years = np.array(S_e)[education]


# Ages 18 to 65
ages = np.arange(18, 66)
T = len(ages)


# ============================================================
# Simulation arrays
# ============================================================

# State:
# 0 = education
# 1 = unemployed
# 2 = employed

state = np.zeros((N, T), dtype=int)

human_capital = np.zeros((N, T))

income = np.zeros((N, T))

# Stores the income from the person's most recent job
last_job_income = np.zeros(N)


# ============================================================
# Simulation
# ============================================================

for t, age in enumerate(ages):

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    # Individuals still in education
    in_education = age < 18 + education_years

    state[in_education, t] = 0
    income[in_education, t] = y_SU

    # Human capital is unchanged while in education
    human_capital[in_education, t] = np.array(h_e0)[education[in_education]]

    # --------------------------------------------------------
    # First year after education
    # --------------------------------------------------------

    # Individuals entering the labour market
    first_labour_year = age == 18 + education_years

    # Everyone enters the labour market as unemployed
    state[first_labour_year, t] = 1
    income[first_labour_year, t] = y_floor

    # Initial human capital after education
    human_capital[first_labour_year, t] = np.array(h_e0)[education[first_labour_year]]

    # --------------------------------------------------------
    # Labour market, human capital and income
    # --------------------------------------------------------

    if t > 0:

        # Individuals already in the labour market
        in_labour_market = ~in_education & ~first_labour_year


        # ----------------------------------------------------
        # Labour market transitions
        # ----------------------------------------------------

        # Labour market status in the previous period
        previous_state = state[in_labour_market, t - 1]

        # Random numbers for labour market transitions
        random_numbers = rng.random(in_labour_market.sum())

        # Unemployed people finding a job
        find_job = ((previous_state == 1) &(random_numbers < lambda_))

        # Employed people losing their job
        lose_job = ((previous_state == 2) &(random_numbers < sigma))

        # Start from previous labour market status
        current_state = previous_state.copy()

        # Unemployed -> employed
        current_state[find_job] = 2

        # Employed -> unemployed
        current_state[lose_job] = 1

        # Store current labour market status
        state[in_labour_market, t] = current_state


        # ----------------------------------------------------
        # Human capital shock
        # ----------------------------------------------------

        # Mean-one lognormal shock
        # This follows the hint in the assignment
        psi = rng.lognormal(-0.5 * sigma_psi**2, sigma_psi, size=N)


        # ----------------------------------------------------
        # Human capital
        # ----------------------------------------------------

        # Human capital from previous period
        previous_human_capital = human_capital[in_labour_market, t - 1]

        # Employed:
        # h(t+1) = h(t) * (1 + education-specific growth) * psi

        # Unemployed:
        # h(t+1) = h(t) * (1 - depreciation) * psi

        human_capital[in_labour_market, t] = np.where(current_state == 2, previous_human_capital* (1+ np.array(delta_e)[education[in_labour_market]])* psi[in_labour_market],previous_human_capital * (1 - delta)* psi[in_labour_market])


        # ----------------------------------------------------
        # Income
        # ----------------------------------------------------

        employed = current_state == 2

        income[in_labour_market, t] = np.where(employed,

            # Employed income
            human_capital[in_labour_market, t],

            # Unemployed income
            np.maximum(rho * last_job_income[in_labour_market],y_floor))


        # ----------------------------------------------------
        # Update last job income
        # ----------------------------------------------------

        # If employed, save current income as last job income.
        # If unemployed, keep the previous last job income.

        last_job_income[in_labour_market] = np.where(employed, income[in_labour_market, t], last_job_income[in_labour_market])



# ============================================================
# Opgave 2.2 - Simulate the income distribution
# ============================================================

# Check the simulation
# Education shares
education_shares = np.bincount(opgave2.education) / opgave2.N

print("Education shares:")
print(education_shares)

print("Target education shares:")
print(opgave2.p_e)


# Unemployment rate by age
unemployment_by_age = np.mean(opgave2.state == 1, axis=0)

print("Unemployment rate by age:")
print(unemployment_by_age)

print("Theoretical steady-state unemployment rate:")
print(opgave2.sigma / (opgave2.sigma + opgave2.lambda_))

# Mean income over the life cycle

mean_income = np.mean(income, axis=0)

# Income percentiles
p10 = np.percentile(income, 10, axis=0)
p25 = np.percentile(income, 25, axis=0)
p50 = np.percentile(income, 50, axis=0)
p75 = np.percentile(income, 75, axis=0)
p90 = np.percentile(income, 90, axis=0)


# Plot
plt.figure(figsize=(10, 6))

plt.plot(ages, mean_income, label="Mean", linewidth=2)
plt.plot(ages, p10, label="10th percentile")
plt.plot(ages, p25, label="25th percentile")
plt.plot(ages, p50, label="Median")
plt.plot(ages, p75, label="75th percentile")
plt.plot(ages, p90, label="90th percentile")

plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Income over the Life Cycle")
plt.legend()
plt.grid(True)
plt.xlim(18, 65)
plt.tight_layout()
plt.show()

# ============================================================
# Income distributions at different ages
# ============================================================

for age in [25, 35, 45, 60]:

    age_index = np.where(ages == age)[0][0]

    income_at_age = income[:, age_index]

    plt.figure(figsize=(8, 5))

    plt.hist(income_at_age, bins=30)

    plt.xlabel("Income")
    plt.ylabel("Number of individuals")
    plt.title(f"Income Distribution at Age {age}")

    plt.show()
# ============================================================
# Opgave 2.3 - Compute the Gini coefficient
# ============================================================

# ============================================================
# Opgave 2.4 - What drives inequality?
# ============================================================

# ============================================================
# Opgave 2.5 - Extension: More risk
# ============================================================