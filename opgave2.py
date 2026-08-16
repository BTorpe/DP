import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# Opgave 2.1 - Simulating an Income Distribution
# ============================================================

# ============================================================
# Parameters
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


# ============================================================
# Random number generator
# ============================================================

rng = np.random.default_rng(123)


# ============================================================
# Simulation
# ============================================================

def simulate_model():

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    education = rng.choice([0, 1, 2], size=N, p=p_e)

    # Education length for each individual
    education_years = np.array(S_e)[education]

    # Ages 18 to 65
    ages = np.arange(18, 66)
    T = len(ages)


    # --------------------------------------------------------
    # Simulation arrays
    # --------------------------------------------------------

    # State:
    # 0 = education
    # 1 = unemployed
    # 2 = employed

    state = np.zeros((N, T), dtype=int)

    human_capital = np.zeros((N, T))

    income = np.zeros((N, T))

    # Income from the person's most recent job
    last_job_income = np.zeros(N)


    # --------------------------------------------------------
    # Simulation over the life cycle
    # --------------------------------------------------------

    for t, age in enumerate(ages):

        # ----------------------------------------------------
        # Individuals still in education
        # ----------------------------------------------------

        in_education = age < 18 + education_years

        state[in_education, t] = 0

        income[in_education, t] = y_SU

        # Human capital is unchanged while in education
        human_capital[in_education, t] = np.array(h_e0)[education[in_education]]


        # ----------------------------------------------------
        # First year after education
        # ----------------------------------------------------

        first_labour_year = age == 18 + education_years

        # Everyone enters the labour market unemployed
        state[first_labour_year, t] = 1

        income[first_labour_year, t] = y_floor

        # Initial human capital after education
        human_capital[first_labour_year, t] = np.array(h_e0)[education[first_labour_year] ]


        # ----------------------------------------------------
        # Individuals already in the labour market
        # ----------------------------------------------------

        if t > 0:

            in_labour_market = (~in_education & ~first_labour_year)


            # ------------------------------------------------
            # Labour market transitions
            # ------------------------------------------------

            previous_state = state[in_labour_market, t - 1]

            random_numbers = rng.random(in_labour_market.sum() )

            # Unemployed people finding a job
            find_job = ((previous_state == 1) & (random_numbers < lambda_))

            # Employed people losing their job
            lose_job = ((previous_state == 2)& (random_numbers < sigma))

            # Start from previous labour market status
            current_state = previous_state.copy()

            # Unemployed -> employed
            current_state[find_job] = 2

            # Employed -> unemployed
            current_state[lose_job] = 1

            # Store current labour market status
            state[in_labour_market, t] = current_state


            # ------------------------------------------------
            # Human capital shock
            # ------------------------------------------------

            psi = rng.lognormal(-0.5 * sigma_psi**2,sigma_psi,size=N)


            # ------------------------------------------------
            # Human capital
            # ------------------------------------------------

            previous_human_capital = human_capital[in_labour_market, t - 1]

            human_capital[in_labour_market, t] = np.where(current_state == 2,

                # Employed
                previous_human_capital* (1 + np.array(delta_e)[education[in_labour_market]])* psi[in_labour_market],

                # Unemployed
                previous_human_capital* (1 - delta)* psi[in_labour_market])


            # ------------------------------------------------
            # Income
            # ------------------------------------------------

            employed = current_state == 2

            income[in_labour_market, t] = np.where(employed,

                # Employed income
                human_capital[in_labour_market, t],

                # Unemployed income
                np.maximum(rho * last_job_income[in_labour_market],y_floor))


            # ------------------------------------------------
            # Update last job income
            # ------------------------------------------------

            last_job_income[in_labour_market] = np.where(employed, income[in_labour_market, t],last_job_income[in_labour_market])


    # ========================================================
    # Return results to make the code run easier when further codeing
    # ========================================================

    return (education, ages,state, human_capital, income)
# ============================================================
# Opgave 2.2 - Simulate the income distribution
# ============================================================
# Everything is in the notebook, so no code is needed here.

# ============================================================
# Opgave 2.3 - Compute the Gini coefficient
# ============================================================
# Everything is in the notebook, so no code is needed here.

# ============================================================
# Opgave 2.4 - What drives inequality?
# ============================================================
# New model with no educational differences, so that all individuals have the same education level and the same initial human capital. This will allow us to isolate the effects of labour market transitions and income shocks on inequality.

def simulate_no_education():

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    education = np.zeros(N, dtype=int)

    # Education length for each individual
    education_years = np.array(S_e)[education]

    # Ages 18 to 65
    ages = np.arange(18, 66)
    T = len(ages)


    # --------------------------------------------------------
    # Simulation arrays
    # --------------------------------------------------------

    # State:
    # 0 = education
    # 1 = unemployed
    # 2 = employed

    state = np.zeros((N, T), dtype=int)

    human_capital = np.zeros((N, T))

    income = np.zeros((N, T))

    # Income from the person's most recent job
    last_job_income = np.zeros(N)


    # --------------------------------------------------------
    # Simulation over the life cycle
    # --------------------------------------------------------

    for t, age in enumerate(ages):

        # ----------------------------------------------------
        # Individuals still in education
        # ----------------------------------------------------

        in_education = age < 18 + education_years

        state[in_education, t] = 0

        income[in_education, t] = y_SU

        # Human capital is unchanged while in education
        human_capital[in_education, t] = np.array(h_e0)[education[in_education]]


        # ----------------------------------------------------
        # First year after education
        # ----------------------------------------------------

        first_labour_year = age == 18 + education_years

        # Everyone enters the labour market unemployed
        state[first_labour_year, t] = 1

        income[first_labour_year, t] = y_floor

        # Initial human capital after education
        human_capital[first_labour_year, t] = np.array(h_e0)[education[first_labour_year] ]


        # ----------------------------------------------------
        # Individuals already in the labour market
        # ----------------------------------------------------

        if t > 0:

            in_labour_market = (~in_education & ~first_labour_year)


            # ------------------------------------------------
            # Labour market transitions
            # ------------------------------------------------

            previous_state = state[in_labour_market, t - 1]

            random_numbers = rng.random(in_labour_market.sum() )

            # Unemployed people finding a job
            find_job = ((previous_state == 1) & (random_numbers < lambda_))

            # Employed people losing their job
            lose_job = ((previous_state == 2)& (random_numbers < sigma))

            # Start from previous labour market status
            current_state = previous_state.copy()

            # Unemployed -> employed
            current_state[find_job] = 2

            # Employed -> unemployed
            current_state[lose_job] = 1

            # Store current labour market status
            state[in_labour_market, t] = current_state


            # ------------------------------------------------
            # Human capital shock
            # ------------------------------------------------

            psi = rng.lognormal(-0.5 * sigma_psi**2,sigma_psi,size=N)


            # ------------------------------------------------
            # Human capital
            # ------------------------------------------------

            previous_human_capital = human_capital[in_labour_market, t - 1]

            human_capital[in_labour_market, t] = np.where(current_state == 2,

                # Employed
                previous_human_capital* (1 + np.array(delta_e)[education[in_labour_market]])* psi[in_labour_market],

                # Unemployed
                previous_human_capital* (1 - delta)* psi[in_labour_market])


            # ------------------------------------------------
            # Income
            # ------------------------------------------------

            employed = current_state == 2

            income[in_labour_market, t] = np.where(employed,

                # Employed income
                human_capital[in_labour_market, t],

                # Unemployed income
                np.maximum(rho * last_job_income[in_labour_market],y_floor))


            # ------------------------------------------------
            # Update last job income
            # ------------------------------------------------

            last_job_income[in_labour_market] = np.where(employed, income[in_labour_market, t],last_job_income[in_labour_market])


    # ========================================================
    # Return results to make the code run easier when further codeing
    # ========================================================

    return (education, ages,state, human_capital, income)


# ============================================================
# Opgave 2.5 - Extension: More risk
# ============================================================