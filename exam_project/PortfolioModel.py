""" a portfolio with a risky and a safe asset (Problem 3)

Starting point for the exam. The methods that were left with NotImplementedError
have been written here (trade, simulate, summary).

"""

from types import SimpleNamespace
import numpy as np

class PortfolioModelClass:
    """ a portfolio of a risky and a safe asset with a rebalancing rule """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        par = self.par = SimpleNamespace()

        # a. returns
        par.mu = 0.05 # mean log return on the risky asset
        par.sigma = 0.20 # standard deviation of the log return on the risky asset
        par.r = 0.01 # log return on the safe asset

        # b. the rebalancing rule
        par.theta_star = 0.50 # target share of wealth in the risky asset
        par.Delta = 0.10 # width of the no-trade band
        par.tau = 0.01 # proportional transaction cost

        # c. preferences
        par.gamma = 3.0 # relative risk aversion

        # d. simulation settings
        par.W0 = 1.0 # initial wealth
        par.T = 40 # number of periods
        par.N = 50_000 # number of simulated portfolios
        par.seed = 2026 # seed for the random number generator

        # e. overwrite with keyword arguments, e.g. PortfolioModelClass(Delta=0.0)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        par = self.par

        text = 'Portfolio model with:\n'
        text += f'  mu    = {par.mu:.4f}, sigma = {par.sigma:.4f}, r = {par.r:.4f}\n'
        text += f'  theta_star = {par.theta_star:.4f}, Delta = {par.Delta:.4f}, tau = {par.tau:.4f}\n'
        text += f'  gamma = {par.gamma:.4f} (relative risk aversion)\n'
        text += f'  W0 = {par.W0:.2f}, T = {par.T}, N = {par.N:,}, seed = {par.seed}'

        return text

    def draw_returns(self):
        """ draw the gross return on the risky asset in all periods and all portfolios

        Returns:

            (ndarray): gross returns with shape (N,T)

        """

        par = self.par

        rng = np.random.default_rng(par.seed)
        eps = rng.normal(size=(par.N,par.T))

        return np.exp(par.mu + par.sigma*eps)

    def u(self,W):
        """ CRRA utility of wealth """

        par = self.par

        return W**(1-par.gamma)/(1-par.gamma)

    def trade(self,theta):
        """ apply the no-trade band rule to a vector of pre-trade shares

        If |theta - theta_star| > Delta the portfolio is traded all the way back
        to the target; otherwise nothing is traded. The rule is applied to all
        N portfolios at once.

        Args:

            theta (ndarray): pre-trade risky share for each of the N portfolios

        Returns:

            theta_post (ndarray): risky share after trading
            traded (ndarray): amount traded, |theta_post - theta|

        """

        par = self.par

        # a. who is outside the band?
        outside = np.abs(theta-par.theta_star) > par.Delta

        # b. those trade back to the target, the rest stay put
        theta_post = np.where(outside,par.theta_star,theta)

        # c. the amount traded (zero for the ones inside the band)
        traded = np.abs(theta_post-theta)

        return theta_post,traded

    def simulate(self,R=None):
        """ simulate all N portfolios forward T periods

        The loop is over the T periods; everything is vectorized over the N
        portfolios. The investor starts at the target with wealth W0. In each
        period we trade according to the band rule, pay the transaction cost,
        realize the return, and update the risky share for next period.

        Args:

            R (ndarray,optional): gross returns with shape (N,T); if None a fresh
                set is drawn with draw_returns() so the seed is respected

        Returns:

            (SimpleNamespace): the simulated paths, also stored in self.sim

        """

        par = self.par
        sim = self.sim

        # a. returns
        if R is None: R = self.draw_returns()
        Rf = np.exp(par.r) # gross return on the safe asset, same every period

        # b. allocate memory, +1 in time so we can store the start-of-period share
        W = np.empty((par.N,par.T+1)) # wealth at the start of each period
        theta = np.empty((par.N,par.T+1)) # risky share at the start of each period
        dist = np.empty((par.N,par.T)) # |theta - theta_star| before trading
        n_trades = np.zeros(par.N) # number of trades over the horizon

        # c. initial conditions: everyone starts at the target
        W[:,0] = par.W0
        theta[:,0] = par.theta_star

        # d. loop forward in time
        for t in range(par.T):

            # i. distance to the target before trading
            dist[:,t] = np.abs(theta[:,t]-par.theta_star)

            # ii. trade back to the target if outside the band
            theta_post,traded = self.trade(theta[:,t])
            n_trades += (traded > 0) # count a trade whenever something was traded

            # iii. pay the transaction cost
            W_post = W[:,t]*(1-par.tau*traded)

            # iv. realize the return
            W[:,t+1] = theta_post*W_post*R[:,t] + (1-theta_post)*W_post*Rf

            # v. the risky share at the start of next period
            theta[:,t+1] = theta_post*W_post*R[:,t]/W[:,t+1]

        # e. store the results
        sim.W = W
        sim.theta = theta
        sim.dist = dist
        sim.n_trades = n_trades
        sim.WT = W[:,-1] # terminal wealth

        return sim

    def summary(self):
        """ the six numbers to report for a rule, including expected utility

        Assumes simulate() has already been run. Returns a dict so the numbers
        are easy to drop into a table.

        Returns:

            (dict): the six reported numbers

        """

        par = self.par
        sim = self.sim
        WT = sim.WT

        return {
            'trades': np.mean(sim.n_trades), # average number of trades
            'distance': np.mean(sim.dist), # average distance to the target
            'mean_WT': np.mean(WT), # mean of terminal wealth
            'median_WT': np.median(WT), # median of terminal wealth
            'p10_WT': np.percentile(WT,10), # 10th percentile of terminal wealth
            'EU': np.mean(self.u(WT)), # expected utility
        }
