import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba import jit
from numba import prange
from scipy.interpolate import CubicSpline

# zero yield curve function
# @njit
def P_0(t, yield_curve):
    return np.exp(-yield_curve(t) * t)

# instantaneous forward rate function
# @njit
def f(yield_curve, P_0, t, dt):
    return -(np.log(P_0(t+dt, yield_curve)) - np.log(P_0(t-dt, yield_curve))) / (2*dt)

# @njit
def theta(t, dt, alpha, sigma, yield_curve, P_0, f):
    return (1/alpha)*((f(yield_curve, P_0, t+dt, dt)-f(yield_curve, P_0, t-dt, dt))/(2*dt)) + f(yield_curve, P_0, t, dt) + (0.5*((sigma/alpha)**2)*(1 - np.exp(-2*alpha*t)))

# @njit
def hull_white_simulation(n_paths, n_steps, T, alpha, sigma, yield_curve, P_0, f, theta):
    dt = T/n_steps
    t_array = np.linspace(0, T, n_steps+1)
    r_paths = np.zeros((n_paths, n_steps+1))
    r_paths[:,0] = yield_curve(0)
    Z = np.random.normal(0, 1, (n_paths, n_steps))

    for i in range(1, n_steps+1):
        # Euler discretization scheme
        r_paths[:,i] = r_paths[:,i-1] + alpha*(theta(t_array[i-1], dt, alpha, sigma, yield_curve, P_0, f) - r_paths[:,i-1])*dt + sigma*np.sqrt(dt)*Z[:,i-1]

    return r_paths, t_array

def fit_yield_curve(maturities, yields, plot_fit = False):

    yield_curve = CubicSpline(maturities, yields)
    if plot_fit:
        maturities_dense = np.linspace(0, 30, 300)
        interpolated_yields = yield_curve(maturities_dense)
        plt.plot(maturities, yields, 'o', label='Original Yields')
        plt.plot(maturities_dense, interpolated_yields, '-', label='Spline Interpolation')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield rate')
        plt.title('Yield Curve Fitting using Spline Interpolation')
        plt.legend()
        plt.grid(True)
        plt.show()
    return yield_curve

def price_bonds(n_paths, n_steps, T, alpha, sigma, yield_curve, P_0, f, theta, face_value = 1, n_coupons=0, coupon_rate=0, plot_paths=False):
    dt = T/n_steps
    r_paths, t_array = hull_white_simulation(n_paths, n_steps, T, alpha, sigma, yield_curve, P_0, f, theta)
    if plot_paths:
        plt.plot(t_array, r_paths.T)
        plt.xlabel('Time')
        plt.ylabel('Short Rate')
        plt.title('Hull White Model Simulation')
        plt.show()
    cumsum = np.cumsum(r_paths[:,1:], axis=1)
    zero_coupon_bond_price = face_value * np.mean(np.exp(-np.sum(r_paths[:,1:], axis=1)*dt))
    if n_coupons == 0:
        duration = T
        return zero_coupon_bond_price, duration
    else:
        coupon_times = np.linspace(1/n_coupons, T, n_coupons*T)
        coupon_values = []
        for coupon_time in coupon_times:
            index = int(coupon_time/dt) - 1
            coupon_values.append(coupon_rate * face_value * np.mean(np.exp(-cumsum[:,index]*dt)))
        non_zero_coupon_price = np.sum(np.array(coupon_values)) + zero_coupon_bond_price
        # Fisher weil duration is calculated to account for yield curve changes.
        duration = np.sum((np.array(coupon_values) * np.array(coupon_times)) / non_zero_coupon_price) + T * zero_coupon_bond_price / non_zero_coupon_price

        return non_zero_coupon_price, duration

def verify_yield_curve(n_paths, n_steps, T, alpha, sigma, yield_curve, P_0, f, theta, maturities, yields):
    dt = T/n_steps
    r_paths, t_array = hull_white_simulation(n_paths, n_steps, T, alpha, sigma, yield_curve, P_0, f, theta)
    cumsum = np.cumsum(r_paths[:,1:], axis=1)
    present_values = np.mean(np.exp(-cumsum*dt), axis=0)
    yield_curve_output = np.log(present_values)/(-t_array[1:])
    plt.plot(maturities, yields, 'o', label='Original Yield Curve')
    plt.plot(t_array[1:], yield_curve_output, label='Simulated Yield Curve')
    plt.xlabel('Time')
    plt.ylabel('Yield')
    plt.title('Yield Curve from Hull White Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Hull White Model Simulation")
    n_paths = 10000 # number of paths
    n_steps = 1000 # number of discretzation steps
    T = 30 # time to maturity
    alpha = 0.1 # mean reversion rate
    sigma = 0.01 # volatility
    n_coupons = 1 # number of coupons per year
    coupon_rate = 0.06 # coupon rate
    dt = T/n_steps

    # yield curve data from the ECB website as of 02-08-2024
    maturities = np.array([0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    yields = np.array([0.03071224, 0.02807127, 0.0258073, 0.02388348, 0.01895377, 0.01721568, 0.01743347, 0.01875552, 0.0206068, 0.02260975, 0.0245264, 0.02621611, 0.02760507, 0.02866429, 0.02939393, 0.02981228, 0.02994802, 0.02983501, 0.02950876, 0.02900414, 0.02835402, 0.02758844, 0.02673428, 0.02581514, 0.02485141, 0.02386049, 0.02285699, 0.02185307, 0.02085866, 0.01988178, 0.01892876, 0.0180045, 0.01711269])
  
    # fit the yield curve
    yield_curve = fit_yield_curve(maturities, yields, plot_fit=True)

    # verify the yield curve from the simulation with the original yield rate data
    verify_yield_curve(n_paths, n_steps, T, alpha, sigma, yield_curve, P_0, f, theta, maturities, yields) 

    # verify_fit(n_paths, n_steps, T, alpha, sigma, P_0, f, theta)
    present_value, duration = price_bonds(n_paths, n_steps, T, alpha, sigma, yield_curve, P_0, f, theta, n_coupons=n_coupons, coupon_rate=coupon_rate, plot_paths=False)
    print(f"Present Value of Bond: {present_value}")
    print(f"Fisher Weil Duration of Bond: {duration}")



