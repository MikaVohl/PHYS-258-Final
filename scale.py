import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

center = np.array([
    [0, 1.00, 10.00, 11.00],  # trial 1
    [0, 1.00, 10.00, 11.00],
    [0, 1.00, 10.00, 11.05],
    [0, 1.00, 10.04, 11.06],
    [0, 0.99, 10.00, 11.05],
    [0, 1.00, 10.05, 11.07],
    [0, 1.00, 10.00, 11.05]
])
bottleft = np.array([
    [0, 1.01, 10.00, 11.05],
    [0, 1.01, 10.00, 11.05],
    [0, 0.99, 10.00, 11.05],
    [0, 1.00, 10.00, 11.00],
    [0, 1.02, 10.00, 11.07],
    [0, 1.00, 10.05, 11.04],
    [0, 1.00, 10.04, 11.05]
])
bottright = np.array([
    [0, 1.00, 10.00, 11.05],
    [0, 1.00, 10.00, 11.00],
    [0, 1.00, 10.05, 11.05],
    [0, 1.00, 10.01, 11.00],
    [0, 0.99, 10.05, 11.07],
    [0, 1.00, 10.05, 11.06],
    [0, 1.00, 10.05, 11.01]
])
topleft = np.array([
    [0, 1.01, 10.00, 11.05],
    [0, 1.00, 10.05, 11.04],
    [0, 1.00, 10.06, 11.06],
    [0, 1.00, 10.00, 11.04],
    [0, 1.00, 10.05, 11.05],
    [0, 0.99, 10.00, 11.00],
    [0, 1.00, 10.00, 11.05]
])
topright = np.array([
    [0, 1.01, 10.04, 11.07],
    [0, 1.00, 10.00, 11.00],
    [0, 1.03, 10.00, 11.00],
    [0, 1.02, 10.03, 11.00],
    [0, 1.00, 10.00, 11.00],
    [0, 1.00, 10.00, 11.00],
    [0, 1.00, 10.00, 11.00]
])

def linear(x, m, b):
    return m * x + b

# Expected (true) mass values for each measurement column
expected = np.array([0, 1, 10, 11])  # [no mass, 1g, 10g, 11g]

all_data = np.concatenate((center, bottleft, bottright, topleft, topright), axis=0)

# Create x (true mass) and y (scale reading) arrays
x_all = np.tile(expected, all_data.shape[0])
y_all = all_data.flatten()
error = y_all - x_all

# curve fit
popt, pcov = curve_fit(linear, x_all, error)
m, b = popt
error_fit = linear(x_all, *popt)
residuals = error - error_fit

def plot_scale():
    plt.figure(figsize=(10, 6))
    plt.scatter(x_all, error, label='Data Points', alpha=0.5)
    plt.plot(x_all, error_fit, 'r-', label='Fitted Line: y = {:.2f}x + {:.2f}'.format(m, b))
    plt.axhline(0, color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('True Mass (g)')
    plt.ylabel('Scale Reading (g)')
    plt.title('Scale Reading vs True Mass')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(x_all, residuals, label='Residuals', alpha=0.5)
    # plt.errorbar(x_all, residuals, yerr=0.1, fmt='o', label='Residuals', capsize=4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True Mass (g)')
    plt.ylabel('Residuals (g)')
    plt.title('Residuals of Scale Reading vs True Mass')
    plt.grid(True)
    plt.legend()
    plt.show()

def scale_to_true_mass_and_uncertainty(y, m, b, sigma_y, sigma_m, sigma_b, cov_mb):
    # Convert scale reading to true mass
    x = (y - b) / (1 + m)
    
    # Partial derivatives for uncertainty propagation:
    dx_dy = 1 / (1 + m)
    dx_dm = -(y - b) / (1 + m)**2
    dx_db = -1 / (1 + m)
    
    # Propagate uncertainties:
    sigma_x = np.sqrt(
        (dx_dy * sigma_y)**2 +
        (dx_dm * sigma_m)**2 +
        (dx_db * sigma_b)**2 +
        2 * dx_dm * dx_db * cov_mb
    )
    
    return x, sigma_x

convert_params = {
    'm': m,
    'b': b,
    'sigma_y': np.std(residuals),
    'sigma_m': np.sqrt(pcov[0, 0]),
    'sigma_b': np.sqrt(pcov[1, 1]),
    'cov_mb': pcov[0, 1]
}

def scale_to_true(y):
    # Convert scale reading to true mass
    x, sigma_x = scale_to_true_mass_and_uncertainty(y, **convert_params)
    return x, sigma_x