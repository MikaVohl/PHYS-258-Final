import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------
# Data Initialization
# ---------------------------
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

# Convert grams to kilograms
center *= 1e-3
bottleft *= 1e-3
bottright *= 1e-3
topleft *= 1e-3
topright *= 1e-3

# ---------------------------
# Define the Model Function
# ---------------------------
def linear(x, m, b):
    return m * x + b

# ---------------------------
# Prepare the Data (Individual Measurements)
# ---------------------------
# Expected (true) mass values for each measurement column (in kg)
expected = np.array([0, 1, 10, 11]) * 1e-3

# Combine data from different regions
all_data = np.concatenate((center, bottleft, bottright, topleft, topright), axis=0)

# Create arrays for the true mass (x_all) and scale reading (y_all)
x_all = np.tile(expected, all_data.shape[0])
y_all = all_data.flatten()

# Calculate the error: difference between scale reading and true mass
error_all = y_all - x_all

# ---------------------------
# Group-Averaging
# ---------------------------
# Get unique true mass values
unique_x = np.unique(x_all)

# For each unique true mass, compute the mean error and standard error (SEM)
group_mean_error = np.array([np.mean(error_all[x_all == ux]) for ux in unique_x])
min_uncertainty = 1e-7
group_sem_error = np.array([
    np.std(error_all[x_all == ux], ddof=1) / np.sqrt(np.sum(x_all == ux))
    for ux in unique_x
])
# Replace zeros (or values below min_uncertainty) with min_uncertainty
group_sem_error[group_sem_error < min_uncertainty] = min_uncertainty


# ---------------------------
# Linear Fit on Group-Averaged Data
# ---------------------------
# Use the group SEM as the sigma argument to weight the fit
popt, pcov = curve_fit(linear, unique_x, group_mean_error, sigma=group_sem_error, absolute_sigma=True)
m, b = popt
fit_line = linear(unique_x, m, b)
group_residuals = group_mean_error - fit_line

# Calculate the standard error of the fitted parameters
m_unc, b_unc = np.sqrt(np.diag(pcov))
print("Slope (m): {:.8e} ± {:.8e}".format(m, m_unc))
print("Intercept (b): {:.8e} ± {:.8e}".format(b, b_unc))

# ---------------------------
# Conversion Functions Based on the Group Fit
# ---------------------------
def scale_to_true_mass_and_uncertainty(y, m, b, sigma_y, sigma_m, sigma_b, cov_mb):
    """
    Converts a scale reading y into a true mass using the calibration:
       y = (1+m)*x + b,
    which is inverted to x = (y - b) / (1+m).
    
    Uncertainties are propagated using the partial derivative method.
    """
    # Calculate true mass from a scale reading
    x = (y - b) / (1 + m)
    
    # Partial derivatives for uncertainty propagation
    dx_dy = 1 / (1 + m)
    dx_dm = -(y - b) / (1 + m)**2
    dx_db = -1 / (1 + m)
    
    # Propagate uncertainties (including covariance between m and b)
    sigma_x = np.sqrt(
        (dx_dy * sigma_y)**2 +
        (dx_dm * sigma_m)**2 +
        (dx_db * sigma_b)**2 +
        2 * dx_dm * dx_db * cov_mb
    )
    return x, sigma_x

# For conversion, choose a representative uncertainty for the scale reading.
# Here we use the average SEM from the group-averaging as sigma_y.
convert_params = {
    'm': m,
    'b': b,
    'sigma_y': np.mean(group_sem_error),
    'sigma_m': np.sqrt(pcov[0, 0]),
    'sigma_b': np.sqrt(pcov[1, 1]),
    'cov_mb': pcov[0, 1]
}

def scale_to_true(y):
    """
    Converts a scale reading (or group-averaged reading) into a calibrated true mass.
    """
    x, sigma_x = scale_to_true_mass_and_uncertainty(y, **convert_params)
    return x, sigma_x

# print("Scale to True Mass Conversion Formula:")
# print("x = (y - b) / (1 + m)")
# print(convert_params)


# ---------------------------
# Plotting Functions Using Group-Averaged Data
# ---------------------------
def plot_scale():
    # Plot the group-averaged error with error bars corresponding to the SEM
    plt.figure(figsize=(7, 5))
    plt.errorbar(unique_x, group_mean_error, yerr=group_sem_error, fmt='o', capsize=4,
                 label='Error in Scale Reading')
    plt.plot(unique_x, fit_line, 'r-', 
             label='Fitted Line: y = {:.2e}x {:.2e}'.format(m, b, '+d'))
    plt.axhline(0, color='red', linestyle='--', label='Ideal Scale Reading')
    plt.xlabel('True Mass (kg)')
    plt.ylabel('Error (kg)')
    plt.title('Scale Reading Error vs True Mass')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the residuals for the group means
    plt.figure(figsize=(7, 5))
    plt.errorbar(unique_x, group_residuals, yerr=group_sem_error, fmt='o', capsize=4,
                 label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True Mass (kg)')
    plt.ylabel('Residual (kg)')
    plt.title('Residuals of Scale Reading Error vs True Mass')
    plt.grid(True)
    plt.legend()
    plt.show()

# plot_scale()