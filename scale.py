import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Data arrays for different positions
# -------------------------------
center = np.array([
    [0, 1.00, 10.00, 11.00],  # trial 1
    [0, 1.00, 10.00, 11.00]   # trial 2
])
bottleft = np.array([
    [0, 1.01, 10.00, 11.05],  # trial 1
    [0, 1.01, 10.00, 11.05]   # trial 2
])
bottright = np.array([
    [0, 1.00, 10.00, 11.05],
    [0, 1.00, 10.00, 11.00]
])
topleft = np.array([
    [0, 1.01, 10.00, 11.05],
    [0, 1.00, 10.05, 11.04]
])
topright = np.array([
    [0, 1.01, 10.04, 11.07],
    [0, 1.00, 10.00, 11.00]
])

# Expected (true) mass values for each measurement column
expected = np.array([0, 1, 10, 11])  # [no mass, 1g, 10g, 11g]

# -------------------------------
# Combine all measurement data (all positions)
# -------------------------------
# With 5 positions (each 2 trials) we have 10 measurements per mass.
all_data = np.concatenate((center, bottleft, bottright, topleft, topright), axis=0)  # shape (10, 4)

# Create x (true mass) and y (scale reading) arrays
x_all = np.tile(expected, all_data.shape[0])   # shape (40,)
y_all = all_data.flatten()                       # shape (40,)

# For center-only data:
x_center = np.tile(expected, center.shape[0])    # shape (8,)
y_center = center.flatten()

# -------------------------------
# Linear regression for calibration (reading vs true mass)
# -------------------------------
# All data:
coeffs_all = np.polyfit(x_all, y_all, 1)
slope_all, intercept_all = coeffs_all

# Center data:
coeffs_center = np.polyfit(x_center, y_center, 1)
slope_center, intercept_center = coeffs_center

print("Calibration (All Measurements): reading = {:.5f} * mass + {:.5f}".format(slope_all, intercept_all))
print("Calibration (Center-only): reading = {:.5f} * mass + {:.5f}".format(slope_center, intercept_center))

# Plot calibration for all data
plt.figure(figsize=(8, 6))
plt.scatter(x_all, y_all, label="All Data", color='blue', marker='o')
x_fit = np.linspace(expected.min(), expected.max(), 100)
plt.plot(x_fit, slope_all * x_fit + intercept_all, label=f"Fit: y = {slope_all:.3f}x + {intercept_all:.3f}", color='blue')
plt.xlabel("True Mass (g)")
plt.ylabel("Scale Reading (g)")
plt.title("Scale Calibration: All Measurements")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot calibration for center-only data
plt.figure(figsize=(8, 6))
plt.scatter(x_center, y_center, label="Center Data", color='red', marker='s')
plt.plot(x_fit, slope_center * x_fit + intercept_center, label=f"Fit: y = {slope_center:.3f}x + {intercept_center:.3f}", color='red')
plt.xlabel("True Mass (g)")
plt.ylabel("Scale Reading (g)")
plt.title("Scale Calibration: Center Measurements Only")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Uncertainty Analysis: Compute mean and standard deviation
# -------------------------------
# For all measurements, compute mean and sample standard deviation for each mass (column)
mean_all = np.mean(all_data, axis=0)
std_all = np.std(all_data, axis=0, ddof=1)  # ddof=1 for sample standard deviation

# For center-only data:
mean_center = np.mean(center, axis=0)
std_center = np.std(center, axis=0, ddof=1)

print("\nUncertainty (All Measurements):")
for m, mean_val, std_val in zip(expected, mean_all, std_all):
    print(f"Mass {m} g: Mean reading = {mean_val:.3f}, Std. Dev. = {std_val:.4f}")

print("\nUncertainty (Center Measurements):")
for m, mean_val, std_val in zip(expected, mean_center, std_center):
    print(f"Mass {m} g: Mean reading = {mean_val:.3f}, Std. Dev. = {std_val:.4f}")

# -------------------------------
# Fit an Equation for Uncertainty vs. Reading
# -------------------------------
# For the four calibration points (mean reading, std) we assume a linear model: sigma = a * reading + b.
# Fit for all data:
coeffs_unc_all = np.polyfit(mean_all, std_all, 1)
slope_unc_all, intercept_unc_all = coeffs_unc_all

# Fit for center-only data:
coeffs_unc_center = np.polyfit(mean_center, std_center, 1)
slope_unc_center, intercept_unc_center = coeffs_unc_center

print("\nUncertainty Equation (All Measurements):")
print("  sigma(y) = {:.5f} * y + {:.5f}".format(slope_unc_all, intercept_unc_all))

print("\nUncertainty Equation (Center Measurements):")
print("  sigma(y) = {:.5f} * y + {:.5f}".format(slope_unc_center, intercept_unc_center))

# -------------------------------
# Plot: Uncertainty vs. Mean Reading with Fitted Lines
# -------------------------------
plt.figure(figsize=(8, 6))
# Plot the data points for uncertainty
plt.errorbar(mean_all, std_all, fmt='o', label="All Data", color='blue', capsize=5)
plt.errorbar(mean_center, std_center, fmt='s', label="Center Data", color='red', capsize=5)
# Create a range for the reading values for plotting the fit lines
x_fit_unc = np.linspace(min(mean_all)-0.5, max(mean_all)+0.5, 100)
plt.plot(x_fit_unc, slope_unc_all * x_fit_unc + intercept_unc_all, '--', color='blue', 
         label="Fitted Uncertainty (All)")
plt.plot(x_fit_unc, slope_unc_center * x_fit_unc + intercept_unc_center, '--', color='red', 
         label="Fitted Uncertainty (Center)")
plt.xlabel("Mean Scale Reading (g)")
plt.ylabel("Uncertainty (Std. Dev.) (g)")
plt.title("Uncertainty vs. Mean Scale Reading")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Additional Plot: Residuals of the Calibration Fit
# -------------------------------
# Compute residuals for the calibration fit
residuals_all = y_all - (slope_all * x_all + intercept_all)
residuals_center = y_center - (slope_center * x_center + intercept_center)

plt.figure(figsize=(8, 6))
plt.scatter(x_all, residuals_all, label="All Data Residuals", color='blue', marker='o')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("True Mass (g)")
plt.ylabel("Residual (Measured - Fit)")
plt.title("Residuals: All Measurements")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x_center, residuals_center, label="Center Data Residuals", color='red', marker='s')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("True Mass (g)")
plt.ylabel("Residual (Measured - Fit)")
plt.title("Residuals: Center Measurements Only")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
