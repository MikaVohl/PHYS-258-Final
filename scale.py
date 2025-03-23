import numpy as np
import matplotlib.pyplot as plt

# Data arrays for different positions on the scale (each row is a trial)
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

# Expected mass values for each measurement condition
expected = np.array([0, 1, 10, 11])  # [no mass, 1g, 10g, 1g+10g]

# Group the data by position
positions = {
    "center": center,
    "bottom left": bottleft,
    "bottom right": bottright,
    "top left": topleft,
    "top right": topright
}

# Prepare a dictionary to store analysis results
analysis_results = {}

# Compute mean and standard deviation for each position
for pos, data in positions.items():
    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)
    analysis_results[pos] = {"mean": mean_val, "std": std_val}

# Plot the measurements with error bars for each position
fig, ax = plt.subplots(figsize=(10, 6))
markers = {"center": "o", "bottom left": "s", "bottom right": "^", "top left": "D", "top right": "*"}

for pos, result in analysis_results.items():
    mean_val = result["mean"]
    std_val = result["std"]
    ax.errorbar(expected, mean_val, yerr=std_val, fmt=markers[pos],
                linestyle='--', capsize=5, label=pos)

# Plot the ideal line (y = x) for reference
ax.plot(expected, expected, 'k-', linewidth=2, label="Ideal")

ax.set_xlabel("Expected Mass (g)")
ax.set_ylabel("Measured Mass (g)")
ax.set_title("Scale Accuracy and Precision Analysis")
ax.legend()
ax.grid(True)
plt.show()

# Print out detailed analysis including errors and a simple linear regression
print("Detailed Analysis:")
for pos, result in analysis_results.items():
    mean_val = result["mean"]
    std_val = result["std"]
    error = mean_val - expected  # deviation from the expected mass
    # Perform a linear regression (fit: measured = slope * expected + intercept)
    coeffs = np.polyfit(expected, mean_val, 1)
    slope, intercept = coeffs[0], coeffs[1]
    print(f"\nPosition: {pos}")
    print(f"  Mean measurements: {mean_val}")
    print(f"  Standard deviations: {std_val}")
    print(f"  Error (measured - expected): {error}")
    print(f"  Linear fit: slope = {slope:.3f}, intercept = {intercept:.3f}")