import numpy as np
import matplotlib.pyplot as plt

# Load and clean the data
data = np.genfromtxt('258-Magneto.csv', delimiter=',')
data = data[~np.isnan(data).any(axis=1)]

# Define column indices based on your CSV structure:
# [Initial Mass (g), Final Mass (g), Height (mm), Voltage Drop (mV), Supply Current (A)]
initial_mass = data[:, 0]
final_mass = data[:, 1]
height = data[:, 2]  # in mm
voltage_drop = np.abs(data[:, 3]) * 1e-3  # Convert mV to V
supply_current = data[:, 4]

# Constants
SHUNT_RESISTANCE = 0.0009521
g = 9.81  # m/s^2
l = 20e-2  # rod length in meters

# Calculate force (convert mass difference from grams to kg)
force = np.abs(final_mass - initial_mass) / 1000 * g  # Force in Newtons

# Get the unique supply current values
unique_supply_currents = np.unique(supply_current)

# Loop over each unique supply current and plot height vs force
for sc in unique_supply_currents:
    # Create a mask to select rows with the current supply current value.
    # Using np.isclose to handle floating point comparisons.
    mask = np.isclose(supply_current, sc)
    
    plt.figure(figsize=(8,6))
    plt.scatter(height[mask], force[mask], label=f'Supply Current = {sc:.2f} A')
    plt.xlabel('Height (mm)')
    plt.ylabel('Force (N)')
    plt.title(f'Force vs Height for Supply Current = {sc:.2f} A')
    plt.grid(True)
    plt.legend()
    plt.show()