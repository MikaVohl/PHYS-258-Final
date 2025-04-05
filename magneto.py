import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('258-Magneto.csv', delimiter=',')
data = data[~np.isnan(data).any(axis=1)]

columns = ["Initial Mass (g)", "Final Mass (g)", "Height (mm)", "Voltage Drop (mV)", "Supply Current (A)"]

initial_mass = data[:, 0]
final_mass = data[:, 1]
height = data[:, 2]  # in mm
voltage_drop = np.abs(data[:, 3])
supply_current = data[:, 4]

# Constants
SHUNT_RESISTANCE = 0.0009521
g = 9.81  # m/s^2
l = 20e-2  # rod length in meters

# Calculate force (convert mass difference from grams to kg)
force = np.abs(final_mass - initial_mass) / 1000 * g  # Force in Newtons
current = voltage_drop / SHUNT_RESISTANCE  # Current in Amperes
print(current)

# Create a scatter plot with the color of each point set by the current value
scatter = plt.scatter(height, force, c=current, cmap='viridis', label='Data')
plt.xlabel('Height (mm)')
plt.ylabel('Force (N)')
plt.title('Force vs Height Colored by Current')
plt.grid(True)
plt.legend()

# Add a colorbar to show the mapping from colors to current values
cbar = plt.colorbar(scatter)
cbar.set_label('Current (A)')

plt.show()