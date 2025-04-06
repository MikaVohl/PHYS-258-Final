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
mu_0 = 4 * np.pi * 1e-7
measured_current = voltage_drop / SHUNT_RESISTANCE # in Amperes

# Calculate force (convert mass difference from grams to kg)
force = np.abs(final_mass - initial_mass) / 1000 * g  # Force in Newtons

exp_mu = (force * 2 * np.pi * height * 1e-3) / (measured_current**2 * l)  # Experimental permeability

# Get the unique supply current values
unique_supply_currents = np.unique(supply_current)

def per_current():
    # Loop over each unique supply current and plot height vs force
    for sc in unique_supply_currents:
        # Create a mask to select rows with the current supply current value.
        # Using np.isclose to handle floating point comparisons.
        mask = np.isclose(supply_current, sc)
        
        plt.figure(figsize=(8,6))
        plt.scatter(height[mask], force[mask], label=f'Supply Current = {sc:.2f} A')
        x = np.linspace(min(height[mask]), max(height[mask]), 100)
        y = (mu_0 * sc**2 * l) / (2 * np.pi * x * 1e-3)  # Force in Newtons
        plt.plot(x, y, 'r-', label='Ideal Line')
        plt.xlabel('Height (mm)')
        plt.ylabel('Force (N)')
        plt.title(f'Force vs Height for Supply Current = {sc:.2f} A')
        plt.grid(True)
        plt.legend()
        plt.show()

def force_vs_height():
    plt.figure(figsize=(8,6))
    plt.scatter(height, force, c=measured_current,  label='Data Points')
    plt.xlabel('Height (mm)')
    plt.ylabel('Force (N)')
    plt.title('Force vs Height for All Currents')
    plt.grid(True)
    plt.legend()
    plt.show()

def force_vs_current():
    plt.figure(figsize=(8,6))
    plt.scatter(measured_current, force, label='Force vs Current')
    x = np.linspace(min(measured_current), max(measured_current), 100)
    y = (mu_0 * x**2 * l) / (2 * np.pi * height.mean() * 1e-3)  # Force in Newtons
    plt.plot(x, y, 'r-', label='Ideal Line')
    plt.xlabel('Current (A)')
    plt.ylabel('Force (N)')
    plt.title('Force vs Current')
    plt.grid(True)
    plt.legend()
    plt.show()

def mu_vs_current():
    plt.figure(figsize=(8,6))
    plt.scatter(measured_current, exp_mu, label='Experimental Permeability')
    plt.axhline(y=mu_0, color='r', linestyle='--', label='Theoretical Permeability (H/m)')
    plt.xlabel('Current (A)')
    plt.ylabel('Experimental Permeability (H/m)')
    plt.title('Experimental Permeability vs Current')
    plt.grid(True)
    plt.legend()
    plt.show()

per_current()
force_vs_height()
force_vs_current()
mu_vs_current()