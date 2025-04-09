import numpy as np
import matplotlib.pyplot as plt
from scale import scale_to_true

# Load and clean the data
data = np.genfromtxt('258-Magneto.csv', delimiter=',')
data = data[~np.isnan(data).any(axis=1)]

# [Initial Mass (g), Final Mass (g), Height (mm), Voltage Drop (mV), Supply Current (A)]
initial_mass = data[:, 0] * 1e-3 # in kg
final_mass = data[:, 1] * 1e-3 # in kg
height = data[:, 2] * 1e-3 # in meters
voltage_drop = np.abs(data[:, 3]) * 1e-3  # Convert mV to V
supply_current = data[:, 4] # in Amperes

# Constants
PLATFORM_MASS = 43.54 * 1e-3 # kg
BASELINE_MASS = 102.62 * 1e-3 - PLATFORM_MASS # kg
SHUNT_RESISTANCE = 0.0009521 # in Ohms
SHUNT_RESISTANCE_UNC = 0.0000017 # in Ohms
PRINT_UNC = 0.2 * 1e-3 # in meters (0.4 mm nozzle size on 3D printer)
VOLTAGE_DROP_UNC = 0.1 * 1e-3 # in Volts (0.1 mV)

initial_mass, init_mass_unc = scale_to_true(initial_mass + BASELINE_MASS)
final_mass, final_mass_unc = scale_to_true(final_mass + BASELINE_MASS)
initial_mass -= BASELINE_MASS
final_mass -= BASELINE_MASS

height_unc = PRINT_UNC # Possibly consider height uncertainty being 0.5mm since thats the absolute max it could be off by
# height_unc = 0.5 * 1e-3 # in meters

g = 9.81  # m/s^2
l = 20 * 1e-2  # rod length in meters
l_unc = 0
# rod length uncertainty is negligible due to the higher uncertainty of other parameters
mu_0 = 4 * np.pi * 1e-7
measured_current = voltage_drop / SHUNT_RESISTANCE # in Amperes
measured_current_unc = np.sqrt(
    (VOLTAGE_DROP_UNC / SHUNT_RESISTANCE)**2 +
    (voltage_drop * SHUNT_RESISTANCE_UNC / SHUNT_RESISTANCE**2)**2
)  # Uncertainty in measured current

force = np.abs(final_mass - initial_mass) * g  # Force in Newtons
force_unc = np.sqrt(final_mass_unc**2 + init_mass_unc**2) * g  # Uncertainty in force


exp_mu = (force * 2 * np.pi * height) / (measured_current**2 * l)  # Experimental permeability
exp_mu_unc = exp_mu * np.sqrt(
    (force_unc / force)**2 +
    (height_unc / height)**2 +
    (measured_current_unc / measured_current)**2 +
    (l_unc / l)**2
)  # Uncertainty in experimental permeability
# Get the unique supply current values

print(np.mean(force), np.mean(force_unc))
print(np.mean(height), np.mean(height_unc))
print(np.mean(measured_current), np.mean(measured_current_unc))
print(np.mean(exp_mu), np.mean(exp_mu_unc))


unique_supply_currents = np.unique(supply_current)

def per_current():
    # Loop over each unique supply current and plot height vs force
    for sc in unique_supply_currents:
        # Create a mask to select rows with the current supply current value.
        # Using np.isclose to handle floating point comparisons.
        mask = np.isclose(supply_current, sc)
        
        plt.figure(figsize=(8,6))
        plt.errorbar(height[mask], force[mask], yerr=force_unc[mask], fmt='o', capsize=4, label=f'Supply Current = {sc:.2f} A')
        x = np.linspace(min(height[mask]), max(height[mask]), 100)
        y = (mu_0 * sc**2 * l) / (2 * np.pi * x)  # Force in Newtons
        plt.plot(x, y, 'r-', label='Ideal Line')
        plt.xlabel('Height (m)')
        plt.ylabel('Force (N)')
        plt.title(f'Force vs Height for Supply Current = {sc:.2f} A')
        plt.grid(True)
        plt.legend()
        plt.show()

def force_vs_height():
    plt.figure(figsize=(8,6))
    plt.scatter(height, force, c=measured_current,  label='Data Points')
    plt.xlabel('Height (m)')
    plt.ylabel('Force (N)')
    plt.title('Force vs Height for All Currents')
    plt.grid(True)
    plt.legend()
    plt.show()

def force_vs_current():
    plt.figure(figsize=(8,6))
    plt.scatter(measured_current, force, label='Force vs Current')
    x = np.linspace(min(measured_current), max(measured_current), 100)
    y = (mu_0 * x**2 * l) / (2 * np.pi * height.mean())  # Force in Newtons
    plt.plot(x, y, 'r-', label='Ideal Line')
    plt.xlabel('Current (A)')
    plt.ylabel('Force (N)')
    plt.title('Force vs Current')
    plt.grid(True)
    plt.legend()
    plt.show()

def mu_vs_current():
    plt.figure(figsize=(8,6))
    # plt.scatter(measured_current, exp_mu, label='Experimental Permeability')
    plt.errorbar(measured_current, exp_mu, yerr=exp_mu_unc, fmt='o', label='Experimental Permeability')
    plt.axhline(y=mu_0, color='r', linestyle='--', label='Theoretical Permeability (H/m)')
    plt.xlabel('Current (A)')
    plt.ylabel('Experimental Permeability (H/m)')
    plt.title('Experimental Permeability vs Current')
    plt.grid(True)
    plt.legend()
    plt.show()

def mu_vs_height():
    plt.figure(figsize=(8,6))
    # plt.scatter(height, exp_mu, label='Experimental Permeability')
    plt.errorbar(height, exp_mu, yerr=exp_mu_unc, fmt='o', label='Experimental Permeability')
    plt.axhline(y=mu_0, color='r', linestyle='--', label='Theoretical Permeability (H/m)')
    plt.xlabel('Height (m)')
    plt.ylabel('Experimental Permeability (H/m)')
    plt.title('Experimental Permeability vs Height')
    plt.grid(True)
    plt.legend()
    plt.show()

per_current()
# force_vs_height()
# force_vs_current()
mu_vs_current()
mu_vs_height()