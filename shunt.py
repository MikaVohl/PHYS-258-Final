import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

current = np.array([1.006, 1.498, 1.996, 2.493, 2.998, 3.494, 4.008, 4.489, 5.004, 5.499, 6.003, 6.500, 7.00, 7.50, 8.00, 8.50, 8.99, 9.50, 0.201, 0.300, 0.405, 0.499, 0.602, 0.702, 0.800, 0.905]) # Amps
voltage = np.array([0.9, 1.4, 1.8, 2.3, 2.8, 3.2, 3.7, 4.2, 4.7, 5.2, 5.6, 6.1, 6.6, 7.1, 7.5, 8.0, 8.5, 9.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) # milliVolts
current_uncertainty = [0.001 if cur < 7 else 0.01 for cur in current]
voltage_uncertainty = 0.1

# Define the linear function for curve fitting
def linear_model(B, x):
    return B[0] * x + B[1]

# Create the model for ODR
model = Model(linear_model)

# Package the data with uncertainties using RealData
data = RealData(current, voltage, sx=current_uncertainty, sy=voltage_uncertainty)

# Create the ODR object; initial guess for [Resistance, offset]
odr_instance = ODR(data, model, beta0=[1.0, 0.0])
output = odr_instance.run()

# Extract the fitted parameters
fitted_resistance, fitted_offset = output.beta
resistance_std_err, offset_std_err = output.sd_beta

fitted_resistance_ohm = fitted_resistance / 1000
resistance_std_err_ohm = resistance_std_err / 1000

print("Fitted resistance: {:.8f} Ω ± {:.8f} Ω".format(fitted_resistance_ohm, resistance_std_err_ohm))
print("Fitted offset: {:.7f} mV ± {:.7f} mV".format(fitted_offset, offset_std_err))

x = np.linspace(0, 10, 100)
y = linear_model(output.beta, x)
plt.plot(x, y, 'r-', label='Fitted line: y = {:.2f}x {:.2f}'.format(fitted_resistance, fitted_offset, '+d'), zorder=1)
plt.scatter(current, voltage, label='Data points', zorder=2)
plt.title('Voltage drop across shunt resistor at various currents')
plt.xlabel('Current (A)')
plt.ylabel('Voltage (mV)')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()
plt.legend()
plt.show()

residuals = voltage - linear_model(output.beta, current)
plt.errorbar(current, residuals, yerr=voltage_uncertainty, fmt='o', label='Residuals', capsize=4)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals of the fit')
plt.xlabel('Current (A)')
plt.ylabel('Residuals (mV)')
plt.show()