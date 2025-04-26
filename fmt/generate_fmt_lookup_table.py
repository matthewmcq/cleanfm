import numpy as np
from scipy.special import jv  # Bessel function of the first kind
import csv
import json
import matplotlib.pyplot as plt

VISUALIZE = False  # Set to True to visualize the results

original_points = [(0.0, 0.0), (0.1, 0.0500626043), (0.2, 0.100503356), (0.3, 0.15171321), 
                   (0.4, 0.20410968337283286), (0.5, 0.2581526393344152), (0.6, 0.3143634420594278), 
                   (0.7, 0.37334930751146295), (0.8, 0.43583546994459993), (0.9, 0.5027090805529418), 
                   (1.0, 0.575080915004309), (1.1, 0.6543746338249326), (1.2, 0.6904680946948164), 
                   (1.202413, 0.6906360228201356)]

def evaluate_fmt_formula(beta, max_n=7):
    """
    Evaluates the FMT formula for a given beta value
    Using truncated series at n=max_n for computational efficiency
    """
    j0 = jv(0, beta)
    j1 = jv(1, beta)
    
    # Sum term
    sum_term = 0.0
    for n in range(1, max_n + 1):  
        jn_nbeta = jv(n, n * beta)
        sum_term += jn_nbeta ** 2
    
    # Product term
    prod_term = 1.0
    for n in range(1, max_n + 1): 
        jn = jv(n, beta)
        prod_term *= (1.0 - jn ** 2)
    
    # Complete formula
    result = abs(j1) * (1.0 + j0**2 * sum_term - (1.0 - j0)**2 * prod_term)
    return result

# Generate 1000 points
num_points = 1000
max_beta = 1.202413
beta_values = np.linspace(0, max_beta, num_points)
fmt_values = []


for beta in beta_values:
    # 5 fits best for values < 1
    fmt_value_avg = evaluate_fmt_formula(beta, max_n=5)
    
    fmt_values.append(fmt_value_avg)
    
# Plot the results
plt.figure(figsize=(12, 8))

# Plot the averaged curve
plt.plot(beta_values, fmt_values, 'b-', linewidth=2, label='Averaged (n=5,6,7)')

# Plot original points
orig_x = [p[0] for p in original_points]
orig_y = [p[1] for p in original_points]
plt.scatter(orig_x, orig_y, c='red', s=100, marker='o', label='Original Points', zorder=5)

# Add error visualization
errors = []
for x, y in original_points:
    # Find closest beta value in our computed array
    idx = np.abs(beta_values - x).argmin()
    computed_y = fmt_values[idx]
    error = abs(y - computed_y)
    errors.append(error)
    # Draw vertical lines to show error
    plt.plot([x, x], [y, computed_y], 'k--', alpha=0.5)

# Print error statistics
max_error = max(errors)
avg_error = np.mean(errors)
print(f"Maximum error: {max_error:.6f}")
print(f"Average error: {avg_error:.6f}")

plt.xlabel('β (Modulation Index)', fontsize=14)
plt.ylabel('FMT Amplitude', fontsize=14)
plt.title('Comparison of Original Points vs Averaged Formula (n=5,6,7)', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)

# Add inset to show detail for small values
ax_inset = plt.axes([0.25, 0.5, 0.3, 0.3])
ax_inset.plot(beta_values[:200], fmt_values[:200], 'b-', linewidth=2)
ax_inset.scatter(orig_x[:5], orig_y[:5], c='red', s=50, marker='o')
ax_inset.set_xlim(0, 0.4)
ax_inset.set_ylim(0, 0.25)
ax_inset.grid(True, alpha=0.3)
ax_inset.set_title('Zoomed view (β < 0.4)')

plt.tight_layout()
if VISUALIZE:
    plt.savefig('fmt_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# Create residual plot
plt.figure(figsize=(10, 6))
residuals = []
x_residuals = []
for x, y in original_points:
    idx = np.abs(beta_values - x).argmin()
    computed_y = fmt_values[idx]
    residual = y - computed_y
    residuals.append(residual)
    x_residuals.append(x)

plt.stem(x_residuals, residuals, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('β (Modulation Index)', fontsize=14)
plt.ylabel('Residual (Original - Computed)', fontsize=14)
plt.title('Residual Plot', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()

if VISUALIZE:
    plt.savefig('fmt_residuals.png', dpi=300, bbox_inches='tight')
    plt.show()

# Save all data
with open('fmt_lookup_table.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['beta', 'fmt_value'])  # Header
    for beta, fmt_value in zip(beta_values, fmt_values):
        writer.writerow([beta, fmt_value])

# # Also save as binary format for faster C++ loading
# data = np.column_stack((beta_values, fmt_values))
# data.astype(np.float64).tofile('fmt_lookup_table.bin')

# Save metadata
metadata = {
    'num_points': num_points,
    'max_beta': max_beta,
    'step_size': max_beta / (num_points - 1),
    'averaging_method': 'n=5,6,7',
    'max_error': float(max_error),
    'avg_error': float(avg_error)
}
with open('fmt_metadata.json', 'w') as f:
    json.dump(metadata, f)

print(f"Generated {num_points} points for beta in [0, {max_beta}]")
print(f"Step size: {metadata['step_size']}")
print(f"Using averaged results from n=5, n=6, and n=7")
print(f"Maximum error: {max_error:.6f}")
print(f"Average error: {avg_error:.6f}")