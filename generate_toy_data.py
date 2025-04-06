import numpy as np
import pandas as pd

# Generate Toy Data
data = pd.DataFrame({
    'X1': np.random.randint(1, 11, 100),        # Integer values 1-10
    'X2': np.random.normal(50, 15, 100),        # Normal distribution (μ=50, σ=15)
    'X3': np.random.exponential(2, 100),        # Exponential distribution
    'X4': np.random.uniform(0, 100, 100),       # Uniform distribution 0-100
    'X5': np.abs(np.random.randn(100) * 10),    # Right-skewed positive values
    'X6': np.random.choice(['A', 'B', 'C', 'D'], 100)  # Categorical feature
})

# Create target using only numerical features
y = pd.Series(np.where(
    ((data['X1'] > 5) & (data['X2'] < 60)) |    # Rule 1
    ((data['X3'] > 2.5) & (data['X4'] < 40)) |  # Rule 2
    (data['X5'] > 8),                            # New rule replacing X6 dependency
    1, 0
))

# Add 15% noise to target
noise_mask = np.random.choice([True, False], 100, p=[0.15, 0.85])
y[noise_mask] = 1 - y[noise_mask]

# Reset index
data = data.reset_index(drop=True)
y = y.reset_index(drop=True)

# Save to CSV
data.to_csv('toy_data.csv', index=False)
y.to_csv('toy_target.csv', index=False)

print("Toy data has been generated and saved to 'toy_data.csv' and 'toy_target.csv'") 