"""Generate sample multivariate time series CSV for testing."""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000
dates = pd.date_range(start="2022-01-01", periods=n, freq="h")

t = np.arange(n)
# Feature 1: CPU usage with seasonal pattern
cpu = 50 + 10 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / 168) + np.random.normal(0, 2, n)
# Feature 2: Memory usage
mem = 60 + 8 * np.sin(2 * np.pi * t / 24 + 1) + np.random.normal(0, 1.5, n)
# Feature 3: Network traffic
net = 200 + 50 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 10, n)

# Inject anomalies
anomaly_indices = [300, 301, 302, 700, 701, 702, 1200, 1201, 1500, 1501, 1502, 1503]
cpu[anomaly_indices] += np.random.uniform(30, 50, len(anomaly_indices))
mem[anomaly_indices] += np.random.uniform(20, 35, len(anomaly_indices))
net[anomaly_indices] += np.random.uniform(200, 400, len(anomaly_indices))

df = pd.DataFrame({"timestamp": dates, "cpu_usage": cpu.round(2), "memory_usage": mem.round(2), "network_traffic": net.round(2)})
df.to_csv("/home/claude/ts_anomaly_app/sample_multivariate.csv", index=False)
print("Sample CSV created.")
print(df.head())
