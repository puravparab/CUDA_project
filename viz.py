import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the performance data
data = pd.read_csv('detailed_performance.csv')

# Create multiple plots
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# Time comparison
axes[0,0].plot(data['MatrixSize'], data['CPU_Time'], 'b-o', label='CPU')
axes[0,0].plot(data['MatrixSize'], data['Total_GPU_Time'], 'r-o', label='GPU Total')
axes[0,0].plot(data['MatrixSize'], data['GPU_Kernel_Time'], 'g-o', label='GPU Kernel')
axes[0,0].set_xlabel('Matrix Size')
axes[0,0].set_ylabel('Time (ms)')
axes[0,0].set_title('Execution Time Comparison')
axes[0,0].legend()
axes[0,0].set_yscale('log')

# Memory transfer breakdown
axes[0,1].plot(data['MatrixSize'], data['Memory_H2D_Time'], 'b-o', label='Host to Device')
axes[0,1].plot(data['MatrixSize'], data['Memory_D2H_Time'], 'r-o', label='Device to Host')
axes[0,1].set_xlabel('Matrix Size')
axes[0,1].set_ylabel('Time (ms)')
axes[0,1].set_title('Memory Transfer Times')
axes[0,1].legend()

# Performance metrics
axes[1,0].plot(data['MatrixSize'], data['Memory_Bandwidth'], 'b-o')
axes[1,0].set_xlabel('Matrix Size')
axes[1,0].set_ylabel('GB/s')
axes[1,0].set_title('Memory Bandwidth')

# GFLOPS
axes[1,1].plot(data['MatrixSize'], data['GFLOPS'], 'r-o')
axes[1,1].set_xlabel('Matrix Size')
axes[1,1].set_ylabel('GFLOPS')
axes[1,1].set_title('Computational Performance')

plt.tight_layout()
plt.show()