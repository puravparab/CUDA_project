import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Read the performance data
data = pd.read_csv('data/results.csv')

# Set seaborn style
sns.set_theme(style="white")
figure_size = (10, 6)

# 1. Time comparison plot
plt.figure(figsize=figure_size)
sns.lineplot(data=data, x='Matrix_Size', y='CPU_Time_ms', marker='o', label='CPU')
sns.lineplot(data=data, x='Matrix_Size', y='GPU_Total_Time_ms', marker='o', label='GPU')
# sns.lineplot(data=data, x='Matrix_Size', y='GPU_Kernel_Time_ms', marker='o', label='GPU Kernel')
plt.xlabel('Matrix size (N X N)')
plt.ylabel('Execution time (ms)')
plt.title('CPU vs GPU Execution Time ')
plt.yscale('log')
plt.savefig('images/execution_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Memory transfer breakdown
plt.figure(figsize=figure_size)
sns.lineplot(data=data, x='Matrix_Size', y='GPU_H2D_Time_ms', marker='o', label='Host to Device')
sns.lineplot(data=data, x='Matrix_Size', y='GPU_D2H_Time_ms', marker='o', label='Device to Host')
plt.xlabel('Matrix Size')
plt.ylabel('Time (ms)')
plt.title('Memory Transfer Times')
plt.savefig('images/memory_transfer_times.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. GFLOPS Comparison
plt.figure(figsize=figure_size)
sns.lineplot(data=data, x='Matrix_Size', y='CPU_GFLOPS', marker='o', label='CPU')
sns.lineplot(data=data, x='Matrix_Size', y='GPU_GFLOPS', marker='o', label='GPU')
plt.xlabel('Matrix Size')
plt.ylabel('Performance (GFLOPS)')
plt.title('Computational Performance Comparison')
plt.yscale('log')
plt.savefig('images/gflops_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Numerical Accuracy
plt.figure(figsize=figure_size)
sns.lineplot(data=data, x='Matrix_Size', y='Max_Difference', marker='o', color='red')
plt.xlabel('Matrix Size')
plt.ylabel('Maximum Absolute Difference')
plt.title('CPU vs GPU Result Accuracy')
plt.yscale('log')
plt.savefig('images/numerical_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots have been saved in the 'images' directory.")