import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = [
    {
        'patch_batch_size': 1024,
        'num_workers': 6,
        'processing_times_s': [46, 52, 84],
        'avg_processing_time_s': (46 + 52 + 84) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 512,
        'num_workers': 6,
        'processing_times_s': [27, 43, 66],
        'avg_processing_time_s': (27 + 43 + 66) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 128,
        'num_workers': 6,
        'processing_times_s': [22, 35, 60],
        'avg_processing_time_s': (22 + 35 + 60) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 128,
        'num_workers': 2,
        'processing_times_s': [57, 93, 170],
        'avg_processing_time_s': (57 + 93 + 170) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 128,
        'num_workers': 0,
        'processing_times_s': [112, 190, 330],
        'avg_processing_time_s': (112 + 190 + 330) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 128,
        'num_workers': 12,
        'processing_times_s': [18, 30, 49],
        'avg_processing_time_s': (18 + 30 + 49) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 64,
        'num_workers': 12,
        'processing_times_s': [17, 27, 47],
        'avg_processing_time_s': (17 + 27 + 47) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 32,
        'num_workers': 12,
        'processing_times_s': [17, 27, 43],
        'avg_processing_time_s': (17 + 27 + 43) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 32,
        'num_workers': 6,
        'processing_times_s': [20, 34, 60],
        'avg_processing_time_s': (20 + 34 + 60) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    },
    {
        'patch_batch_size': 64,
        'num_workers': 6,
        'processing_times_s': [21, 35, 60],
        'avg_processing_time_s': (21 + 35 + 60) / 3,
        'hardware': 'NVIDIA L4 GPU, g2-standard-12 (12 vCPUs, 48 GB Memory)'
    }
]

df = pd.DataFrame(data)

# Create a pivot table for the heatmap
pivot_table = df.pivot_table(
    index='patch_batch_size',
    columns='num_workers',
    values='avg_processing_time_s'
)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".1f",
    cmap='viridis_r',
    linewidths=.5,
    cbar_kws={'label': 'Average Processing Time (s)'}
)

plt.title('Average Processing Time Heatmap', fontsize=16)
plt.xlabel('Number of Workers', fontsize=12)
plt.ylabel('Patch Batch Size', fontsize=12)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()

# 2025-08-06 15:45:42 | Average processing time per slide: 44.137 seconds