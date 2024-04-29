import pandas as pd
import numpy as np
import time

def simulate_vcf_data_cpu(num_records):
    chromosome_indices = np.random.randint(0, 4, num_records)
    chromosomes = ['chr1', 'chr2', 'chr3', 'chr4']
    positions = np.random.randint(1, 1000000, num_records)
    depths = np.random.randint(1, 100, num_records)
    qualities = np.random.randint(1, 100, num_records)
    allele_frequencies = np.random.random(num_records)

    chromosome_names = [chromosomes[i] for i in chromosome_indices]

    df = pd.DataFrame({
        'chromosome': chromosome_names,
        'position': positions,
        'depth': depths,
        'quality': qualities,
        'allele_frequency': allele_frequencies
    })
    return df

start_time = time.time()
df = simulate_vcf_data_cpu(100000000)
cpu_time = time.time() - start_time
print("CPU Data Simulation and Load Time: {:.2f} seconds".format(cpu_time))

# Filtering and Summary Statistics
start_time = time.time()
filtered_df = df[(df['depth'] >= 20) & (df['quality'] >= 50) & (df['allele_frequency'] >= 0.05)]
summary_stats_cpu = filtered_df.groupby('chromosome').agg({
    'depth': 'mean',
    'quality': 'mean',
    'allele_frequency': 'mean'
})
cpu_filter_time = time.time() - start_time
print("CPU Filtering and Summary Statistics Time: {:.2f} seconds".format(cpu_filter_time))
print(summary_stats_cpu)
