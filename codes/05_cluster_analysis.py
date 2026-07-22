"""
STAGE 5: CLUSTER ANALYSIS & PROFILING
Analyze characteristics of each cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 5: CLUSTER ANALYSIS & PROFILING")
print("=" * 70)

# Load data
cleaned_data = pd.read_csv('02_cleaned_data.csv')
cluster_labels = np.load('cluster_labels.npy')

# Add cluster labels to data
df_with_clusters = cleaned_data.copy()
df_with_clusters['Cluster'] = cluster_labels

print(f"\nData loaded: {df_with_clusters.shape}")
print(f"Clusters: {np.unique(cluster_labels)}")

# STEP 1: Profile each cluster
print("\n1. CLUSTER PROFILES")
print("=" * 70)

cluster_names = {
    0: 'Cluster 0',
    1: 'Cluster 1',
    2: 'Cluster 2',
    3: 'Cluster 3'
}

key_features = ['Income', 'Age', 'TotalSpending', 'TotalPurchases', 'Recency',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'Kidhome', 'Teenhome', 'CampaignsAccepted']

for cluster in sorted(df_with_clusters['Cluster'].unique()):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
    n_customers = len(cluster_data)
    pct = n_customers / len(df_with_clusters) * 100

    print(f"\n{cluster_names[cluster]} - {n_customers} customers ({pct:.1f}%)")
    print("-" * 70)

    for feature in key_features:
        mean_val = cluster_data[feature].mean()
        std_val = cluster_data[feature].std()
        min_val = cluster_data[feature].min()
        max_val = cluster_data[feature].max()

        print(f"{feature:25s}: mean={mean_val:10.2f}, std={std_val:8.2f}, "
              f"range=[{min_val:8.2f}, {max_val:8.2f}]")

# STEP 2: Create comparison table
print("\n2. CLUSTER COMPARISON TABLE")
print("=" * 70)

comparison_data = []

for cluster in sorted(df_with_clusters['Cluster'].unique()):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]

    comparison_data.append({
        'Cluster': cluster,
        'Count': len(cluster_data),
        'Pct': f"{len(cluster_data)/len(df_with_clusters)*100:.1f}%",
        'Avg_Income': f"${cluster_data['Income'].mean():.0f}",
        'Avg_Age': f"{cluster_data['Age'].mean():.1f}",
        'Avg_Spending': f"${cluster_data['TotalSpending'].mean():.0f}",
        'Avg_Purchases': f"{cluster_data['TotalPurchases'].mean():.1f}",
        'Avg_Recency': f"{cluster_data['Recency'].mean():.0f}d",
        'Campaigns': f"{cluster_data['CampaignsAccepted'].mean():.2f}",
        'Web%': f"{cluster_data['NumWebPurchases'].sum()/cluster_data['TotalPurchases'].sum()*100:.0f}%"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# STEP 3: Create visualizations
print("\n3. CREATING COMPARISON VISUALIZATIONS")
print("-" * 70)

# Plot 1: Income by cluster
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

features_to_plot = ['Income', 'Age', 'TotalSpending', 'TotalPurchases', 'Recency', 'CampaignsAccepted']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]

    cluster_values = []
    cluster_nums = []

    for cluster in sorted(df_with_clusters['Cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster][feature]
        cluster_values.append(cluster_data)
        cluster_nums.append(f'C{cluster}')

    bp = ax.boxplot(cluster_values, labels=cluster_nums, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(feature, fontsize=11, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
    ax.set_title(f'{feature} Distribution by Cluster', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('06_cluster_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: 06_cluster_comparison.png")

# Plot 2: Heatmap of cluster characteristics
print("Creating cluster characteristics heatmap...")

heatmap_features = ['Income', 'Age', 'TotalSpending', 'TotalPurchases', 'Recency',
                    'NumWebPurchases', 'Kidhome', 'Teenhome', 'CampaignsAccepted']

heatmap_data = []
for cluster in sorted(df_with_clusters['Cluster'].unique()):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
    row = [cluster_data[feature].mean() for feature in heatmap_features]
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data,
                          index=[f'Cluster {i}' for i in sorted(df_with_clusters['Cluster'].unique())],
                          columns=heatmap_features)

# Normalize for better visualization
heatmap_normalized = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min())

fig, ax = plt.subplots(figsize=(14, 6))

im = ax.imshow(heatmap_normalized.values, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(len(heatmap_features)))
ax.set_xticklabels(heatmap_features, rotation=45, ha='right', fontweight='bold')
ax.set_yticks(range(len(heatmap_df)))
ax.set_yticklabels(heatmap_df.index, fontweight='bold')

ax.set_title('Cluster Characteristics Heatmap (Normalized)', fontsize=13, fontweight='bold', pad=20)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized Value', fontweight='bold')

# Add values
for i in range(len(heatmap_df)):
    for j in range(len(heatmap_features)):
        value = heatmap_df.values[i, j]
        color = 'white' if heatmap_normalized.values[i, j] > 0.5 else 'black'
        ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                color=color, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('06_cluster_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: 06_cluster_heatmap.png")

# Plot 3: Spending patterns by category
print("Creating spending patterns visualization...")

spending_features = ['MntWines', 'MntMeatProducts', 'MntFruits', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(spending_features))
width = 0.2

for i, cluster in enumerate(sorted(df_with_clusters['Cluster'].unique())):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
    spending_means = [cluster_data[feat].mean() for feat in spending_features]

    ax.bar(x + i*width, spending_means, width, label=f'Cluster {cluster}',
           color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

ax.set_xlabel('Product Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Spending ($)', fontsize=12, fontweight='bold')
ax.set_title('Product Spending by Cluster', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Wines', 'Meat', 'Fruits', 'Fish', 'Sweets', 'Gold'], rotation=0)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('06_spending_patterns.png', dpi=300, bbox_inches='tight')
print("Saved: 06_spending_patterns.png")

# STEP 4: Save results
print("\n4. SAVING RESULTS")
print("-" * 70)

# Save cluster assignments with original data
output_df = df_with_clusters.copy()
output_df.to_csv('customer_segments_final.csv', index=False)

# Save comparison table
comparison_df.to_csv('cluster_profiles.csv', index=False)

# Save heatmap data
heatmap_df.to_csv('cluster_characteristics.csv')

print("Saved: customer_segments_final.csv, cluster_profiles.csv, cluster_characteristics.csv")

print("\n" + "=" * 70)
print("STAGE 5 COMPLETE: Cluster Analysis Done")
print("All results saved!")
print("=" * 70)
