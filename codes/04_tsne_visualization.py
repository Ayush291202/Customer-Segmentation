"""
STAGE 4: T-SNE VISUALIZATION
Reduce PCA to 2D using t-SNE for interactive exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 4: T-SNE VISUALIZATION")
print("=" * 70)

# Load data
X_pca = np.load('X_pca_scaled.npy')
cluster_labels = np.load('cluster_labels.npy')
cleaned_data = pd.read_csv('02_cleaned_data.csv')

print(f"\nData loaded:")
print(f"  PCA data: {X_pca.shape}")
print(f"  Cluster labels: {cluster_labels.shape}")
print(f"  Original data: {cleaned_data.shape}")

# STEP 1: Apply t-SNE
print("\n1. APPLYING T-SNE (This may take 1-2 minutes)")
print("-" * 70)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1)
X_tsne = tsne.fit_transform(X_pca)

print(f"\nt-SNE output shape: {X_tsne.shape}")
print(f"t-SNE range:")
print(f"  X: [{X_tsne[:, 0].min():.2f}, {X_tsne[:, 0].max():.2f}]")
print(f"  Y: [{X_tsne[:, 1].min():.2f}, {X_tsne[:, 1].max():.2f}]")

# STEP 2: Create visualizations
print("\n2. CREATING VISUALIZATIONS")
print("-" * 70)

# Plot 1: t-SNE colored by cluster
fig, ax = plt.subplots(figsize=(14, 10))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

for cluster in np.unique(cluster_labels):
    mask = cluster_labels == cluster
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               s=80, alpha=0.6, label=cluster_names[cluster],
               color=colors[cluster], edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
ax.set_title('Customer Segments - t-SNE Visualization', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('05_tsne_clusters.png', dpi=300, bbox_inches='tight')
print("Saved: 05_tsne_clusters.png")

# Plot 2: t-SNE colored by Income
fig, ax = plt.subplots(figsize=(14, 10))

income = cleaned_data['Income'].values
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=income, s=80, alpha=0.6,
                     cmap='YlOrRd', edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
ax.set_title('Customer Segments - Colored by Income', fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Annual Income ($)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('05_tsne_income.png', dpi=300, bbox_inches='tight')
print("Saved: 05_tsne_income.png")

# Plot 3: t-SNE colored by Total Spending
fig, ax = plt.subplots(figsize=(14, 10))

spending = cleaned_data['TotalSpending'].values
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=spending, s=80, alpha=0.6,
                     cmap='viridis', edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
ax.set_title('Customer Segments - Colored by Total Spending', fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Total Spending ($)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('05_tsne_spending.png', dpi=300, bbox_inches='tight')
print("Saved: 05_tsne_spending.png")

# Plot 4: t-SNE colored by Age
fig, ax = plt.subplots(figsize=(14, 10))

age = cleaned_data['Age'].values
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=age, s=80, alpha=0.6,
                     cmap='coolwarm', edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
ax.set_title('Customer Segments - Colored by Age', fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Age (years)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('05_tsne_age.png', dpi=300, bbox_inches='tight')
print("Saved: 05_tsne_age.png")

# Plot 5: t-SNE colored by Recency (days since last purchase)
fig, ax = plt.subplots(figsize=(14, 10))

recency = cleaned_data['Recency'].values
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=recency, s=80, alpha=0.6,
                     cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
ax.set_title('Customer Segments - Colored by Recency (Days Since Purchase)', fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Recency (days)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('05_tsne_recency.png', dpi=300, bbox_inches='tight')
print("Saved: 05_tsne_recency.png")

# STEP 3: Save t-SNE results
print("\n3. SAVING RESULTS")
print("-" * 70)

tsne_df = pd.DataFrame({
    'TSNE_1': X_tsne[:, 0],
    'TSNE_2': X_tsne[:, 1],
    'Cluster': cluster_labels,
    'Income': cleaned_data['Income'].values,
    'TotalSpending': cleaned_data['TotalSpending'].values,
    'Age': cleaned_data['Age'].values,
    'Recency': cleaned_data['Recency'].values
})

tsne_df.to_csv('tsne_results.csv', index=False)
np.save('X_tsne.npy', X_tsne)

print("Saved: tsne_results.csv, X_tsne.npy")

print("\n" + "=" * 70)
print("STAGE 4 COMPLETE: t-SNE Visualization Done")
print(f"t-SNE coordinates shape: {X_tsne.shape}")
print(f"Ready for final analysis!")
print("=" * 70)
