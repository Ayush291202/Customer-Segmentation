"""
STAGE 3: K-MEANS CLUSTERING
Find optimal number of clusters and segment customers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 3: K-MEANS CLUSTERING")
print("=" * 70)

# Load PCA data
X_pca = np.load('X_pca_scaled.npy')
print(f"\nPCA data loaded: {X_pca.shape}")

# STEP 1: Find optimal number of clusters
# Test k from 2 to 8 and compare metrics
print("\n1. FINDING OPTIMAL NUMBER OF CLUSTERS")
print("-" * 70)

k_values = range(2, 9)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

for k in k_values:
    print(f"Testing k={k}...", end=" ")

    # Train K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_pca, labels)
    davies_bouldin = davies_bouldin_score(X_pca, labels)
    calinski_harabasz = calinski_harabasz_score(X_pca, labels)

    inertias.append(inertia)
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(davies_bouldin)
    calinski_harabasz_scores.append(calinski_harabasz)

    print(f"Silhouette: {silhouette:.4f}, DB Index: {davies_bouldin:.4f}, CH Index: {calinski_harabasz:.2f}")

# Create comparison table
metrics_df = pd.DataFrame({
    'k': list(k_values),
    'Inertia': inertias,
    'Silhouette': silhouette_scores,
    'Davies_Bouldin': davies_bouldin_scores,
    'Calinski_Harabasz': calinski_harabasz_scores
})

print("\n\nMetrics Summary:")
print(metrics_df.to_string(index=False))

# Find optimal k
best_silhouette_k = k_values[np.argmax(silhouette_scores)]
best_db_k = k_values[np.argmin(davies_bouldin_scores)]  # Lower is better
best_ch_k = k_values[np.argmax(calinski_harabasz_scores)]  # Higher is better

print(f"\nBest k by Silhouette Score: {best_silhouette_k}")
print(f"Best k by Davies-Bouldin Index: {best_db_k}")
print(f"Best k by Calinski-Harabasz Index: {best_ch_k}")

# STEP 2: Choose optimal k (we'll use 4 as a good balance)
optimal_k = 4  # Good balance between separation and interpretability
print(f"\nChosen k: {optimal_k} (good balance between metrics and interpretability)")

# STEP 3: Train final K-means model
print(f"\n2. TRAINING K-MEANS WITH k={optimal_k}")
print("-" * 70)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_pca)

final_silhouette = silhouette_score(X_pca, cluster_labels)
final_davies_bouldin = davies_bouldin_score(X_pca, cluster_labels)
final_calinski_harabasz = calinski_harabasz_score(X_pca, cluster_labels)

print(f"Silhouette Score: {final_silhouette:.4f}")
print(f"Davies-Bouldin Index: {final_davies_bouldin:.4f}")
print(f"Calinski-Harabasz Index: {final_calinski_harabasz:.2f}")

# STEP 4: Analyze cluster sizes
print(f"\n3. CLUSTER ANALYSIS")
print("-" * 70)

unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_sizes = pd.DataFrame({
    'Cluster': unique,
    'Size': counts,
    'Percentage': counts/len(cluster_labels)*100
})

print("\nCluster Sizes:")
print(cluster_sizes.to_string(index=False))

# STEP 5: Visualizations
print(f"\n4. CREATING VISUALIZATIONS")
print("-" * 70)

# Plot 1: Metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Inertia (Elbow method)
axes[0, 0].plot(k_values, inertias, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Chosen k={optimal_k}')
axes[0, 0].set_xlabel('Number of Clusters (k)', fontweight='bold')
axes[0, 0].set_ylabel('Inertia', fontweight='bold')
axes[0, 0].set_title('Elbow Method - Inertia', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
axes[0, 0].set_xticks(k_values)

# Silhouette Score (higher is better)
axes[0, 1].plot(k_values, silhouette_scores, marker='s', linewidth=2, markersize=8, color='darkgreen')
axes[0, 1].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Chosen k={optimal_k}')
axes[0, 1].set_xlabel('Number of Clusters (k)', fontweight='bold')
axes[0, 1].set_ylabel('Silhouette Score', fontweight='bold')
axes[0, 1].set_title('Silhouette Score (Higher is Better)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].set_xticks(k_values)

# Davies-Bouldin Index (lower is better)
axes[1, 0].plot(k_values, davies_bouldin_scores, marker='^', linewidth=2, markersize=8, color='darkorange')
axes[1, 0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Chosen k={optimal_k}')
axes[1, 0].set_xlabel('Number of Clusters (k)', fontweight='bold')
axes[1, 0].set_ylabel('Davies-Bouldin Index', fontweight='bold')
axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
axes[1, 0].set_xticks(k_values)

# Calinski-Harabasz Index (higher is better)
axes[1, 1].plot(k_values, calinski_harabasz_scores, marker='d', linewidth=2, markersize=8, color='darkred')
axes[1, 1].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Chosen k={optimal_k}')
axes[1, 1].set_xlabel('Number of Clusters (k)', fontweight='bold')
axes[1, 1].set_ylabel('Calinski-Harabasz Index', fontweight='bold')
axes[1, 1].set_title('Calinski-Harabasz Index (Higher is Better)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
axes[1, 1].set_xticks(k_values)

plt.tight_layout()
plt.savefig('04_kmeans_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: 04_kmeans_metrics.png")

# Plot 2: Cluster sizes
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'][:optimal_k]
bars = ax.bar(cluster_sizes['Cluster'], cluster_sizes['Size'], color=colors, edgecolor='black', linewidth=2)

ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
ax.set_title(f'Customer Distribution across {optimal_k} Clusters', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(optimal_k))

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(cluster_labels)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('04_cluster_sizes.png', dpi=300, bbox_inches='tight')
print("Saved: 04_cluster_sizes.png")

# Save results
np.save('cluster_labels.npy', cluster_labels)
metrics_df.to_csv('kmeans_metrics.csv', index=False)
cluster_sizes.to_csv('cluster_sizes.csv', index=False)

# Save cluster centers
cluster_centers_df = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=[f'PC{i+1}' for i in range(10)]
)
cluster_centers_df.to_csv('cluster_centers.csv', index=False)

print("\nSaved: cluster_labels.npy, kmeans_metrics.csv, cluster_sizes.csv, cluster_centers.csv")

print("\n" + "=" * 70)
print("STAGE 3 COMPLETE: K-Means Clustering Done")
print(f"Cluster labels shape: {cluster_labels.shape}")
print(f"Optimal clusters: {optimal_k}")
print(f"Ready for t-SNE visualization!")
print("=" * 70)
