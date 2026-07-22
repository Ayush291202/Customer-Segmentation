"""
STAGE 2: PCA (PRINCIPAL COMPONENT ANALYSIS)
Reduce 29 features to 10 principal components
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 2: PCA ANALYSIS")
print("=" * 70)

# Load cleaned data
X = pd.read_csv('02_cleaned_data.csv')

print(f"\nData loaded: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# STEP 1: Standardize the features
# This is important because features have different scales
print("\n1. STANDARDIZING FEATURES")
print("-" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Original data range:")
print(f"  Min: {X.values.min():.2f}")
print(f"  Max: {X.values.max():.2f}")
print(f"\nAfter scaling (StandardScaler):")
print(f"  Mean: {X_scaled.mean():.4f} (should be ~0)")
print(f"  Std: {X_scaled.std():.4f} (should be ~1)")

# STEP 2: Apply PCA
# Keep all components first to see explained variance
print("\n2. APPLYING PCA")
print("-" * 70)

pca_all = PCA()
pca_all.fit(X_scaled)

# Get explained variance
explained_variance = pca_all.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"Total features: {len(X.columns)}")
print(f"\nExplained variance by component:")
for i in range(min(10, len(explained_variance))):
    print(f"  PC{i+1}: {explained_variance[i]*100:6.2f}% (Cumulative: {cumulative_variance[i]*100:6.2f}%)")

# STEP 3: Choose number of components
# We want to explain ~80-85% of variance
n_components = 10
variance_80 = np.where(cumulative_variance >= 0.80)[0][0] + 1
variance_85 = np.where(cumulative_variance >= 0.85)[0][0] + 1

print(f"\nVariance explained by different numbers of components:")
print(f"  To explain 80% variance: {variance_80} components ({cumulative_variance[variance_80-1]*100:.2f}%)")
print(f"  To explain 85% variance: {variance_85} components ({cumulative_variance[variance_85-1]*100:.2f}%)")
print(f"  We will use: {n_components} components ({cumulative_variance[n_components-1]*100:.2f}%)")

# STEP 4: Apply PCA with 10 components
print(f"\n3. FITTING PCA WITH {n_components} COMPONENTS")
print("-" * 70)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA components shape: {X_pca.shape}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# STEP 5: Analyze PCA loadings (which features contribute to each component)
print(f"\n4. PCA LOADINGS ANALYSIS")
print("-" * 70)
print("(Which features contribute most to each principal component)\n")

loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=X.columns
)

# Show top contributors for first 3 components
for pc in range(3):
    pc_name = f'PC{pc+1}'
    print(f"\n{pc_name} - Explains {explained_variance[pc]*100:.2f}% of variance")
    print("-" * 70)

    # Get absolute loadings (importance regardless of direction)
    loadings_abs = loadings_df[pc_name].abs().sort_values(ascending=False)

    print("Top 5 positive contributors (increase with this component):")
    for i, (feature, loading) in enumerate(loadings_df[pc_name].nlargest(5).items(), 1):
        print(f"  {i}. {feature:25s}: {loading:+.4f}")

    print("\nTop 5 negative contributors (decrease with this component):")
    for i, (feature, loading) in enumerate(loadings_df[pc_name].nsmallest(5).items(), 1):
        print(f"  {i}. {feature:25s}: {loading:+.4f}")

# STEP 6: Create visualization
print(f"\n5. CREATING VISUALIZATIONS")
print("-" * 70)

# Plot 1: Explained Variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
axes[0].bar(range(1, n_components+1), explained_variance[:n_components]*100,
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Variance Explained by Each PC', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xticks(range(1, n_components+1))

# Cumulative variance
axes[1].plot(range(1, n_components+1), cumulative_variance[:n_components]*100,
             marker='o', linewidth=2.5, markersize=8, color='darkgreen')
axes[1].axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% threshold')
axes[1].axhline(y=85, color='orange', linestyle='--', linewidth=2, label='85% threshold')
axes[1].fill_between(range(1, n_components+1), cumulative_variance[:n_components]*100,
                      alpha=0.2, color='darkgreen')
axes[1].set_xlabel('Number of Components', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_xticks(range(1, n_components+1))
axes[1].set_ylim([0, 105])

plt.tight_layout()
plt.savefig('03_pca_variance.png', dpi=300, bbox_inches='tight')
print("Saved: 03_pca_variance.png")

# Plot 2: PCA Loadings Heatmap (for first 5 components)
fig, ax = plt.subplots(figsize=(12, 10))

loadings_plot = loadings_df.iloc[:, :5].copy()
im = ax.imshow(loadings_plot.values, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

ax.set_xticks(range(5))
ax.set_xticklabels([f'PC{i+1}' for i in range(5)], fontsize=11, fontweight='bold')
ax.set_yticks(range(len(loadings_plot)))
ax.set_yticklabels(loadings_plot.index, fontsize=10)
ax.set_title('PCA Loadings - Feature Contributions', fontsize=13, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Loading Value', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(loadings_plot)):
    for j in range(5):
        value = loadings_plot.values[i, j]
        text_color = 'white' if abs(value) > 0.25 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                color=text_color, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('03_pca_loadings.png', dpi=300, bbox_inches='tight')
print("Saved: 03_pca_loadings.png")

# Save results
np.save('X_pca_scaled.npy', X_pca)
loadings_df.to_csv('pca_loadings.csv')
pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(n_components)],
    'Variance_Explained': explained_variance[:n_components],
    'Cumulative_Variance': cumulative_variance[:n_components]
}).to_csv('pca_variance.csv', index=False)

print("\nSaved: X_pca_scaled.npy, pca_loadings.csv, pca_variance.csv")

print("\n" + "=" * 70)
print("STAGE 2 COMPLETE: PCA Analysis Done")
print(f"PCA Data shape: {X_pca.shape}")
print(f"Ready for K-means clustering!")
print("=" * 70)
