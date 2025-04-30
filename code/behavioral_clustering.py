"""
Behaviuoral clustering. 
This script loads features from multiple sessions, samples frames uniformly across sessions, and runs UMAP for dimensionality reduction.
It also includes functions for visualizing the UMAP embedding and clustering the data using watershed segmentation.

@author Anna Teruel-Sanchis, April 2025
"""
import numpy as np
import pandas as pd
import os
import time
import umap #type: ignore
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.ndimage import center_of_mass
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy.ma as ma
import matplotlib.cm as cm



def load_features(directory, file_format='csv'):
    """
    Load feature files (.csv or .h5) into a dictionary of DataFrames.

    Args:
        directory (str): Path to the directory containing feature files.
        file_format (str): 'csv' or 'h5'. Default is 'csv'.

    Returns:
        dict: Dictionary where keys are filenames (without extension) and values are DataFrames.
    """
    features_dict = {}

    for file in os.listdir(directory):
        if file.endswith(f'.{file_format}'):
            path = os.path.join(directory, file)
            if file_format == 'csv':
                df = pd.read_csv(path)
            elif file_format == 'h5':
                df = pd.read_hdf(path)
            else:
                raise ValueError("file_format must be either 'csv' or 'h5'")

            name = os.path.splitext(file)[0]  # filename without extension
            features_dict[name] = df

    return features_dict

def sample_frames(directory,file_format, frames_total, random_seed=42):
    """
    Uniformly sample frames from multiple sessions to create a representative set.
    Every frame is a point in high-dimensional space (e.g., 100,000 frames × 30 features).
    But if you use ALL frames, the embedding (e.g. UMAP) would take forever to compute and be noisy.
    Many frames may be similar because animals do repetitive behaviors. So we want to sample a representative set of frames.
    This function samples frames from each session to create a balanced dataset. This sampling method is not random. 
    The idea is to get a fair representation of the whole behavioral recording. 

    Args:
        directory (str): Path to the directory containing feature files.
        file_format (str): 'csv' or 'h5'. Default is 'csv'.
        feats_dict (dict): Dictionary where keys are session names and values are DataFrames of features.
        frames_total (int): Total number of frames you want after sampling, in your final dataset
        random_seed (int): Seed for reproducibility (default 42). 

    Returns:
        np.ndarray: Sampled features, shape = (frames_total, n_features)
    """
    feats_dict = load_features(directory, file_format)
    
    np.random.seed(random_seed)
    sampled_feats = []
    
    n_sessions = len(feats_dict)
    frames_per_session = frames_total // n_sessions

    for session_name, session_feats in feats_dict.items():
        n_frames = session_feats.shape[0]
        
        if n_frames >= frames_per_session:
            indices = np.linspace(0, n_frames - 1, frames_per_session).astype(int)
        else:
            indices = np.random.choice(np.arange(n_frames), frames_per_session, replace=True)

        sampled = session_feats.select_dtypes(include=[np.number]).iloc[indices].to_numpy()
        sampled_feats.append(sampled)

    sampled_feats = np.vstack(sampled_feats)
    return sampled_feats

def run_umap(sampled_feats, n_neighbors=50, min_dist=0.1, n_components=2, random_state=42):
    """
    Run UMAP dimensionality reduction on sampled features.

    Args:
        sampled_feats (np.ndarray): Sampled features (frames x features).
        n_neighbors (int): Number of neighbors considered when building the local manifold. Lower neighbors capture more local structure.
                    Higher neighbors capture more global structure.
        min_dist (float): Minimum distance in low-dimensional space. how tightly UMAP packs points together. 
                    Higer distance is better for continuity
        n_components (int): Number of output dimensions (2 for visualization).
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray: 2D embedded features.
    """
    print('Running UMAP embedding...')
    time_start = time.time()
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        n_components=n_components,
        random_state=random_state
    )
    embedding = reducer.fit_transform(sampled_feats)
    
    print(f'UMAP completed in {time.time() - time_start:.2f} seconds.')
    print(f'UMAP embedding shape: {embedding.shape}')
    
    return embedding

def train_embedding_model(sampled_feats, embedding, save_model=False, save_path=None):
    """
    Train a model to predict UMAP embeddings for new data.

    Args:
        sampled_feats (np.ndarray): Original high-dimensional features (frames x features).
        embedding (np.ndarray): Corresponding UMAP embedding (frames x 2).
        save_model (bool): Whether to save the trained model.
        save_path (str): Path to save the model if save_model=True.

    Returns:
        MLPRegressor: Trained model.
    """
    x_train, x_test, y_train, y_test = train_test_split(sampled_feats, embedding, test_size=0.2, random_state=42)
    
    mlp = MLPRegressor(hidden_layer_sizes=(500, 250, 125, 50), max_iter=500, random_state=42)
    mlp.fit(x_train, y_train)
    
    print(f"Training R^2: {mlp.score(x_train, y_train):.2f}")
    print(f"Testing R^2: {mlp.score(x_test, y_test):.2f}")
    
    if save_model and save_path:
        import joblib
        joblib.dump(mlp, save_path)
        print(f"Model saved to {save_path}")
    
    return mlp

def predict_embeddings(model, new_feats):
    """
    Predict embeddings for new data using a trained model.

    Args:
        model (MLPRegressor): Trained model.
        new_feats (np.ndarray): New high-dimensional features.

    Returns:
        np.ndarray: Predicted embeddings.
    """
    return model.predict(new_feats)

def plot_umap_embedding(embedding, 
                        color='black', 
                        alpha=0.5, 
                        size=1, 
                        save=False, 
                        save_dir=None,
                        format = 'svg', 
):
    """
    Plot the UMAP embedding as a scatter plot.

    Args:
        embedding (np.ndarray): 2D UMAP embedding (frames x 2).
        color (str): Color of the points in the scatter plot.
        alpha (float): Transparency of the points.
        size (int): Size of the points in the scatter plot.
        save (bool): Whether to save the plot.
        save_dir (str): Directory to save the plot if save=True.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=color, s=size, alpha=alpha)
    plt.title("UMAP Embedding", fontsize=16)
    plt.axis('off')
    
    if save:
        if save_dir:
            file_path = f"{save_dir}/umap_embedding.{format}"
            plt.savefig(file_path, dpi=300, bbox_inches='tight', format=format)
            print(f"UMAP embedding plot saved to {file_path}")
        else:
            print("Warning: save=True but no save_dir provided. Plot will not be saved.")
    
    plt.show()

def map_density(embedding, 
                bins=200, 
                sigma=3.5, 
                percentile=30,
                cmap='plasma', 
                plot=True, 
                save=False,
                save_dir=None, 
                format='svg'):
    """
    Compute a density map from the UMAP embedding using a 2D histogram and watershed segmentation.
    This function identifies clusters in the embedding space by calculating a smoothed density map,
    applying a density threshold, and performing watershed segmentation to assign cluster labels.

    The density map is computed using a 2D histogram of the embedding, smoothed with a Gaussian filter.
    Local maxima in the density map are used as seeds for the watershed algorithm, which segments the
    embedding space into distinct clusters.

    Args:
        embedding (np.ndarray): 2D UMAP embedding (frames x 2). Each row represents a point in the 
                                embedding space, and the two columns represent the x and y coordinates.
        bins (int, optional): Number of bins for the 2D histogram. Higher values result in finer resolution
                              for the density map. Defaults to 200.
        sigma (float, optional): Standard deviation for the Gaussian filter applied to the density map.
                                 Larger values result in smoother density maps. Defaults to 3.5.
        percentile (int, optional): Percentile threshold for density cutoff. Points below this density
                                     threshold are excluded from clustering. Defaults to 30.
        cmap (str, optional): Colormap for visualizing the density map. Defaults to 'plasma'.
        plot (bool, optional): Whether to plot the density map and the resulting clusters. If True, the
                               function visualizes the density map, cluster boundaries, and cluster labels.
                               Defaults to True.
        save (bool, optional): Whether to save the plot as an image file. If True, the plot will be saved
                                 to the specified directory in the specified format. Defaults to False.
        save_dir (str, optional): Directory where the plot will be saved if `save` is True. Defaults to None.
        format (str, optional): Format for saving the plot if `save` is True. Supported formats include
                                'png', 'svg', 'pdf', etc. Defaults to 'svg'.

    Returns:
        labeled_map (np.ndarray): A 2D array where each element corresponds to a cluster label. The shape
                                  matches the grid of the density map. Background points (not part of any
                                  cluster) are labeled as NaN.
        density_map (np.ndarray): A 2D array representing the smoothed density map of the embedding space.
                                  Higher values indicate regions with higher point density.
        xe (np.ndarray): 1D array of bin edges along the x-axis of the embedding space, corresponding to
                         the 2D histogram.
        ye (np.ndarray): 1D array of bin edges along the y-axis of the embedding space, corresponding to
                         the 2D histogram.

    Example:
        >>> labeled_map, density_map, xe, ye = map_density(embedding, bins=200, sigma=3.5, percentile=30, plot=True)
        >>> print(f"Labeled map shape: {labeled_map.shape}")
        >>> print(f"Density map shape: {density_map.shape}")
    """ 
    
    print("Computing density map...")

    density_map, xe, ye = np.histogram2d(
        embedding[:, 0], embedding[:, 1], bins=bins, density=True
    )
    density_map = gaussian_filter(density_map, sigma=sigma)

    density_cutoff = np.percentile(density_map, percentile)
    density_mask = density_map > density_cutoff

    local_max = peak_local_max(
        density_map, min_distance=1, footprint=np.ones((3, 3)), labels=density_mask
    )

    local_max_mask = np.zeros_like(density_map, dtype=bool)
    local_max_mask[tuple(local_max.T)] = True

    markers, _ = label(local_max_mask)
    labeled_map = watershed(-density_map, 
                            markers, 
                            mask=density_mask, 
                            connectivity=2)
    labeled_map = labeled_map.astype("float64")

    if plot:
        from matplotlib.colors import ListedColormap

        plt.figure(figsize=(12, 10))

        # Mask very low density regions
        threshold = np.percentile(density_map, 30)
        density_map[density_map < threshold] = np.nan

        custom_cmap = cm.get_cmap(cmap).copy()
        custom_cmap.set_bad(color='white')

        im = plt.imshow(
            density_map.T,
            cmap=custom_cmap,
            origin="lower",
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
            alpha=0.9, 
            vmin=threshold,
        )

        plt.contour(
            labeled_map.T,
            levels=np.arange(1, np.max(labeled_map) + 1),
            colors="black",
            linewidths=2,
            origin="lower",
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
            alpha=0.7
        )

        for i in np.unique(labeled_map):
            if i == 0:
                continue
            mask = labeled_map == i
            if np.sum(mask) < 20:
                continue  # skip tiny clusters
            com = center_of_mass(mask)
            cx = xe[0] + (xe[-1] - xe[0]) * com[0] / labeled_map.shape[0]
            cy = ye[0] + (ye[-1] - ye[0]) * com[1] / labeled_map.shape[1]
            plt.text(cx, cy, str(int(i)), color="black", fontsize=10, ha="center", va="center")

        cbar = plt.colorbar(im, fraction=0.03, pad=0.04)
        cbar.set_label("PDF", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        plt.axis("off")
        # plt.title("Clustered Behavioral Map", fontsize=14)
        # plt.tight_layout()
        if save:
            if save_dir:
                file_path = f"{save_dir}/umap_heatmap.{format}"
                plt.savefig(file_path, dpi=300, bbox_inches='tight', format=format)
                print(f"UMAP heatmap plot saved to {file_path}")
            else:
                print("Warning: save=True but no save_dir provided. Plot will not be saved.")
        
    plt.show()

    return labeled_map, density_map, xe, ye

def hierarchical_clustering(embedding, labeled_map, xe, ye, method='ward', plot=True):
    """
    Perform hierarchical clustering on behavior clusters and optionally plot a dendrogram.

    Args:
        embedding (np.ndarray): UMAP 2D embedding (n_frames x 2).
        labeled_map (np.ndarray): Watershed cluster map from density-based clustering.
        xe (np.ndarray): x-axis bin edges from histogram2d.
        ye (np.ndarray): y-axis bin edges from histogram2d.
        method (str): Linkage method for hierarchical clustering (default 'ward').
        plot (bool): Whether to plot the dendrogram.

    Returns:
        Z (np.ndarray): The linkage matrix used to construct the dendrogram.
    """
    x_idx = np.digitize(embedding[:, 0], xe) - 1
    y_idx = np.digitize(embedding[:, 1], ye) - 1

    valid = (
        (x_idx >= 0) & (x_idx < labeled_map.shape[0]) &
        (y_idx >= 0) & (y_idx < labeled_map.shape[1])
    )
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    embedding_valid = embedding[valid]

    cluster_labels = labeled_map[x_idx, y_idx]

    valid_points = ~np.isnan(cluster_labels) & (cluster_labels > 0)
    labels = cluster_labels[valid_points].astype(int)
    embedding_valid = embedding_valid[valid_points]
    unique_labels = np.unique(labels)
    centroids = np.array([
        embedding_valid[labels == lbl].mean(axis=0)
        for lbl in unique_labels
    ])

    Z = linkage(centroids, method=method)

    if plot:
        plt.figure(figsize=(20, 15))
        dendrogram(Z, labels=unique_labels)
        plt.title("Hierarchical Clustering of Behavioral Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

    return Z