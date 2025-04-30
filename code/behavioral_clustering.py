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
import seaborn as sns
from scipy.ndimage import zoom



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
                cmap='YlOrRd', 
                plot=True, 
                save=False,
                save_dir=None, 
                format='svg',
                bw_adjust=0.8,
                kde_thresh=0.02,
                kde_levels=100,
                contour_smoothing=3):
    """
    Compute a smooth density map from a 2D embedding using a KDE plot and identify clusters using 
    watershed segmentation. This function supports optional plotting with smoothed cluster boundaries,
    customizable visual appearance, and saving the resulting figure.

    The density map is computed from a 2D histogram of the embedding and smoothed with a Gaussian filter. 
    Cluster labels are obtained by applying watershed segmentation using local maxima in the density map. 
    The resulting clusters are visualized as contour boundaries over a seaborn-based KDE heatmap.

    Args:
        embedding (np.ndarray): 2D embedding (e.g., UMAP or t-SNE), shape (n_points, 2). Each row is a 
                                2D coordinate representing a frame or sample in reduced space.
        bins (int, optional): Number of bins for the 2D histogram used in density estimation. Higher values 
                              provide finer resolution. Default is 200.
        sigma (float, optional): Standard deviation for the Gaussian filter used to smooth the histogram-based 
                                 density map. Larger values yield smoother maps. Default is 3.5.
        percentile (int, optional): Percentile cutoff used to threshold low-density regions. Points below 
                                    this threshold are excluded from watershed clustering. Default is 30.
        cmap (str, optional): Colormap used for visualizing the KDE density plot. Default is 'YlOrRd'.
        plot (bool, optional): Whether to plot the KDE density map overlaid with watershed-based cluster 
                               contours and labels. Default is True.
        save (bool, optional): Whether to save the figure to disk. If True, the plot will be saved using the 
                               specified format and directory. Default is False.
        save_dir (str, optional): Directory path to save the output plot if `save` is True. If None, the plot 
                                  will not be saved. Default is None.
        format (str, optional): File format to save the figure. Supported formats include 'svg', 'png', 'pdf', etc.
                                Default is 'svg'.
        bw_adjust (float, optional): Bandwidth adjustment for seaborn's KDE estimation. Higher values smooth 
                                     the KDE more. Default is 0.8.
        kde_thresh (float, optional): Minimum KDE density threshold below which values are not plotted. 
                                      Useful to hide very low-density background. Default is 0.02.
        kde_levels (int, optional): Number of contour levels in the KDE. Controls visual resolution. Default is 100.
        contour_smoothing (int, optional): Zoom factor applied to upsample the watershed map before contouring, 
                                           which results in smoother contour lines. Default is 3.

    Returns:
        labeled_map (np.ndarray): 2D array of shape (bins, bins), where each nonzero element corresponds 
                                  to a unique cluster label. Background regions are 0.
        density_map (np.ndarray): Smoothed 2D histogram density map before thresholding or masking.
        xe (np.ndarray): 1D array of bin edges along the x-axis used in the histogram.
        ye (np.ndarray): 1D array of bin edges along the y-axis used in the histogram.

    Example:
        >>> labeled_map, density_map, xe, ye = map_density(embedding, bins=200, sigma=4, percentile=35, 
        ...                                                kde_levels=120, contour_smoothing=4, plot=True)
        >>> print(np.unique(labeled_map))
    """
    
    print("Computing density map...")

    # 2D histogram and smoothing
    density_map, xe, ye = np.histogram2d(
        embedding[:, 0], embedding[:, 1], bins=bins, density=True
    )
    density_map = gaussian_filter(density_map, sigma=sigma)

    # watershed segmentation
    density_cutoff = np.percentile(density_map, percentile)
    density_mask = density_map > density_cutoff
    local_max = peak_local_max(density_map, min_distance=1, #in the paper they use 1 as default
                               footprint=np.ones((3, 3)), labels=density_mask)
    local_max_mask = np.zeros_like(density_map, dtype=bool)
    local_max_mask[tuple(local_max.T)] = True
    markers, _ = label(local_max_mask)
    labeled_map = watershed(-density_map, markers, mask=density_mask, connectivity=2).astype("float64")

    if plot:
        fig, ax = plt.subplots(figsize=(14, 12))

        # KDE heatmap
        sns.kdeplot(
            x=embedding[:, 0], 
            y=embedding[:, 1], 
            fill=True, 
            cmap=cmap, 
            bw_adjust=bw_adjust,
            thresh=kde_thresh,
            levels=kde_levels,
            ax=ax
        )

        # smoother contours
        labeled_map_up = zoom(labeled_map, contour_smoothing, order=0)
        x_range = xe[-1] - xe[0]
        y_range = ye[-1] - ye[0]
        extent_up = [xe[0], xe[0] + x_range, ye[0], ye[0] + y_range]

        # overlay contours
        ax.contour(
            labeled_map_up.T,
            levels=np.arange(1, np.max(labeled_map) + 1),
            colors="black",
            linewidths=2,
            origin="lower",
            extent=extent_up,
            alpha=0.7
        )

        # Add cluster labels
        for i in np.unique(labeled_map):
            if i == 0:
                continue
            mask = labeled_map == i
            if np.sum(mask) < 20:
                continue
            com = center_of_mass(mask)
            cx = xe[0] + (xe[-1] - xe[0]) * com[0] / labeled_map.shape[0]
            cy = ye[0] + (ye[-1] - ye[0]) * com[1] / labeled_map.shape[1]
            ax.text(cx, cy, str(int(i)), color="black", fontsize=10, ha="center", va="center")


        mappable = ax.collections[0]
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("PDF", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.axis("off")

        if save and save_dir:
            filepath = f"{save_dir}/umap_heatmap.{format}"
            plt.savefig(filepath, dpi=300, format=format, bbox_inches='tight', pad_inches=0.5)
            print(f"Plot saved to {filepath}")
        elif save:
            print("Warning: save=True but save_dir not specified.")

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

    return Z,  cluster_labels