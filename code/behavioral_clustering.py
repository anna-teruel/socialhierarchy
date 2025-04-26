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

def run_watershed_clustering(embedding, bins=200, sigma=3.5, percentile=30, plot=True):
    """
    Run watershed clustering on a 2D UMAP embedding.

    Args:
        embedding (np.ndarray): 2D UMAP embedding (frames x 2).
        bins (int): Number of bins for 2D histogram. Higher bins = more resolution.
        sigma (float): Gaussian smoothing parameter.
        percentile (float): Density threshold percentile to define peaks.
        plot (bool): Whether to plot the density and labels.

    Returns:
        labeled_map (np.ndarray): Watershed labeled map.
        density_map (np.ndarray): 2D density map.
        xe, ye (np.ndarray): Bin edges in x and y.
    """
    print('Computing density map...')
    
    density_map, xe, ye = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=bins, density=True)
    density_map = gaussian_filter(density_map, sigma=sigma)

    local_max = peak_local_max(density_map, indices=False, footprint=np.ones((3, 3)), labels=density_map)
    mask = density_map > np.percentile(density_map, percentile)
    local_max = local_max & mask
    markers, _ = label(local_max)

    labeled_map = watershed(-density_map, markers, mask=mask)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(density_map.T, origin='lower', cmap='magma')
        plt.colorbar(label='Density')
        plt.title('Density Map')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(labeled_map.T, origin='lower', cmap='tab20')
        plt.title('Watershed Clusters')
        plt.show()

    return labeled_map, density_map, xe, ye