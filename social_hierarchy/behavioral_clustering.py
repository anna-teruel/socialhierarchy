"""
Behaviuoral clustering. 
@author Anna Teruel-Sanchis, April 2025
"""
import numpy as np
import pandas as pd

def sample_frames(feats_dict, frames_total, random_seed=42):
    """
    Uniformly sample frames from multiple sessions to create a representative set.
    Every frame is a point in high-dimensional space (e.g., 100,000 frames × 30 features).
    But if you use ALL frames, the embedding (e.g. UMAP) would take forever to compute and be noisy.
    Many frames may be similar because animals do repetitive behaviors. So we want to sample a representative set of frames.
    This function samples frames from each session to create a balanced dataset. This sampling method is not random. 
    The idea is to get a fair representation of the whole behavioral recording. 

    Args:
        feats_dict (dict): Dictionary where keys are session names and values are DataFrames of features.
        frames_total (int): Total number of frames you want after sampling, in your final dataset
        random_seed (int): Seed for reproducibility (default 42). 

    Returns:
        np.ndarray: Sampled features, shape = (frames_total, n_features)
    """

    np.random.seed(random_seed)
    sampled_feats = []
    
    n_sessions = len(feats_dict)
    frames_per_session = frames_total // n_sessions

    for session_name, session_feats in feats_dict.items():
        n_frames = session_feats.shape[0]
        
        if n_frames >= frames_per_session:
            indices = np.linspace(0, n_frames - 1, frames_per_session).astype(int) #sample uniformly across session
        else:
            indices = np.random.choice(np.arange(n_frames), frames_per_session, replace=True) #if session is shorter, sample with replacement

        sampled = session_feats.iloc[indices].to_numpy()
        sampled_feats.append(sampled)

    sampled_feats = np.vstack(sampled_feats) # Stack all sampled frames into one big array
    return sampled_feats

