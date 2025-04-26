"""
Behavioral feature extraction functions for the analysis of social hierarchy behavior
@author Anna Teruel-Sanchis, April 2025
"""

import pandas as pd
import numpy as np 
from shapely import polygons
from shapely.geometry import MultiPoint

def interpolate_bp(df, method='linear', limit_direction='both'):
    """
    Sometimes, if predictions fails, the x and y coordinates are NaN or missing values. 
    This function fills those missing values by interpolating the x and y coordinates. 
    
    Args:
        df (pd.DataFrame): the input dataframe (DLC multiindex: scorer, individual, bodypart, coord)
        method (str): interpolation method, includes 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'. Default 'linear'
        limit_direction (str): Direction for filling NaNs (default 'both')
    
    Returns:
        pd.DataFrame: New dataframe with interpolated x and y coordinates
    """
    x = df.loc[:, (slice(None), slice(None), slice(None), 'x')]
    y = df.loc[:, (slice(None), slice(None), slice(None), 'y')]
    x_interp = x.interpolate(method=method, limit_direction=limit_direction, axis=0)
    y_interp = y.interpolate(method=method, limit_direction=limit_direction, axis=0)
    df.loc[:, (slice(None), slice(None), slice(None), 'x')] = x_interp
    df.loc[:, (slice(None), slice(None), slice(None), 'y')] = y_interp

    return df

def get_hulls(data, bp_list):
    """
    Compute convex hulls for a list of body parts across all frames.
    The convex hull is the smallest convex shape that encloses all points in a set.
    This function computes the convex hull for each frame and returns a list of hulls.
    Each hull is represented as a shapely Polygon object. If there are fewer than 3 points, the hull is None.

    Args:
        data (pd.DataFrame): DataFrame containing x and y coordinates of body parts.
        bp_list (list): List of body parts to compute hulls for.

    Returns:
        hulls (list): List of shapely Polygons (or None) representing the convex hulls for each frame.
    """    
    x_all = data.loc[:, (slice(None), slice(None), bp_list, 'x')]
    y_all = data.loc[:, (slice(None), slice(None), bp_list, 'y')]
    coords_array = np.stack((x_all.to_numpy(), y_all.to_numpy()), axis=-1)
    hulls = [MultiPoint(coords).convex_hull if len(coords) >= 3 else None for coords in coords_array]
    return hulls

def compute_iou(hulls_ref, hulls_others, other_labels):
    """
    Compute IoU (intersection over union) between a reference list of hulls and multiple others.
    IoU is a measure of overlap between two shapes, defined as the area of intersection divided by the area of union.
    The IoU is calculated for each frame and stored in a DataFrame.

    Args:
        hulls_ref (list): List of shapely Polygons (or None) for the reference individual.
        hulls_others (list of lists): List containing lists of hulls for other individuals.
        other_labels (list): List of strings, names for other individuals (e.g., ['m2', 'm3', 'm4']).

    Returns:
        pd.DataFrame: DataFrame where each column is IoU with one other individual.
    """
    ious = []
    for frame_idx in range(len(hulls_ref)):
        row = {}
        h_ref = hulls_ref[frame_idx]
        
        for label, hulls_other in zip(other_labels, hulls_others):
            h_other = hulls_other[frame_idx]
            if h_ref and h_other:
                intersection_area = h_ref.intersection(h_other).area
                union_area = h_ref.union(h_other).area
                if union_area > 0:
                    row[f'{label}_iou'] = intersection_area / union_area
                else:
                    row[f'{label}_iou'] = 0
            else:
                row[f'{label}_iou'] = 0
        
        ious.append(row)
    
    return pd.DataFrame(ious)

def compute_centroid(df, bodyparts):
    """
    Computes centroid (mean x, mean y) across selected bodyparts.
    The centroid is the average position of all points in a shape.
    The centroid is calculated for each frame and returned as a 2D array.
    
    Args:
        df (pd.DataFrame): DataFrame containing x and y coordinates of body parts.
        bodyparts (list): List of body parts to compute centroid for.
    
    Returns:
        np.ndarray: 2D array of shape (n_frames, 2) containing the x and y coordinates of the centroid.
    """
    x = df.loc[:, (slice(None), slice(None), bodyparts, 'x')].mean(axis=1)
    y = df.loc[:, (slice(None), slice(None), bodyparts, 'y')].mean(axis=1)
    return np.stack([x, y], axis=1)

def compute_body_length(df, anterior_bp, posterior_bp):
    """
    Computes body length between two bodyparts.
    The body length is the Euclidean distance between the two points.
    
    Args:
        df (pd.DataFrame): DataFrame containing x and y coordinates of body parts.
        anterior_bp (str): Name of the anterior body part.
        posterior_bp (str): Name of the posterior body part.
    
    Returns:
        np.ndarray: 1D array of body lengths for each frame.
    """
    anterior = df.loc[:, (slice(None), slice(None), anterior_bp)]
    posterior = df.loc[:, (slice(None), slice(None), posterior_bp)]
    length = np.linalg.norm(anterior.values - posterior.values, axis=1)
    return length

def compute_orientation(a, b, c):
    """
    Calculate the angle between three points a-b-c.
    The angle is calculated using the cross product and dot product of the vectors formed by the points.
    
    Args:
        a (dict): Coordinates of point a (x, y).
        b (dict): Coordinates of point b (x, y).
        c (dict): Coordinates of point c (x, y).
    
    Returns:
        np.ndarray: 1D array of angles in degrees for each frame.
        
    Example:
        a = {'x': 1, 'y': 2}
        b = {'x': 3, 'y': 4}
        c = {'x': 5, 'y': 6}
        angles = compute_orientation(a, b, c)
        print(angles)  # Output: array of angles in degrees
    """
    crossprod = (c['x']-b['x'])*(b['y']-a['y']) - (c['y']-b['y'])*(b['x']-a['x'])
    dotprod = (c['x']-b['x'])*(b['x']-a['x']) + (c['y']-b['y'])*(b['y']-a['y'])
    angles = np.arctan2(crossprod, dotprod)
    angles = np.where(angles < 0, angles * -1, angles)
    angles = np.degrees(angles)
    return angles

def compute_speed(df, bp):
    """
    Computes speed of a bodypart between consecutive frames.
    The speed is calculated as the Euclidean distance between consecutive frames.
    Speed is defined in pixels/frame.
    
    Args:
        df (pd.DataFrame): DataFrame containing x and y coordinates of body parts.
        bp (str): Name of the body part to compute speed for.
    
    Returns:
        np.ndarray: 1D array of speeds for each frame.
    """
    coords = df.loc[:, (slice(None), slice(None), bp)].values
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    speed = np.concatenate([[0], diffs])
    return speed



