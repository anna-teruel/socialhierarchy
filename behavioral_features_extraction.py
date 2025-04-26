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

def compute_individual_features(df, individual, bp_list, anterior_bp, posterior_bp, bp_angle):
    """
    Compute individual features (geometry, motion, head orientation) for one individual.
    
    Args:
        df (pd.DataFrame): Full dataframe.
        individual (str): Individual to process (e.g., 'm1').
        bp_list (list): List of bodyparts for centroid/hull calculation.
        anterior_bp (str): Name of anterior bodypart (e.g., 'nose').
        posterior_bp (str): Name of posterior bodypart (e.g., 'tailbase').
        bp_angle (str): Body part to compute angles for (e.g., 'nose', 'rightear', 'leftear). Min number of bodyparts = 3.
    
    Returns:
        pd.DataFrame: DataFrame with features for the individual.
    """
    features = {}
    df_ind = df.loc[:, (slice(None), individual)]

    centroid = compute_centroid(df_ind, bp_list)
    body_length = compute_body_length(df_ind, anterior_bp, posterior_bp)

    centroid_speed = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
    centroid_speed = np.insert(centroid_speed, 0, 0)  # pad first frame

    features['centroid_x'] = centroid[:, 0]
    features['centroid_y'] = centroid[:, 1]
    features['body_length'] = body_length
    features['centroid_speed'] = centroid_speed
    
    assert len(bp_angle) == 3, "bp_angle must contain exactly three bodyparts."

    a = {
        'x': df.loc[:, (slice(None), individual, bp_angle[0], 'x')].values,
        'y': df.loc[:, (slice(None), individual, bp_angle[0], 'y')].values
    }
    b = {
        'x': df.loc[:, (slice(None), individual, bp_angle[1], 'x')].values,
        'y': df.loc[:, (slice(None), individual, bp_angle[1], 'y')].values
    }
    c = {
        'x': df.loc[:, (slice(None), individual, bp_angle[2], 'x')].values,
        'y': df.loc[:, (slice(None), individual, bp_angle[2], 'y')].values
    }

    head_orientation = compute_orientation(a, b, c)
    features['head_orientation'] = head_orientation

    features_df = pd.DataFrame(features)
    features_df['individual'] = individual

    return features_df


def compute_social_features(df, individuals, bp_list):
    """
    Compute social interaction features between individuals (IoU, distances, angles).
    
    Args:
        df (pd.DataFrame): Full dataframe.
        individuals (list): List of individuals (e.g., ['m1', 'm2', 'm3', 'm4']).
        bp_list (list): List of bodyparts for centroid and hulls.

    Returns:
        dict: Dictionary of feature DataFrames, one per individual.
    """
    hulls = {ind: get_hulls(df.loc[:, (slice(None), ind)], bp_list) for ind in individuals}
    centroids = {ind: compute_centroid(df.loc[:, (slice(None), ind)], bp_list) for ind in individuals}
    
    social_features = {}

    for ind in individuals:
        features = {}
        ref_hulls = hulls[ind]
        others = [hulls[other] for other in individuals if other != ind]
        other_labels = [other for other in individuals if other != ind]

        # IoU
        iou_df = compute_iou(ref_hulls, others, other_labels)
        for col in iou_df.columns:
            features[col] = iou_df[col]

        # Social distances and angles
        for other in other_labels:
            snout_self = df.loc[:, (slice(None), ind, 'snout')].values
            tailbase_other = df.loc[:, (slice(None), other, 'tailbase')].values
            centroid_self = centroids[ind]
            centroid_other = centroids[other]

            # Snout to tailbase distance between individuals
            snout_tailbase_dist = np.linalg.norm(snout_self - tailbase_other, axis=1)
            features[f'{ind}_snout_to_{other}_tailbase_distance'] = snout_tailbase_dist

            # Centroid to centroid distance between individuals
            centroid_dist = np.linalg.norm(centroid_self - centroid_other, axis=1)
            features[f'{ind}_centroid_to_{other}_centroid_distance'] = centroid_dist

            # Snout to centroid angle between individuals
            snout = {
                'x': df.loc[:, (slice(None), ind, 'nose', 'x')].values,
                'y': df.loc[:, (slice(None), ind, 'nose', 'y')].values
            }
            head = {
                'x': df.loc[:, (slice(None), ind, 'head', 'x')].values,
                'y': df.loc[:, (slice(None), ind, 'head', 'y')].values
            }
            centroid_o = {
                'x': centroid_other[:, 0],
                'y': centroid_other[:, 1]
            }
            snout_centroid_angle = compute_orientation(snout, head, centroid_o)
            features[f'{ind}_snout_to_{other}_centroid_angle'] = snout_centroid_angle

        features_df = pd.DataFrame(features)
        features_df['individual'] = ind
        social_features[ind] = features_df

    return social_features

def compute_full_features(df, individuals, bp_list, anterior_bp, posterior_bp):
    """
    Main function to compute both individual and social features, returns one combined DataFrame.
    """
    df = interpolate_bp(df)

    # Individual features
    individual_features = []
    for ind in individuals:
        ind_features = compute_individual_features(df, ind, bp_list, anterior_bp, posterior_bp)
        individual_features.append(ind_features)

    # Social features
    social_features_dict = compute_social_features(df, individuals, bp_list)

    # Merge individual + social
    full_features = []
    for ind, ind_df in zip(individuals, individual_features):
        social_df = social_features_dict[ind]
        merged = pd.concat([ind_df.reset_index(drop=True), social_df.drop(columns='individual').reset_index(drop=True)], axis=1)
        full_features.append(merged)

    full_df = pd.concat(full_features, axis=0).reset_index(drop=True)
    return full_df

