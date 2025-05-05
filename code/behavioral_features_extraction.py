"""
Behavioral feature extraction functions for the analysis of social hierarchy behavior
@author Anna Teruel-Sanchis, April 2025
"""
import os
import glob
import numpy as np 
import pandas as pd
from shapely import polygons
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from scipy.interpolate import CubicSpline

def rename_h5s(h5_path, output_path, rename_map):
    """
    Rename individuals of a MultiIndex column in an HDF5 file.

    Args:
        h5_path (str): Path to input HDF5 file.
        output_path (str): Path to save the modified HDF5.
        rename_map (dict): Mapping from old to new individual names, e.g., {'m1': 'ind_1'}
    """
    df = pd.read_hdf(h5_path)

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("The DataFrame does not have a MultiIndex.")

    new_columns = df.columns.to_frame(index=False)
    new_columns['individuals'] = new_columns['individuals'].map(rename_map).fillna(new_columns['individuals'])
    df.columns = pd.MultiIndex.from_frame(new_columns)

    df.to_hdf(output_path, key='df', mode='w', format='fixed')
    print(f"Saved to: {output_path}")

def check_nans(directory, 
               file_format='csv'):
    """
    Check percentage of NaNs in all files in a directory (either .h5 or .csv).
    
    Args:
        directory (str): Path to the directory containing the files.
        file_format (str): File format to process ('csv' or 'h5').

    Prints:
        Filename and percentage of NaNs for each file.
    """
    if file_format not in ['csv', 'h5']:
        raise ValueError("file_format must be 'csv' or 'h5'.")

    files = [f for f in os.listdir(directory) if f.endswith(f'.{file_format}')]
    print(f"Found {len(files)} .{file_format} files in {directory}")

    for file in sorted(files):
        file_path = os.path.join(directory, file)
        try:
            if file_format == 'csv':
                df = pd.read_csv(file_path)
            elif file_format == 'h5':
                df = pd.read_hdf(file_path)
            
            total_nans = df.isna().sum().sum()
            total_elements = df.size
            percent_nans = (total_nans / total_elements) * 100
            print(f"{file}: {percent_nans:.4f}% NaNs")
        
        except Exception as e:
            print(f"Could not process {file}: {e}")

def apply_arima(series, 
                order):
        """
        Apply ARIMA interpolation to a time series.
        ARIMA models the relationship between a variable and its own past values. 
        It works by forecasting future points based on the values that came before them.

        Args:
            series (pd.Series): Time series data to interpolate.
            order (tuple): ARIMA model order (p, d, q).

        Returns:
            pd.Series: Interpolated time series.
        """        
        not_na = series.dropna()
        if len(not_na) < 2:  
            return series
        
        model = ARIMA(not_na, order=order)
        model_fitted = model.fit()
        forecast = model_fitted.predict(start=not_na.index[0], end=not_na.index[-1])
        
        return forecast
    
def apply_spline(series):
        """
        Apply Spline interpolation to a time series.
        Spline interpolation fits a smooth curve through the data points. 
        It’s particularly useful when the data exhibits smooth transitions and 
        you don’t expect drastic changes between points.
        
        Args:
            series (pd.Series): Time series data to interpolate.
            
        Returns:
            pd.Series: Interpolated time series.
        """        
        not_na = series.dropna()
        if len(not_na) < 2:  
            return series
        cs = CubicSpline(not_na.index, not_na.values, bc_type='natural')
        return cs(series.index)
            
def interpolate_data(df, 
                     method='spline', 
                     order=(1, 1, 0)):
    """
    Interpolate missing data using either ARIMA or Spline interpolation.
    Testing on different datasets, Spline interpolation works better. 

    Args:
        df (pd.DataFrame): Dataframe containing the x, y coordinates with missing values.
        method (str): Interpolation method. Choose 'arima' or 'spline'. Default is 'spline'.
        order (tuple): ARIMA model order (p, d, q). Used only if method='arima'. Default is (1, 1, 0).

    Returns:
        pd.DataFrame: DataFrame with interpolated x, y coordinates.
    """
    df_percent = (df.isna().sum().sum() / df.size) * 100
    print(f"Percentage of NaNs in original DataFrame: {df_percent:.2f}%")

    x = df.loc[:, (slice(None), slice(None), slice(None), 'x')]
    y = df.loc[:, (slice(None), slice(None), slice(None), 'y')]
    likelihood = df.loc[:, (slice(None), slice(None), slice(None), 'likelihood')]
    
    scorer = x.columns.levels[0][0]
    for individual in x.columns.levels[1]:
        for bodypart in x.columns.levels[2]:
            if method == 'arima':
                x_interp = apply_arima(x.loc[:, (scorer, individual, bodypart, 'x')], order)
                y_interp = apply_arima(y.loc[:, (scorer, individual, bodypart, 'y')], order)
                likelihood_interp = apply_arima(likelihood.loc[:, (scorer, individual, bodypart, 'likelihood')], order)    
            elif method == 'spline':
                x_interp = apply_spline(x.loc[:, (scorer, individual, bodypart, 'x')])
                y_interp = apply_spline(y.loc[:, (scorer, individual, bodypart, 'y')])
                likelihood_interp = apply_spline(likelihood.loc[:, (scorer, individual, bodypart, 'likelihood')])
            else:
                raise ValueError("Invalid method. Choose 'arima' or 'spline'.")
            df.loc[:, (scorer, individual,bodypart, 'x')] = x_interp
            df.loc[:, (scorer, individual, bodypart, 'y')] = y_interp
            df.loc[:, (scorer, individual, bodypart, 'likelihood')] = likelihood_interp

    df_percent_after = (df.isna().sum().sum() / df.size) * 100
    print(f"Percentage of NaNs after interpolation: {df_percent_after:.2f}%")
    return df

def get_hulls(data, 
              bp_list):
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

def get_iou(hulls_ref, 
            hulls_others, 
            other_labels):
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

def get_centroid(df, 
                bodyparts):
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

def get_body_length(df, 
                    anterior_bp, 
                    posterior_bp):
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

def get_orientation(a, 
                    b, 
                    c):
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
        angles = get_orientation(a, b, c)
        print(angles)  # Output: array of angles in degrees
    """
    crossprod = (c['x']-b['x'])*(b['y']-a['y']) - (c['y']-b['y'])*(b['x']-a['x'])
    dotprod = (c['x']-b['x'])*(b['x']-a['x']) + (c['y']-b['y'])*(b['y']-a['y'])
    angles = np.arctan2(crossprod, dotprod)
    angles = np.where(angles < 0, angles * -1, angles)
    angles = np.degrees(angles)
    return angles

def get_speed(df, 
              bp):
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

def get_relative_velocity(source, 
                          target, 
                          fps=30): 
    """
    Compute the relative velocity of a source point with respect to a target point.

    Relative velocity is defined as the velocity of the source projected onto the 
    unit vector pointing from the source to the target. Positive values indicate 
    approaching motion; negative values indicate withdrawal.

    Args:
        source (np.ndarray): Array of shape (n_frames, 2) with x, y coordinates of the source body part (e.g. centroid m1).
        target (np.ndarray): Array of shape (n_frames, 2) with x, y coordinates of the target body part (e.g. centroid m2).
        dt (float): Time interval between frames in seconds. Default is 1/30 (30 FPS).

    Returns:
        np.ndarray: Array of shape (n_frames,) containing the relative velocity at each frame.
    """
    v_disp = np.diff(source, axis=0, prepend=source[:1]) / fps #using prepend to avoid losing the first frame's data 
    direction = target - source
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms[norms == 0] = 1  # evita división por cero
    direction_unit = direction / norms
    # direction_unit = direction / np.linalg.norm(direction, axis=1, keepdims=True) #euclidian norm (magnitude) of the vector
    approach_velocity = np.sum(v_disp * direction_unit, axis=1)
    return approach_velocity

def get_hysteresis_interaction(distances, 
                               angles, 
                               d_enter, 
                               d_exit, 
                               a_enter, 
                               a_exit, 
                               allowed_miss):
    """
    Identify interaction periods based on distance and angle thresholds using a hysteresis model.
    A hysteresis model is a system that exhibits different behaviors depending on its current state and past states. 
    It's a system built under the assumption that a pcurrent state depends not inly on the current input, but also on the system's past states. 
    The idea of applying this model to define interaction periods is to prevent rapid toggling between states due to 
    noise or temporary fluctuations in distance or angle between pairs of individuals. 

    Interaction starts when both distance and angle are below their respective start thresholds.
    It continues as long as they remain below the stop thresholds, allowing for a number of 
    temporary misses (frames not meeting the criteria) defined by `allowed_miss`. Defining the thresholds depends on the 
    specific behavior being analyzed. But as a guideline: d_enter should represent the maximum distance at which interaction 
    is initiated, e.g. max distance at which two individuals are considered close 

    Args:
        distances (np.ndarray): Array of distances between two body parts per frame.
        angles (np.ndarray): Array of relative angles (in degrees) between body parts per frame.
        d_enter (float): Distance threshold to initiate interaction.
        d_exit (float): Distance threshold to maintain interaction.
        a_enter (float): Angle threshold to initiate interaction.
        a_exit (float): Angle threshold to maintain interaction.
        allowed_miss (int): Number of consecutive frames allowed to miss the criteria before ending the interaction.

    Returns:
        np.ndarray: Boolean array of the same length indicating interaction state (True/False) per frame.
    """
    interacting = False
    miss_count = 0
    result = []
    for d, a in zip(distances, angles):
        if not interacting:
            if d < d_enter and a < a_enter:
                interacting = True
                miss_count = 0
        else:
            if d < d_exit and a < a_exit:
                miss_count = 0
            else:
                miss_count += 1
                if miss_count > allowed_miss:
                    interacting = False
        result.append(interacting)
    return np.array(result, dtype=int)

def get_unsigned_angle(v1, 
                       v2): 
    """
    Compute the unsigned angle between two vectors in degrees.
    The unsigned angle is the angle between two vectors without considering their direction.
    For our purpose, we want to compute the angle between: 
    1. stimulus body axis: a vector representing the orientation of the stimulus indivodual, calculated as a vector from its snout to its tailbase
    2. subject direction: a vector pointning from the stimulus centrpoiod to the subject centroid.
    
    This approach helps us measure analyze how aligned the subject's movement is relative to the stimulus body orientation. 
    
    With this function, we can also compute mutual alignment of axes between two animals.
    
    Args:
        v1 (np.ndarray): First vector of shape (n_frames, 2).
        v2 (np.ndarray): Second vector of shape (n_frames, 2).
    
    Returns:
        np.ndarray: Array of angles in degrees for each frame.
    """
    dot_product = np.einsum('ij,ij->i', v1, v2)
    # norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    norms_v1 = np.linalg.norm(v1, axis=1)
    norms_v2 = np.linalg.norm(v2, axis=1)
    norms = norms_v1 * norms_v2
    zero_mask = norms == 0
    norms[zero_mask] = 1
    cos_theta = np.clip(dot_product / norms, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    angles[zero_mask] = 0.0
    return np.degrees(angles)

def get_angular_velocity(head_angles, 
                         fps=30):
    """
    Compute angular velocity from head angles.
    How quickly an angle changes over time. 
    It is useful for analyzing head movements or trunk orientation, which can represent attention shifts or reactions to stimuli.
    
    Args:
        head_angles (np.ndarray): Array of head angles in degrees for each frame.
        dt (float): Time interval between frames in seconds. Default is 1/30 (30 FPS).
    
    Returns:
        np.ndarray: Array of angular velocities in degrees per second for each frame.
    """
    angle_diff = np.diff(head_angles, prepend=head_angles[0])
    angle_diff = (angle_diff + 180) % 360 - 180  
    angular_velocity = np.abs(angle_diff) / fps  
    return angular_velocity  

def get_axis_orientation(df, 
                         anterior_bp, 
                         posterior_bp):
    """
    Compute the orientation of a line segment defined by two points, relative to the horizontal x-axis. 
    
    Args:
        df (pd.DataFrame): DataFrame containing x and y coordinates of body parts.
        anterior_bp (str): Name of the anterior body part.
        posterior_bp (str): Name of the posterior body part.

    Returns:
        np.ndarray: 1D array of orientation angles in degrees for each frame.
    """
    anterior = df.loc[:, (slice(None), slice(None), anterior_bp)].values
    posterior = df.loc[:, (slice(None), slice(None), posterior_bp)].values 

    delta = posterior - anterior
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    return np.degrees(angles)


def get_individual_features(df, 
                            individual, 
                            bp_list,
                            head_bp,  
                            anterior_bp, 
                            posterior_bp, 
                            bp_angle):
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

    centroid = get_centroid(df_ind, bp_list)
    body_length = get_body_length(df_ind, anterior_bp, posterior_bp)
    body_orientation = get_axis_orientation(df_ind, anterior_bp, posterior_bp)
    head = get_centroid(df_ind, head_bp)

    centroid_speed = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
    centroid_speed = np.insert(centroid_speed, 0, 0)  # pad first frame
    head_speed = np.linalg.norm(np.diff(head, axis=0), axis=1)
    head_speed = np.insert(head_speed, 0, 0)  # pad first frame
    
    assert len(bp_angle) == 3, "bp_angle must contain exactly three bodyparts."

    scorer = df.columns.get_level_values('scorer')[0]
    a = {
        'x': df.loc[:, (scorer, individual, bp_angle[0], 'x')].values,
        'y': df.loc[:, (scorer, individual, bp_angle[0], 'y')].values
    }
    b = {
        'x': df.loc[:, (scorer, individual, bp_angle[1], 'x')].values,
        'y': df.loc[:, (scorer, individual, bp_angle[1], 'y')].values
    }
    c = {
        'x': df.loc[:, (scorer, individual, bp_angle[2], 'x')].values,
        'y': df.loc[:, (scorer, individual, bp_angle[2], 'y')].values
    }

    head_orientation = get_orientation(a, b, c)
    head_angular_velocity = get_angular_velocity(head_orientation)
    body_angular_velocity = get_angular_velocity(body_orientation)
    
    #total individual features: 7
    features['body_length'] = body_length
    features['body_orientation'] = body_orientation
    features['centroid_speed'] = centroid_speed
    features['head_speed'] = head_speed
    features['head_orientation'] = head_orientation
    features['head_angular_velocity'] = head_angular_velocity
    features['body_angular_velocity'] = body_angular_velocity

    features_df = pd.DataFrame(features)

    return features_df


def get_social_features(df, 
                        individuals, 
                        bp_list, 
                        d_enter=50,
                        d_exit=70,
                        a_enter=80,
                        a_exit=90,
                        allowed_miss=1):
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
    centroids = {ind: get_centroid(df.loc[:, (slice(None), ind)], bp_list) for ind in individuals}
    
    social_features = {}

    for ind in individuals:
        features = {}
        ref_hulls = hulls[ind]
        # others = [hulls[other] for other in individuals if other != ind]
        # other_labels = [other for other in individuals if other != ind]
        others = [hulls[other] for other in individuals]
        other_labels = individuals
        
        # IoU
        iou_df = get_iou(ref_hulls, others, other_labels)
        for col in iou_df.columns:
            features[col] = iou_df[col]

        # Social distances and angles
        for other in other_labels:
            snout_self = df.loc[:, (slice(None), ind, 'snout', ['x', 'y'])].values
            tailbase_self = df.loc[:, (slice(None), ind, 'tailbase', ['x', 'y'])].values
            snout_other = df.loc[:, (slice(None), other, 'snout', ['x', 'y'])].values
            tailbase_other = df.loc[:, (slice(None), other, 'tailbase', ['x', 'y'])].values
            centroid_self = centroids[ind]
            centroid_other = centroids[other]

            snout_tailbase_dist = np.linalg.norm(snout_self - tailbase_other, axis=1) # Snout to tailbase distance between individuals
            centroid_dist = np.linalg.norm(centroid_self - centroid_other, axis=1) # Centroid to centroid distance between individuals
            snout_dist = np.linalg.norm(snout_self - snout_other, axis=1) # Snout to snout distance between individuals

            scorer = df.columns.get_level_values('scorer')[0]
            snout = {
                'x': df.loc[:, (scorer, ind, 'snout', 'x')].values,
                'y': df.loc[:, (scorer, ind, 'snout', 'y')].values
            }
            center = {
                'x': df.loc[:, (scorer, ind, 'center', 'x')].values,
                'y': df.loc[:, (scorer, ind, 'center', 'y')].values
            }
            centroid_o = {
                'x': centroid_other[:, 0],
                'y': centroid_other[:, 1]
            }
            snout_centroid_angle = get_orientation(snout, center, centroid_o) # Snout to centroid angle between individuals

            rel_velocity = get_relative_velocity(centroid_self, centroid_other) # Relative velocity between individuals
            
            axis_self = snout_self - tailbase_self
            axis_other = snout_other - tailbase_other
            direction_vector = centroid_other - centroid_self
            axis_to_angle = get_unsigned_angle(axis_self, direction_vector) # Angle between the axis of the individual and the direction vector to the other individual
            mutual_axis_angle = get_unsigned_angle(axis_self, axis_other) # Angle between the axes of both individuals
            
            interact_flag = get_hysteresis_interaction(
                distances=centroid_dist,
                angles=snout_centroid_angle,
                d_enter=d_enter,  # example threshold, adjust if needed #TODO Ask Robin for these thresholds in pixels
                d_exit=d_exit,   # example threshold, adjust if needed #TODO Ask Robin for these thresholds in pixels
                a_enter=a_enter,   # example threshold, adjust if needed, copied from Robin's matlab script
                a_exit=a_exit,    # example threshold, adjust if needed, copied form Robin's matlab script
                allowed_miss=1  # number of frames allowed to miss criteria before ending interaction, copied from Robin's matlab script. 
            )
            
            #adding all features to the features dict
            # total number of social features: 8
            features[f'{ind}_snout_to_{other}_tailbase_distance'] = snout_tailbase_dist
            features[f'{ind}_centroid_to_{other}_centroid_distance'] = centroid_dist
            features[f'{ind}_snout_to_{other}_snout_distance'] = snout_dist
            features[f'{ind}_snout_to_{other}_centroid_angle'] = snout_centroid_angle
            features[f'{ind}_relative_velocity_to_{other}'] = rel_velocity
            features[f'{ind}_axis_to_{other}_angle'] = axis_to_angle
            features[f'{ind}_mutual_axis_angle_with_{other}'] = mutual_axis_angle
            features[f'{ind}_interaction_with_{other}'] = interact_flag
            

        #mean centroid distance to all other animals as a global interaction metric
        centroid_dist_cols = [f'{ind}_centroid_to_{other}_centroid_distance' for other in other_labels]
        features['avg_centroid_distance'] = pd.DataFrame({col: features[col] for col in centroid_dist_cols}).mean(axis=1)

        features_df = pd.DataFrame(features)
        social_features[ind] = features_df

    return social_features


def get_all_features(df_path,
                     file_format, 
                     individuals, 
                     bp_list, 
                     head_bp,
                     anterior_bp, 
                     posterior_bp, 
                     bp_angle, 
                     save_dir=None, 
                     out_format='csv',
                     interpolate_method ='spline'):
    """
    Main function to compute both individual and social features, returns one combined DataFrame.
    
    Args:
        df_path (str): Path to the HDF5 file containing the DataFrame.
        individuals (list): List of individuals (e.g., ['m1', 'm2', 'm3', 'm4']).
        bp_list (list): List of bodyparts for centroid and hulls.
        head_bp (list): List of bodyparts for head centroid calculation (e.g., ['snout', 'rightear', 'leftear']).
        anterior_bp (str): Name of anterior bodypart (e.g., 'nose').
        posterior_bp (str): Name of posterior bodypart (e.g., 'tailbase').
        bp_angle (list): Body parts to compute angles for (e.g., ['nose', 'rightear', 'leftear']). Min number of bodyparts = 3.
        save_dir (str): Directory to save the output files. If None, files are not saved.
        file_format (str): Format to save the output files ('csv' or 'h5'). Default is 'csv'.
    
    Returns:
        dict: Dictionary of feature DataFrames, one per individual.
    """
    if file_format == 'csv':
        df = pd.read_csv(df_path, header=[0,1,2,3]) #read as multi-index, number of levels in columns
    elif file_format == 'h5':
        df = pd.read_hdf(df_path)
    else:
        raise ValueError("file_format must be 'csv' or 'h5'")
    df = interpolate_data(df, interpolate_method)
    base_filename = os.path.splitext(os.path.basename(df_path))[0]

    # Individual features
    individual_features = []
    for ind in individuals:
        ind_features = get_individual_features(df, ind, bp_list, head_bp, anterior_bp, posterior_bp, bp_angle)
        individual_features.append(ind_features)

    # Social features
    social_features_dict = get_social_features(df, individuals, bp_list)

    # Save
    all_features = {}
    for ind, ind_df in zip(individuals, individual_features):
        social_df = social_features_dict[ind]
        merged = pd.concat([ind_df.reset_index(drop=True), social_df.reset_index(drop=True)], axis=1)

        # Save per individual
        if save_dir is not None:
            filename = os.path.join(save_dir, f"{base_filename}_{ind}.{out_format}")
            if out_format == 'csv':
                merged.to_csv(filename, index=False)
            elif out_format == 'h5':
                merged.to_hdf(filename, key='df', mode='w')
            else:
                raise ValueError("file_format must be 'csv' or 'h5'")

        all_features[ind] = merged

def batch_features(input_dir, 
                   file_format, 
                   individuals, 
                   bp_list, 
                   head_bp,
                   anterior_bp, 
                   posterior_bp, 
                   bp_angle, 
                   save_dir, 
                   out_format):
    """
    Batch process all .h5 files in a directory with get_all_features.

    Args:
        input_dir (str): Directory containing .h5 files.
        individuals (list): List of individuals (e.g., ['m1', 'm2', 'm3', 'm4']).
        bp_list (list): List of bodyparts for centroid and hulls.
        anterior_bp (str): Name of anterior bodypart (e.g., 'nose').
        posterior_bp (str): Name of posterior bodypart (e.g., 'tailbase').
        bp_angle (list): List of body parts for orientation calculation (e.g., ['snout', 'rightear', 'leftear']).
        save_dir (str, optional): Directory to save outputs. If None, saves next to each input file.
        file_format (str): 'csv' or 'h5' output format.
    
    Returns:
        dict: Dictionary where keys are input filenames and values are dicts of feature DataFrames per individual.
    """
    if file_format == 'h5':
        files = glob.glob(os.path.join(input_dir, '*.h5'))
    elif file_format == 'csv':
        files = glob.glob(os.path.join(input_dir, '*.csv'))
    all_results = {}

    print(f"Found {len(files)} files to process.")

    for h5_file in files:
        print(f"Processing file: {os.path.basename(h5_file)}")

        if save_dir is None:
            save_subdir = os.path.dirname(h5_file)  
        else:
            save_subdir = save_dir
            os.makedirs(save_subdir, exist_ok=True)

        results = get_all_features(
            df_path=h5_file,
            file_format = file_format,
            individuals=individuals,
            bp_list=bp_list,
            head_bp=head_bp,
            anterior_bp=anterior_bp,
            posterior_bp=posterior_bp,
            bp_angle=bp_angle,
            save_dir=save_subdir,
            out_format=out_format,
        )

        all_results[h5_file] = results

    print("Batch processing complete.")

