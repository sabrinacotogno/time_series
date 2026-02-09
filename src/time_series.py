from typing import List, Tuple
import numpy as np

 

def find_consecutive_nans(feature_data: np.ndarray) -> list:

    """

    Find all sequences of consecutive NaN values in a 1D array.

    Returns list of tuples with (start_idx, length).

 

    Parameters

    ----------

    feature_data : np.ndarray

        1D array to search for consecutive NaN values

 

    Returns

    -------

    list

        List of tuples with (start_idx, length)

    """

    # Convert to numpy array if not already

    if not isinstance(feature_data, np.ndarray):

        feature_data = np.array(feature_data)

 

    is_nan = np.isnan(feature_data)

    if not np.any(is_nan):

        return []

 

    # Raise an error if it's not 1D array

    if is_nan.ndim != 1:

        raise ValueError("Input array must be 1D")

 

    # Find the differences between consecutive elements

    diff = np.diff(np.where(is_nan)[0])

    # Find where the difference is greater than 1 (breaks in consecutive sequences)

    breaks = np.where(diff > 1)[0]

 

    # Get the start indices and lengths of consecutive sequences

    nan_sequences = []

    current_start = np.where(is_nan)[0][0]

 

    for i, break_idx in enumerate(breaks):

        length = np.where(is_nan)[0][break_idx] - current_start + 1

        nan_sequences.append((current_start, length))

        current_start = np.where(is_nan)[0][break_idx + 1]

 

    # Handle the last sequence

    length = np.where(is_nan)[0][-1] - current_start + 1

    nan_sequences.append((current_start, length))

 

    return nan_sequences

 

def remove_consecutive_nan_from_array(

    data: np.ndarray,

    max_consecutive_nans: int = 4,

    max_removed_nan_fraction: float = 0.1,

) -> Tuple[np.ndarray, dict]:

    """

    Clean NaN values from a 3D numpy array by processing each sample separately

    but coordinating feature removal across all samples.

 

    Parameters:

    -----------

    data : numpy.ndarray

        3D array with shape (n_samples, n_time_steps, n_features)

    max_consecutive_nans : int, default=5

        Maximum allowed number of consecutive NaN values in a feature

    max_removed_nan_fraction : float, default=0.1

        Maximum allowed fraction of samples removed from the array before

        raising an error. Removed samples could contain either too much

        consecutive NaNs ot NaNs at boundaries

 

    Returns:

    --------

    Tuple[numpy.ndarray, dict]

        - Cleaned 3D array with shape (n_samples, n_time_steps, n_features_new)

        - Dictionary containing metadata about the cleaning process:

            - 'original_shape': original array shape

            - 'final_shape': final array shape

            - 'removed_features': list of removed features

            - 'features_with_boundary_nans': dict mapping sample_idx to list of features

            - 'interpolated_positions': dict mapping (sample_idx, feature_idx) to positions

            - 'boundary_nan_fractions': dict mapping feature_idx to boundary NaN fractions

 

    Raises:

    -------

    ValueError

        - If input is not 3D array

        - If any feature has too many consecutive NaNs

        - If too many features have NaNs at boundaries

    """

    # Input validation

    if not isinstance(data, np.ndarray):

        raise TypeError("Input 'data' must be a numpy array")

    if data.ndim != 3:

        raise ValueError("Input array must be 3-dimensional")

 

    n_samples, n_timesteps, n_features = data.shape

 

    # Initialize metadata

    metadata = {

        "original_shape": data.shape,

        "final_shape": None,

        "removed_samples": set(),

        "samples_with_boundary_nans": {},

        "interpolated_positions": {},

        "boundary_nan_fractions": {},

    }

 

    # First pass: identify features to remove across all samples

    for feature_idx in range(n_features):

        feature_data = data[:, :, feature_idx]

        boundary_samples = []

 

        # Check for boundary NaNs

        for sample_idx in range(n_samples):

            sample_data = feature_data[sample_idx]

 

            # Check for boundary NaNs

            if np.isnan(sample_data[0]) or np.isnan(sample_data[-1]):

                boundary_samples.append(sample_idx)

 

            # Check for consecutive NaNs

            nan_sequences = find_consecutive_nans(sample_data)

            for start_idx, length in nan_sequences:

                if length > max_consecutive_nans:

                    metadata["removed_samples"].add(sample_idx)

 

        # Store boundary features for this sample

        metadata["samples_with_boundary_nans"][feature_idx] = boundary_samples

        boundary_nan_fraction = len(boundary_samples) / n_samples

        metadata["boundary_nan_fractions"][feature_idx] = boundary_nan_fraction

 

        # Add boundary samples to removal set

        metadata["removed_samples"].update(boundary_samples)

 

    # Convert set to sorted list for consistent ordering

    samples_to_remove = sorted(metadata["removed_samples"])

    removed_nan_fraction = len(samples_to_remove) / n_samples

 

    # Check boundary NaN fraction

    if removed_nan_fraction > max_removed_nan_fraction:

        raise ValueError(

            f"There are two many removed samples from the array: "

            f"({removed_nan_fraction:.2%}) samples have been removed, "

            f"exceeding maximum allowed fraction ({max_removed_nan_fraction:.2%})"

        )

 

    # Create cleaned data array

    if samples_to_remove:

        data_clean = np.delete(data, samples_to_remove, axis=0)

    else:

        data_clean = data.copy()

 

    # Second pass: interpolate remaining NaNs

    for feature_idx in range(n_features):

        for sample_idx in range(data_clean.shape[0]):  # Use new feature count

            sample_data = data_clean[sample_idx, :, feature_idx]

 

            if np.any(np.isnan(sample_data)):

                # Find positions for interpolation

                nan_sequences = find_consecutive_nans(sample_data)

 

                # Store interpolation positions in metadata

                key = (feature_idx, sample_idx)

                metadata["interpolated_positions"][key] = [

                    (start, start + length - 1) for start, length in nan_sequences

                ]

 

                # Perform interpolation

                non_nan_idx = np.where(~np.isnan(sample_data))[0]

                data_clean[sample_idx, :, feature_idx] = np.interp(

                    np.arange(n_timesteps), non_nan_idx, sample_data[non_nan_idx]

                )

 

    metadata["final_shape"] = data_clean.shape

 

    return data_clean, metadata

 

def remove_nan_timestamps(

    data: np.ndarray, max_allowed_nan_fraction: float = 0.3

) -> Tuple[List[np.ndarray], dict]:

    """

    Clean NaN values from a 3D numpy array by processing each sample independently

    and removing timestamps where any feature has NaN values.

 

    Parameters:

    -----------

    data : numpy.ndarray

        3D array with shape (n_samples, n_time_steps, n_features)

    max_allowed_nan_fraction : float, default=0.3

        Maximum allowed fraction of timestamps that can be removed from a sample

        before raising an error

 

    Returns:

    --------

    Tuple[List[numpy.ndarray], dict]

        - List of 2D arrays, each with shape (n_remaining_timestamps, n_features)

        - Dictionary containing metadata about the cleaning process:

            - 'original_shape': original array shape

            - 'final_shapes': list of shapes for each cleaned sample

            - 'removed_timestamps': dict mapping sample_idx to list of removed timestamps

            - 'removed_fractions': dict mapping sample_idx to fraction of removed timestamps

            - 'removed_samples': list of samples that exceeded max_allowed_nan_fraction

 

    Raises:

    -------

    TypeError: If input is not a numpy array

    ValueError: If input is not 3D array

    """

    # Input validation

    if not isinstance(data, np.ndarray):

        raise TypeError("Input 'data' must be a numpy array")

    if data.ndim != 3:

        raise ValueError("Input array must be 3-dimensional")

 

    n_samples, n_timesteps, n_features = data.shape

 

    # Initialize output and metadata

    cleaned_samples = []

    metadata = {

        "original_shape": data.shape,

        "final_shapes": [],

        "removed_timestamps": {},

        "removed_fractions": {},

        "removed_samples": [],

    }

 

    # Process each sample independently

    for sample_idx in range(n_samples):

        sample_data = data[sample_idx]  # Shape: (n_timesteps, n_features)

 

        # Find timestamps where any feature has NaN

        nan_mask = np.any(np.isnan(sample_data), axis=1)

        valid_timestamps = ~nan_mask

 

        # Calculate fraction of removed timestamps

        removed_fraction = np.mean(nan_mask)

        metadata["removed_fractions"][sample_idx] = removed_fraction

 

        # Store removed timestamp indices

        metadata["removed_timestamps"][sample_idx] = np.where(nan_mask)[0].tolist()

 

        # Check if sample exceeds max allowed NaN fraction

        if removed_fraction > max_allowed_nan_fraction:

            metadata["removed_samples"].append(sample_idx)

            metadata["final_shapes"].append((0, 0))  # Sample will be removed

            continue

 

        # Keep only timestamps where all features have valid values

        cleaned_sample = sample_data[valid_timestamps]

        cleaned_samples.append(cleaned_sample)

        metadata["final_shapes"].append(cleaned_sample.shape)

 

    if not cleaned_samples:

        raise ValueError("All samples exceeded maximum allowed NaN fraction")

 

    return cleaned_samples, metadata