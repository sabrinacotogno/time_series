import numpy as np

from scipy.signal import savgol_filter

 

def smooth_timeseries(

    data: np.ndarray,

    method: str = "moving_average",

    window_size: int = 5,

    gaussian_std: float = 1.0,

    savgol_order: int = 3,

    min_periods: int = None,

    center: bool = True,

) -> np.ndarray:

    """

    Smooth multiple time series using various methods.

 

    Parameters:

    -----------

    data : np.ndarray

        Input time series data with shape (time_points, features)

    method : str

        Smoothing method ('moving_average', 'exponential', 'gaussian', 'savgol')

    window_size : int

        Size of the smoothing window

    gaussian_std : float

        Standard deviation for Gaussian smoothing

    savgol_order : int

        Polynomial order for Savitzky-Golay filter

    min_periods : int

        Minimum number of observations required for smoothing

    center : bool

        Whether to center the window (True) or use past values only (False)

 

    Returns:

    --------

    np.ndarray

        Smoothed time series with same shape as input

    """

    # Input validation

    if not isinstance(data, np.ndarray):

        raise TypeError("Input must be a numpy array")

 

    if len(data.shape) != 2:

        raise ValueError("Input must be 2D array with shape (time_points, features)")

 

    if window_size < 1:

        raise ValueError("Window size must be at least 1")

 

    if window_size > data.shape[0]:

        raise ValueError("Window size cannot be larger than number of time points")

 

    # Set default min_periods if not specified

    if min_periods is None:

        min_periods = window_size // 2

 

    # Initialize output array

    smoothed = np.zeros_like(data)

    n_timepoints, n_features = data.shape

 

    if method == "moving_average":

        for i in range(n_features):

            if center:

                # Centered window

                pad_width = window_size // 2

                padded = np.pad(data[:, i], (pad_width, pad_width), mode="edge")

                windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)

                smoothed[:, i] = np.mean(windows, axis=1)

            else:

                # Non-centered window (using only past values)

                # Initialize with the first value repeated

                smoothed[: window_size - 1, i] = data[0, i]

 

                # Calculate cumulative sum for efficient moving average

                cumsum = np.cumsum(np.insert(data[:, i], 0, 0))

                smoothed[window_size - 1 :, i] = (

                    cumsum[window_size:] - cumsum[:-window_size]

                ) / window_size

 

    elif method == "exponential":

        # Note: Exponential smoothing is inherently forward-looking

        # and cannot be centered

        alpha = 2 / (window_size + 1)

        for i in range(n_features):

            smoothed[:, i] = data[:, i].copy()

            for t in range(1, n_timepoints):

                smoothed[t, i] = alpha * data[t, i] + (1 - alpha) * smoothed[t - 1, i]

 

    elif method == "gaussian":

        # Create Gaussian window

        if center:

            window = np.arange(-window_size // 2, window_size // 2 + 1)

        else:

            window = np.arange(window_size)

 

        gaussian_window = np.exp(-(window**2) / (2 * gaussian_std**2))

        gaussian_window = gaussian_window / gaussian_window.sum()

 

        for i in range(n_features):

            if center:

                smoothed[:, i] = np.convolve(data[:, i], gaussian_window, mode="same")

            else:

                # Use 'valid' mode and pad the beginning

                conv_result = np.convolve(data[:, i], gaussian_window, mode="valid")

                pad_width = window_size - 1

                smoothed[:, i] = np.pad(conv_result, (pad_width, 0), mode="edge")

 

    elif method == "savgol":

        if window_size % 2 == 0:

            window_size += 1  # Savgol requires odd window size

 

        for i in range(n_features):

            smoothed[:, i] = savgol_filter(

                data[:, i],

                window_size,

                savgol_order,

                mode="nearest",

                deriv=0,

                delta=1.0,

                axis=-1,

            )

 

    else:

        raise ValueError(

            "Method must be one of: 'moving_average', 'exponential', "

            "'gaussian', 'savgol'"

        )

 

    return smoothed