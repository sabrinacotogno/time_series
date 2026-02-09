"""
BatchGrouper: sklearn-style estimator for grouping batches based on 
multivariate time series correlation.

Author: Custom Implementation
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, Literal
import networkx as nx
from scipy.stats import pearsonr


class BatchGrouper(BaseEstimator, TransformerMixin):
    """
    Groups batches of multivariate time series based on correlation patterns.
    
    This estimator computes correlation matrices for each parameter across batches,
    combines them using a minimum threshold strategy, and creates non-overlapping
    groups using graph-based clustering.
    
    Parameters
    ----------
    corr_threshold : float, default=0.7
        Minimum correlation threshold to retain an edge between batches.
        Must be between -1 and 1.
    
    grouping_method : {'connected_components', 'louvain', 'label_propagation'}, default='connected_components'
        Algorithm to use for creating groups from the correlation graph:
        - 'connected_components': Finds fully connected components
        - 'louvain': Community detection using Louvain algorithm
        - 'label_propagation': Semi-supervised label propagation
    
    min_group_size : int, default=2
        Minimum number of batches required to form a group.
        Smaller groups will be labeled as outliers (-1).
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_batches,)
        Group labels for each batch. -1 indicates outlier/ungrouped batches.
    
    n_groups_ : int
        Number of groups found (excluding outliers).
    
    correlation_matrices_ : list of ndarray
        List of correlation matrices, one per parameter.
        Each matrix has shape (n_batches, n_batches).
    
    global_correlation_matrix_ : ndarray of shape (n_batches, n_batches)
        Combined correlation matrix used for grouping.
    
    graph_ : networkx.Graph
        The graph representation of batch relationships.
    
    Examples
    --------
    >>> import numpy as np
    >>> from batch_grouper import BatchGrouper
    >>> 
    >>> # Simulate data: 10 batches, 3 parameters, 50 timepoints each
    >>> data = np.random.randn(10, 3, 50)
    >>> 
    >>> # Create and fit the grouper
    >>> grouper = BatchGrouper(corr_threshold=0.6, min_group_size=2)
    >>> grouper.fit(data)
    >>> 
    >>> # Get group labels
    >>> labels = grouper.labels_
    >>> 
    >>> # Predict group for new batches
    >>> new_data = np.random.randn(5, 3, 50)
    >>> new_labels = grouper.predict(new_data)
    """
    
    def __init__(
        self,
        corr_threshold: float = 0.7,
        grouping_method: Literal['connected_components', 'louvain', 'label_propagation'] = 'connected_components',
        min_group_size: int = 2
    ):
        self.corr_threshold = corr_threshold
        self.grouping_method = grouping_method
        self.min_group_size = min_group_size
    
    def _validate_input(self, X: np.ndarray) -> None:
        """Validate input data format."""
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if X.ndim != 3:
            raise ValueError(
                f"Input must be 3-dimensional (n_batches, n_params, n_timepoints), "
                f"got shape {X.shape}"
            )
        
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 batches for grouping")
        
        if X.shape[2] < 2:
            raise ValueError("Need at least 2 timepoints for correlation computation")
    
    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        if not -1 <= self.corr_threshold <= 1:
            raise ValueError(
                f"corr_threshold must be between -1 and 1, got {self.corr_threshold}"
            )
        
        if self.min_group_size < 1:
            raise ValueError(
                f"min_group_size must be at least 1, got {self.min_group_size}"
            )
        
        valid_methods = ['connected_components', 'louvain', 'label_propagation']
        if self.grouping_method not in valid_methods:
            raise ValueError(
                f"grouping_method must be one of {valid_methods}, "
                f"got '{self.grouping_method}'"
            )
    
    def _compute_parameter_correlation_matrix(
        self, 
        X: np.ndarray, 
        param_idx: int
    ) -> np.ndarray:
        """
        Compute pairwise Pearson correlation matrix for a single parameter.
        
        Parameters
        ----------
        X : ndarray of shape (n_batches, n_params, n_timepoints)
            Input data
        param_idx : int
            Index of the parameter to compute correlations for
        
        Returns
        -------
        corr_matrix : ndarray of shape (n_batches, n_batches)
            Correlation matrix for the specified parameter
        """
        n_batches = X.shape[0]
        corr_matrix = np.zeros((n_batches, n_batches))
        
        # Extract time series for this parameter across all batches
        param_data = X[:, param_idx, :]  # shape: (n_batches, n_timepoints)
        
        # Compute pairwise correlations
        for i in range(n_batches):
            for j in range(i, n_batches):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Compute Pearson correlation
                    corr, _ = pearsonr(param_data[i], param_data[j])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr  # Symmetric
        
        return corr_matrix
    
    def _build_global_correlation_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Build global correlation matrix by combining all parameter correlations.
        
        Strategy:
        - Element (i,j) is 0 if ANY parameter has corr(i,j) < threshold
        - Otherwise, element (i,j) = min(corr(i,j) across all parameters)
        
        Parameters
        ----------
        X : ndarray of shape (n_batches, n_params, n_timepoints)
            Input data
        
        Returns
        -------
        global_corr : ndarray of shape (n_batches, n_batches)
            Global correlation matrix
        """
        n_batches, n_params, _ = X.shape
        
        # Compute correlation matrix for each parameter
        self.correlation_matrices_ = []
        for param_idx in range(n_params):
            corr_matrix = self._compute_parameter_correlation_matrix(X, param_idx)
            self.correlation_matrices_.append(corr_matrix)
        
        # Stack all correlation matrices
        corr_stack = np.stack(self.correlation_matrices_, axis=0)  # (n_params, n_batches, n_batches)
        
        # Create mask: True where ALL parameters meet threshold
        threshold_mask = np.all(corr_stack >= self.corr_threshold, axis=0)
        
        # Compute minimum correlation across parameters
        min_corr = np.min(corr_stack, axis=0)
        
        # Apply mask: set to 0 where threshold not met
        global_corr = np.where(threshold_mask, min_corr, 0.0)
        
        # Ensure diagonal is 1
        np.fill_diagonal(global_corr, 1.0)
        
        return global_corr
    
    def _build_graph(self, corr_matrix: np.ndarray) -> nx.Graph:
        """
        Build a graph from the correlation matrix.
        
        Parameters
        ----------
        corr_matrix : ndarray of shape (n_batches, n_batches)
            Correlation matrix
        
        Returns
        -------
        graph : networkx.Graph
            Graph where nodes are batches and edges exist for corr > 0
        """
        n_batches = corr_matrix.shape[0]
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(n_batches))
        
        # Add edges where correlation exceeds threshold
        for i in range(n_batches):
            for j in range(i + 1, n_batches):
                if corr_matrix[i, j] > 0:  # Non-zero correlation (passed threshold)
                    G.add_edge(i, j, weight=corr_matrix[i, j])
        
        return G
    
    def _apply_grouping_algorithm(self, graph: nx.Graph) -> np.ndarray:
        """
        Apply grouping algorithm to the graph.
        
        Parameters
        ----------
        graph : networkx.Graph
            Graph of batch relationships
        
        Returns
        -------
        labels : ndarray of shape (n_nodes,)
            Group labels for each node
        """
        n_nodes = graph.number_of_nodes()
        
        if self.grouping_method == 'connected_components':
            # Find connected components
            components = list(nx.connected_components(graph))
            labels = np.full(n_nodes, -1, dtype=int)
            
            group_id = 0
            for component in components:
                if len(component) >= self.min_group_size:
                    for node in component:
                        labels[node] = group_id
                    group_id += 1
        
        elif self.grouping_method == 'louvain':
            # Louvain community detection (requires python-louvain package)
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(graph)
                labels = np.array([partition[i] for i in range(n_nodes)])
            except ImportError:
                raise ImportError(
                    "Louvain method requires 'python-louvain' package. "
                    "Install with: pip install python-louvain"
                )
        
        elif self.grouping_method == 'label_propagation':
            # Label propagation
            communities = nx.community.label_propagation_communities(graph)
            labels = np.full(n_nodes, -1, dtype=int)
            
            for group_id, community in enumerate(communities):
                if len(community) >= self.min_group_size:
                    for node in community:
                        labels[node] = group_id
        
        return labels
    
    def fit(self, X: np.ndarray, y=None):
        """
        Fit the grouper to the data.
        
        Parameters
        ----------
        X : ndarray of shape (n_batches, n_params, n_timepoints)
            Training data
        y : Ignored
            Not used, present for API consistency
        
        Returns
        -------
        self : BatchGrouper
            Fitted estimator
        """
        # Validate inputs
        self._validate_input(X)
        self._validate_params()
        
        # Store number of parameters for later validation
        self.n_params_ = X.shape[1]
        
        # Build global correlation matrix
        self.global_correlation_matrix_ = self._build_global_correlation_matrix(X)
        
        # Build graph
        self.graph_ = self._build_graph(self.global_correlation_matrix_)
        
        # Apply grouping algorithm
        self.labels_ = self._apply_grouping_algorithm(self.graph_)
        
        # Count groups (excluding outliers with label -1)
        self.n_groups_ = len(np.unique(self.labels_[self.labels_ >= 0]))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new batches to existing groups based on correlation.
        
        Parameters
        ----------
        X : ndarray of shape (n_new_batches, n_params, n_timepoints)
            New batches to assign to groups
        
        Returns
        -------
        labels : ndarray of shape (n_new_batches,)
            Predicted group labels (-1 for outliers)
        """
        # Check if fitted
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Must call fit() before predict()")
        
        self._validate_input(X)
        
        if X.shape[1] != self.n_params_:
            raise ValueError(
                f"X has {X.shape[1]} parameters but estimator was fitted with "
                f"{self.n_params_} parameters"
            )
        
        # For now, implement a simple nearest-neighbor approach
        # This can be extended with more sophisticated methods
        
        # TODO: Implement prediction logic
        # For example: compute correlation with representative batches from each group
        # and assign to the group with highest mean correlation
        
        raise NotImplementedError(
            "predict() method will be implemented in the next version. "
            "Current version only supports fit() for grouping existing batches."
        )
    
    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit the model and return group labels for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_batches, n_params, n_timepoints)
            Training data
        y : Ignored
            Not used, present for API consistency
        
        Returns
        -------
        labels : ndarray of shape (n_batches,)
            Group labels
        """
        return self.fit(X, y).labels_
    
    def get_group_info(self) -> dict:
        """
        Get information about the groups.
        
        Returns
        -------
        info : dict
            Dictionary containing:
            - 'n_groups': Number of groups
            - 'group_sizes': Array of group sizes
            - 'n_outliers': Number of outlier batches
        """
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Must call fit() before get_group_info()")
        
        unique_labels = np.unique(self.labels_)
        group_labels = unique_labels[unique_labels >= 0]
        
        group_sizes = np.array([
            np.sum(self.labels_ == label) for label in group_labels
        ])
        
        n_outliers = np.sum(self.labels_ == -1)
        
        return {
            'n_groups': self.n_groups_,
            'group_sizes': group_sizes,
            'n_outliers': n_outliers,
            'group_labels': group_labels
        }
    
    def get_group_members(self, group_id: int) -> np.ndarray:
        """
        Get batch indices belonging to a specific group.
        
        Parameters
        ----------
        group_id : int
            Group identifier (must be >= 0)
        
        Returns
        -------
        members : ndarray
            Array of batch indices in the group
        """
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Must call fit() before get_group_members()")
        
        if group_id < 0:
            raise ValueError("group_id must be non-negative. Use -1 for outliers.")
        
        return np.where(self.labels_ == group_id)[0]
    
    def visualize_correlation_matrix(self, param_idx: Optional[int] = None):
        """
        Visualize correlation matrix (requires matplotlib).
        
        Parameters
        ----------
        param_idx : int, optional
            If provided, visualize correlation for specific parameter.
            If None, visualize global correlation matrix.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Visualization requires matplotlib")
        
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Must call fit() before visualization")
        
        if param_idx is not None:
            if param_idx < 0 or param_idx >= len(self.correlation_matrices_):
                raise ValueError(f"param_idx must be between 0 and {len(self.correlation_matrices_)-1}")
            corr_matrix = self.correlation_matrices_[param_idx]
            title = f"Correlation Matrix - Parameter {param_idx}"
        else:
            corr_matrix = self.global_correlation_matrix_
            title = "Global Correlation Matrix"
        
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title(title)
        plt.xlabel('Batch Index')
        plt.ylabel('Batch Index')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate data: 20 batches, 3 parameters, 100 timepoints
    # Create some structure: first 8 batches similar, next 7 similar, rest random
    n_batches = 20
    n_params = 3
    n_timepoints = 100
    
    data = np.zeros((n_batches, n_params, n_timepoints))
    
    # Group 1: batches 0-7 (similar)
    base_signal_1 = np.sin(np.linspace(0, 4*np.pi, n_timepoints))
    for i in range(8):
        for p in range(n_params):
            data[i, p, :] = base_signal_1 + np.random.randn(n_timepoints) * 0.2
    
    # Group 2: batches 8-14 (similar)
    base_signal_2 = np.cos(np.linspace(0, 4*np.pi, n_timepoints))
    for i in range(8, 15):
        for p in range(n_params):
            data[i, p, :] = base_signal_2 + np.random.randn(n_timepoints) * 0.2
    
    # Rest: random
    for i in range(15, n_batches):
        for p in range(n_params):
            data[i, p, :] = np.random.randn(n_timepoints)
    
    # Fit the grouper
    grouper = BatchGrouper(
        corr_threshold=0.6,
        grouping_method='connected_components',
        min_group_size=3
    )
    
    labels = grouper.fit_predict(data)
    
    print("Group Labels:", labels)
    print("\nGroup Info:")
    info = grouper.get_group_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nGroup Members:")
    for group_id in info['group_labels']:
        members = grouper.get_group_members(group_id)
        print(f"  Group {group_id}: {members}")
