import numpy as np # type: ignore
from collections import Counter

class Node:
    """
    Node class for Decision Tree
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left            # Left subtree (samples where feature <= threshold)
        self.right = right          # Right subtree (samples where feature > threshold)
        self.value = value          # Predicted class (only for leaf nodes)
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    """
    Decision Tree Classifier implementation from scratch
    """
    def __init__(self, max_depth=100, min_samples_split=2, min_gain=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root = None
    
    def fit(self, X, y):
        """
        Build the decision tree
        
        Parameters:
        -----------
        X : np.ndarray
            Training features of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
        """
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Target values
        depth : int
            Current depth of the tree
            
        Returns:
        --------
        Node
            Root node of the tree/subtree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # Create a leaf node
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # If no improvement, create a leaf node
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Recursively grow the left and right trees
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                    left=left_tree, right=right_tree)
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold for splitting the data
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Target values
            
        Returns:
        --------
        best_feature : int or None
            Index of the best feature to split on
        best_threshold : float or None
            Best threshold value for the split
        """
        best_gain = self.min_gain
        best_feature = None
        best_threshold = None
        
        current_impurity = self._gini_impurity(y)
        
        # Try all features
        for feature_idx in range(self.n_features):
            # Get unique values for the feature
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # For efficiency with many unique values, consider a subset of thresholds
            if len(thresholds) > 100:  # If too many unique values
                percentiles = np.percentile(feature_values, np.linspace(0, 100, 100))
                thresholds = np.unique(percentiles)
            
            # Try all thresholds
            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                # Skip if one side is empty
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Calculate information gain
                left_impurity = self._gini_impurity(y[left_indices])
                right_impurity = self._gini_impurity(y[right_indices])
                
                # Weighted impurity of children
                n_left = np.sum(left_indices)
                n_right = np.sum(right_indices)
                n_total = len(y)
                
                weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
                
                # Calculate information gain
                information_gain = current_impurity - weighted_impurity
                
                # Update if we have a better split
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for a set of labels
        
        Parameters:
        -----------
        y : np.ndarray
            Array of class labels
            
        Returns:
        --------
        float
            Gini impurity
        """
        m = len(y)
        if m == 0:
            return 0
        
        # Count occurrences of each class
        counts = Counter(y)
        impurity = 1
        
        # Calculate Gini impurity
        for label in counts:
            prob_label = counts[label] / m
            impurity -= prob_label ** 2
            
        return impurity
    
    def _most_common_label(self, y):
        """
        Find the most common label in a set
        
        Parameters:
        -----------
        y : np.ndarray
            Array of class labels
            
        Returns:
        --------
        Most common label
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """
        Predict class for samples in X
        
        Parameters:
        -----------
        X : np.ndarray
            Features of shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray
            Predicted classes of shape (n_samples,)
        """
        return np.array([self._predict_sample(sample) for sample in X])
    
    def _predict_sample(self, sample):
        """
        Predict class for a single sample
        
        Parameters:
        -----------
        sample : np.ndarray
            Features of shape (n_features,)
            
        Returns:
        --------
        Predicted class
        """
        node = self.root
        
        while not node.is_leaf():
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
                
        return node.value






