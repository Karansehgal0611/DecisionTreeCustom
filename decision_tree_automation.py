import numpy as np # type: ignore
import pandas as pd# type: ignore
import matplotlib.pyplot as plt# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report# type: ignore
import seaborn as sns# type: ignore

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        # Decision node attributes
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the feature
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        
        # Leaf node attribute
        self.value = value              # Class prediction (for leaf nodes)

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.feature_names = None
        self.n_classes = None
        self.tree_structure = []  # For visualization
    
    def fit(self, X, y, feature_names=None):
        """
        Build the decision tree
        """
        self.n_features = X.shape[1]
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(self.n_features)]
        self.n_classes = len(np.unique(y))
        
        # Start recursive building of the tree
        self.root = self._grow_tree(X, y)
        
        # Extract tree structure for visualization
        self.tree_structure = []
        self._extract_tree_structure(self.root, [], 0)
        
        return self
    
    def _grow_tree(self, X, y, depth=0):
        """
        Recursive function to build the tree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find the best split
        feature_idx, threshold = self._best_split(X, y)
        
        # If no split improves the criterion, create a leaf node
        if feature_idx is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices
        
        # Recursive partitioning
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
    
    def _best_split(self, X, y):
        """
        Find the best split
        """
        m = X.shape[0]
        if m <= 1:
            return None, None
        
        # Count of each class in the current node
        parent_impurity = self._calculate_impurity(y)
        
        # Initialize variables
        best_feature_idx, best_threshold = None, None
        best_info_gain = -np.inf
        
        # Loop through all features
        for feature_idx in range(self.n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Continue if there's only one unique value
            if len(thresholds) == 1:
                continue
            
            # Loop through all thresholds
            for threshold in thresholds:
                # Split the data
                left_indices = feature_values <= threshold
                right_indices = ~left_indices
                
                # Skip if one node is empty
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Calculate the information gain
                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                
                # Weight by the number of samples in each split
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                weighted_impurity = (n_left / m) * left_impurity + (n_right / m) * right_impurity
                
                # Calculate information gain
                info_gain = parent_impurity - weighted_impurity
                
                # Update if we found a better split
                if info_gain > best_info_gain:
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_info_gain = info_gain
        
        return best_feature_idx, best_threshold
    
    def _calculate_impurity(self, y):
        """
        Calculate impurity (Gini or entropy)
        """
        m = len(y)
        if m == 0:
            return 0
        
        # Calculate proportion of each class
        proportion = np.bincount(y.astype(int)) / m
        
        # Remove zero entries (classes that aren't present)
        proportion = proportion[proportion > 0]
        
        if self.criterion == 'gini':
            # Gini impurity
            return 1 - np.sum(np.square(proportion))
        else:
            # Entropy
            return -np.sum(proportion * np.log2(proportion))
    
    def _most_common_label(self, y):
        """
        Return the most common class in a node
        """
        return np.bincount(y.astype(int)).argmax()
    
    def predict(self, X):
        """
        Predict class for X
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse tree until reaching a leaf node
        """
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def _extract_tree_structure(self, node, path, depth):
        """
        Extract the tree structure for visualization
        """
        if node.value is not None:
            # This is a leaf node
            self.tree_structure.append({
                'depth': depth,
                'path': path.copy(),
                'value': node.value,
                'samples': None  # We don't track this in our implementation, but could be added
            })
        else:
            # This is a decision node
            feature_name = self.feature_names[node.feature_idx]
            left_path = path.copy()
            left_path.append(f"{feature_name} <= {node.threshold}")
            self._extract_tree_structure(node.left, left_path, depth + 1)
            
            right_path = path.copy()
            right_path.append(f"{feature_name} > {node.threshold}")
            self._extract_tree_structure(node.right, right_path, depth + 1)


def load_dataset(filepath, target_column=None):
    """
    Load a dataset from a CSV file and split into X, y
    """
    # Load the data
    data = pd.read_csv(filepath)
    
    # If target column is specified, use it; otherwise assume last column
    if target_column is None:
        target_column = data.columns[-1]
    
    # Split into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Convert to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    return X, y, feature_names

def visualize_decision_tree(tree, class_names=None):
    """
    Visualize the decision tree structure.
    Saves the tree as 'decision_tree_visualization.png'.
    """
    plt.figure(figsize=(20, 10))

    def plot_node(node_info, x, y, node_type='decision'):
        if node_type == 'decision':
            node_text = '\n'.join(node_info.get('path', []))
            bbox_props = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=2)
        else:  # leaf node
            node_text = f"Class: {node_info.get('value')}"
            if class_names is not None and isinstance(node_info.get('value'), int) and node_info['value'] < len(class_names):
                node_text = f"Class: {class_names[node_info['value']]}"
            bbox_props = dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="black", lw=2)

        plt.text(x, y, node_text, ha="center", va="center", size=10, bbox=bbox_props)

    # Group nodes by depth
    nodes_by_depth = {}
    for node in tree.tree_structure:
        depth = node.get('depth', 0)
        nodes_by_depth.setdefault(depth, []).append(node)

    sorted_depths = sorted(nodes_by_depth.keys())
    max_depth = max(sorted_depths)

    for depth in sorted_depths:
        nodes = nodes_by_depth[depth]
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = (i + 0.5) / n_nodes
            y = 1 - (depth / (max_depth + 1))

            if 'value' in node and node['value'] is not None:
                plot_node(node, x, y, node_type='leaf')
            else:
                plot_node(node, x, y)

    plt.axis('off')
    plt.title('Decision Tree Visualization')
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png')
    plt.show()


def visualize_feature_importance(tree, feature_names):
    """
    Visualize feature importance
    """
    # Count how many times each feature appears in the decision nodes
    feature_counts = {}
    
    def count_features(node):
        if node.value is None:  # Decision node
            feature_idx = node.feature_idx
            feature_name = feature_names[feature_idx]
            feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
            count_features(node.left)
            count_features(node.right)
    
    count_features(tree.root)
    
    # Convert to DataFrame and sort
    feature_importance = pd.DataFrame({
        'Feature': list(feature_counts.keys()),
        'Importance': list(feature_counts.values())
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def run_decision_tree_analysis(filepath, target_column=None, test_size=0.3, max_depth=5, criterion='gini'):
    """
    Complete pipeline to load data, train a decision tree, evaluate, and visualize
    """
    print(f"Loading dataset from {filepath}...")
    X, y, feature_names = load_dataset(filepath, target_column)
    
    # Get class names (assuming y contains integers)
    n_classes = len(np.unique(y))
    class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Split data
    print(f"Splitting data into train and test sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train the decision tree
    print(f"Training decision tree (max_depth={max_depth}, criterion={criterion})...")
    tree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    tree.fit(X_train, y_train, feature_names)
    
    # Make predictions
    print("Making predictions...")
    y_pred = tree.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_decision_tree(tree, class_names)
    visualize_feature_importance(tree, feature_names)
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    return tree, accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Decision Tree Classifier')
    parser.add_argument('--filepath', type=str, required=True, help='Path to the CSV dataset')
    parser.add_argument('--target', type=str, default=None, help='Name of the target column')
    parser.add_argument('--test_size', type=float, default=0.3, help='Proportion of data to use for testing')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum depth of the decision tree')
    parser.add_argument('--criterion', type=str, default='gini', choices=['gini', 'entropy'], 
                        help='Function to measure quality of a split')
    
    args = parser.parse_args()
    
    run_decision_tree_analysis(
        filepath=args.filepath,
        target_column=args.target,
        test_size=args.test_size,
        max_depth=args.max_depth,
        criterion=args.criterion
    )