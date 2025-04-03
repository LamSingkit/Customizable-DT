import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, feature=None, threshold=None, count = None, avg = None, left=None, right=None, *, leaf_value=None):
        self.feature = feature
        self.threshold = threshold
        self.count = count
        self.avg = avg
        self.left = left
        self.right = right
        self.leaf_value = leaf_value

def mean_encode(data, y):
    # Mean Encode All Categorical Features in the DataFrame
    # Return dictionary of encoded columns
    encoded_columns = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            mean_target = y.groupby(data[column]).mean().to_dict()
            encoded_col_name = f'{column}_encode'
            data[encoded_col_name] = data[column].map(mean_target)
            encoded_columns[encoded_col_name] = column
    return encoded_columns

def m_estimate_smoothing_encode(data, y, m = 10):
    # Return dictionary of encoded columns
    encoded_columns = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            global_mean = y.mean()
            stats = y.groupby(data[column]).agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
            encoded_col_name = f'{column}_encode'
            data[encoded_col_name] = data[column].map(smoothed.to_dict())
            encoded_columns[encoded_col_name] = column
    return encoded_columns

def gini_impurity(y):
    _, counts = np.unique(y, return_counts=True) # outputs unique values and their counts in list
    probabilities = counts / counts.sum() # Calculate probabilities of each unique value 
    gini = 1 - np.sum(probabilities**2)
    return gini

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def variance(y):
    return np.var(y)

def best_split(data, y, split_criterion='Gini'):
    min_criterion = float('inf')
    best_feature, best_threshold = None, None

    for feature in data.columns:
        # Skip categorical features
        if data[feature].dtype == 'object':
            continue
        
        thresholds = data[feature].unique()
        for threshold in thresholds:
            left_mask = data[feature] <= threshold
            right_mask = data[feature] > threshold
            if left_mask.any() and right_mask.any():
                y_left, y_right = y[left_mask], y[right_mask]
                n_left, n_right = len(y_left), len(y_right)
                
                if split_criterion == 'Gini':
                    l_criterion = gini_impurity(y_left)
                    r_criterion = gini_impurity(y_right)
                    weighted_criterion = (n_left / len(y)) * l_criterion + (n_right / len(y)) * r_criterion

                elif split_criterion == 'Entropy':
                    l_criterion = entropy(y_left)
                    r_criterion = entropy(y_right)
                    weighted_criterion = (n_left / len(y)) * l_criterion + (n_right / len(y)) * r_criterion

                elif split_criterion == 'Variance':
                    parent_variance = variance(y)
                    weighted_child_variance = (n_left / len(y)) * variance(y_left) + (n_right / len(y)) * variance(y_right)
                    variance_reduction = parent_variance - weighted_child_variance
                    # Negate the variance reduction so that maximizing reduction becomes minimizing the criterion.
                    weighted_criterion = -variance_reduction

                else:
                    raise ValueError("Invalid split criterion. Choose 'Gini', 'Entropy', or 'Variance'.")

                if weighted_criterion < min_criterion:
                    min_criterion = weighted_criterion
                    best_feature = feature
                    best_threshold = threshold

    return best_feature, best_threshold, min_criterion

def build_tree(data, y, split_criterion, depth=0, max_depth=None, node_feature = None, node_threshold = None):
    if len(set(y)) == 1 or len(y) == 0:
        # Return a leaf node if all target values are the same or no data is left
        return TreeNode(feature=node_feature, threshold=node_threshold, leaf_value=y.mode()[0] if len(y) != 0 else None)

    if max_depth is not None and depth >= max_depth:
        # Return a leaf node if the maximum depth is reached
        return TreeNode(feature=node_feature, threshold=node_threshold, leaf_value=y.mode()[0])

    # Find the best split
    feature, threshold, _ = best_split(data, y, split_criterion)

    if feature is None:
        # Return a leaf node if no valid split is found
        return TreeNode(feature=node_feature, threshold=node_threshold, leaf_value=y.mode()[0])

    # Calculate count and avg
    count = len(y)
    avg = y.mean()

    # Split data
    left_mask = data[feature] <= threshold
    right_mask = data[feature] > threshold
    left_data, right_data = data[left_mask], data[right_mask]
    left_y, right_y = y[left_mask], y[right_mask]

    # If the feature is categorical, calculate the categorical threshold
    # Else, build the tree with the original threshold
    if encoding_map and feature in encoding_map:
        original_feature = encoding_map[feature]
        left_categories = left_data[original_feature].unique()
        right_categories = right_data[original_feature].unique()
        left_child = build_tree(left_data, left_y, split_criterion, depth+1, max_depth, node_feature = original_feature, node_threshold = left_categories)
        right_child = build_tree(right_data, right_y, split_criterion, depth+1, max_depth, node_feature = original_feature, node_threshold = right_categories)
    else:
        left_child = build_tree(left_data, left_y, split_criterion, depth+1, max_depth, node_feature = feature, node_threshold = threshold)
        right_child = build_tree(right_data, right_y, split_criterion, depth+1, max_depth, node_feature = feature, node_threshold = threshold)

    # Return current node
    return TreeNode(feature=node_feature, threshold=node_threshold, count = count, avg = avg, left=left_child, right=right_child)

def display_tree(node, prefix="", is_left=None):
    if node.leaf_value is not None:
        branch = ""
        if is_left is not None:
            if is_left:
                branch = "├── "
                if isinstance(node.threshold, (np.ndarray)):
                    print(f"{prefix}{branch}{node.feature} in {node.threshold} Predict {node.leaf_value}")
                else:
                    print(f"{prefix}{branch}{node.feature} <= {node.threshold:.2f} Predict {node.leaf_value}")
            else:
                branch = "└── "
                if isinstance(node.threshold, (np.ndarray)):
                    print(f"{prefix}{branch}{node.feature} in {node.threshold} Predict {node.leaf_value}")
                else:
                    print(f"{prefix}{branch}{node.feature} > {node.threshold:.2f} Predict {node.leaf_value}")
        return

    if is_left is None:
        # Root node
        print(f"Count:{node.count}, Avg: {node.avg*100:.1f}%")
    else:
        branch = "├── " if is_left else "└── "
        if is_left:
            if isinstance(node.threshold, (np.ndarray)):
                print(f"{prefix}{branch}{node.feature} in {node.threshold}, Count:{node.count}, Avg: {node.avg*100:.1f}%")
            else:
                print(f"{prefix}{branch}{node.feature} <= {node.threshold:.2f}, Count:{node.count}, Avg: {node.avg*100:.1f}%")
        else:
            if isinstance(node.threshold, (np.ndarray)):
                print(f"{prefix}{branch}{node.feature} in {node.threshold}, Count:{node.count}, Avg: {node.avg*100:.1f}%")
            else:
                print(f"{prefix}{branch}{node.feature} > {node.threshold:.2f}, Count:{node.count}, Avg: {node.avg*100:.1f}%")
    # Calculate new prefix for children
    if is_left is not None:
        new_prefix = prefix + ("│   " if is_left else "    ")
    else:
        new_prefix = prefix + "    "

    if node.left:
        display_tree(node.left, new_prefix, True)
    if node.right:
        display_tree(node.right, new_prefix, False)

## Compile Toy Data
data2 = pd.DataFrame({
    'X1': np.random.randint(1, 11, 100),        # Integer values 1-10
    'X2': np.random.normal(50, 15, 100),        # Normal distribution (μ=50, σ=15)
    'X3': np.random.exponential(2, 100),        # Exponential distribution
    'X4': np.random.uniform(0, 100, 100),       # Uniform distribution 0-100
    'X5': np.abs(np.random.randn(100) * 10),    # Right-skewed positive values
    'X6': np.random.choice(['A', 'B', 'C', 'D'], 100)  # Categorical feature
})

# Create target using only numerical features
y2 = pd.Series(np.where(
    ((data2['X1'] > 5) & (data2['X2'] < 60)) |    # Rule 1
    ((data2['X3'] > 2.5) & (data2['X4'] < 40)) |  # Rule 2
    (data2['X5'] > 8),                            # New rule replacing X6 dependency
    1, 0
))

# Add 15% noise to target
noise_mask = np.random.choice([True, False], 100, p=[0.15, 0.85])
y2[noise_mask] = 1 - y2[noise_mask]

# Reset index
data2 = data2.reset_index(drop=True)
y2 = y2.reset_index(drop=True)


encoding_map = mean_encode(data2, y2)
tree = build_tree(data2, y2, split_criterion = 'Gini', max_depth=None)
display_tree(tree)
