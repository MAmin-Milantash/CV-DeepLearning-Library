def visualize_inception_outputs(outputs):
    """outputs: list of feature maps from each branch"""
    for i, feature_map in enumerate(outputs):
        print(f"Branch {i} output shape: {feature_map.shape}")
        # Optional: call visualize_feature_map(feature_map)