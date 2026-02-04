def plot_residual_flow(input_features, output_features):
    """Compare feature statistics before and after residual block"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure()
    plt.plot(np.mean(input_features.cpu().detach().numpy(), axis=(0,2,3)), label="Input Mean")
    plt.plot(np.mean(output_features.cpu().detach().numpy(), axis=(0,2,3)), label="Output Mean")
    plt.legend()
    plt.title("Residual Flow")
    plt.show()