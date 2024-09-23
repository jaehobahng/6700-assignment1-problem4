import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from umap.umap_ import UMAP


def main():
    """
    Main function that loads the digits dataset from sklearn, applies UMAP to the data, and creates a scatter plot using Matplotlib.

    1. Load the digits dataset from sklearn.
    2. Apply UMAP function to the data to reduce the dimensionality.
    3. Transform the data using the UMAP model.
    4. Create a scatter plot with Matplotlib.
    5. Add colorbar and labels to the plot.
    6. Save the plot as a PNG file. 
    """

    # Load the dataset and apply UMAP
    digits = load_digits()
    umap_2d = UMAP()
    umap_2d.fit(digits.data)

    # Transform the data
    projections = umap_2d.transform(digits.data)

    # Create a scatter plot with Matplotlib
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projections[:, 0],
        projections[:, 1],
        c=digits.target,
        cmap='Spectral',
        s=5
    )

    # Add colorbar and labels
    plt.colorbar(scatter, label='Digit')
    plt.title('UMAP projection of the Digits dataset')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Save the plot as a PNG file
    plt.savefig('./public/umap.png', dpi=300)

if __name__ == "__main__":
    main()