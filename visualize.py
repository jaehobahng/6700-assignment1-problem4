import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from umap import UMAP


def main():
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
    # plt.show()

    # digits = load_digits()
    # umap_2d = UMAP()
    # umap_2d.fit(digits.data)

    # projections = umap_2d.transform(digits.data)

    # fig = px.scatter(
    #     projections,
    #     x=0,
    #     y=1,
    #     color=digits.target.astype(str),
    #     labels={"color": "digit"}
    # )

    # fig.write_html("./public/index.html")

if __name__ == "__main__":
    main()