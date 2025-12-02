import numpy as np
import matplotlib.pyplot as plt

def generate_random_unit_vectors(num_vectors, dimensions):
    """Generate random unit vectors uniformly distributed on the unit sphere."""
    # Generate random points from a normal distribution
    vectors = np.random.normal(size=(num_vectors, dimensions))
    # Normalize the vectors to ensure they lie on the unit sphere
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

def plot_distribution(vectors, dimensions):
    """Plot the distribution of the first component of the vectors."""
    x0_values = vectors[:, 0]
    
    plt.figure(figsize=(12, 6))
    plt.hist(x0_values, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Distribution of $x[0]$ in {dimensions}-dimensional Space')
    plt.xlabel('$x[0]$')
    plt.ylabel('Density')
    plt.xlim(-1, 1)
    plt.grid()
    plt.show()

# Parameters
num_vectors = 10000

# Generate and plot for 3 dimensions
vectors_3d = generate_random_unit_vectors(num_vectors, 3)
plot_distribution(vectors_3d, 3)

# Generate and plot for 1000 dimensions
vectors_1000d = generate_random_unit_vectors(num_vectors, 1000)
plot_distribution(vectors_1000d, 1000)