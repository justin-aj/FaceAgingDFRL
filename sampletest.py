import numpy as np
import matplotlib.pyplot as plt

# Parameters
image_size = 64
num_steps = 100
diffusion_rate = 0.2

# Initialize image (random noise)
image = np.random.rand(image_size, image_size)

# Diffusion function
def diffuse_image(image, rate):
    new_image = image.copy()
    for _ in range(num_steps):
        new_image += rate * (
            np.roll(new_image, 1, axis=0) + np.roll(new_image, -1, axis=0) +
            np.roll(new_image, 1, axis=1) + np.roll(new_image, -1, axis=1) -
            4 * new_image
        )
        new_image = np.clip(new_image, 0, 1)  # Keep values in range
    return new_image

# Generate image
generated_image = diffuse_image(image, diffusion_rate)

# Plot result
plt.imshow(generated_image, cmap='inferno')
plt.colorbar(label='Intensity')
plt.title('Simple Diffusion-Based Image Generation')
plt.show()
