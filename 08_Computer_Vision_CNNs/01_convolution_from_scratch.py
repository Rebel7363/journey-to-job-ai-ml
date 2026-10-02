import numpy as np


def conv2d_scratch(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Performs 2D convolution from scratch using pure NumPy.
    Input image shape: (H, W)
    Kernel shape: (kH, kW)
    """
    if padding > 0:
        image_padded = np.pad(
            image,
            ((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0
        )
    else:
        image_padded = image

    h_in, w_in = image_padded.shape
    k_h, k_w = kernel.shape

    # Compute output spatial dimensions
    out_h = (h_in - k_h) // stride + 1
    out_w = (w_in - k_w) // stride + 1

    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            r_end = r_start + k_h
            c_start = j * stride
            c_end = c_start + k_w

            region = image_padded[r_start:r_end, c_start:c_end]
            output[i, j] = np.sum(region * kernel)

    return output


def main():
    # 6x6 Synthetic Grayscale Image with a sharp vertical edge
    image = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ], dtype=np.float32)

    # Vertical Sobel Filter
    sobel_v = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Horizontal Sobel Filter
    sobel_h = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    print("--- Original 6x6 Image Matrix ---")
    print(image)

    # Convolve with Vertical Edge Detector
    vertical_edges = conv2d_scratch(image, sobel_v, stride=1, padding=1)
    print("\n--- Output: Vertical Edge Map (Padding=1, Stride=1) ---")
    print(vertical_edges)

    # Convolve with Horizontal Edge Detector
    horizontal_edges = conv2d_scratch(image, sobel_h, stride=1, padding=1)
    print("\n--- Output: Horizontal Edge Map (Padding=1, Stride=1) ---")
    print(horizontal_edges)


if __name__ == "__main__":
    main()