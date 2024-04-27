import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2
import argparse


def check_image_path(image_path):
    """
    Check if the provided image path is valid and the file is an image.

    Parameters:
    image_path (str): The path to the image file

    Returns:
    bool: True if the image path is valid, False otherwise
    """
    # Check if the image path exists
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return False

    # Attempt to read the image using OpenCV to check if it's valid
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: The file {image_path} is not a valid image.")
        return False

    return True


def gaussian_kernel(size, sigma=1):
    """
    Create the Gaussian Filter with given size and sigma.

    Parameters:
    size (int): The size of the Gaussian filter
    sigma (float): The standard deviation of the Gaussian filter

    Returns:
    numpy.ndarray: The Gaussian filter
    """
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def convolution_padding(image, kernel):
    """
    Manually convolve an image with a kernel.

    Parameters:
    image (numpy.ndarray): The input image
    kernel (numpy.ndarray): The kernel to convolve the image with

    Returns:
    numpy.ndarray: The convolved image
    """
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    output = np.zeros_like(image, dtype=np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Extract the region of interest
            region = padded_image[x:x + kernel_height, y:y + kernel_width]
            # Perform element-wise multiplication and sum the result
            output[x, y] = np.sum(region * kernel)
    return output


def sobel_filter(image):
    """
    Apply the Sobel filter to an image to extract horizontal and vertical edges.

    Parameters:
    image (numpy.ndarray): The input image

    Returns:
    tuple: The images of the horizontal and vertical edges
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    Ix = convolution_padding(image, Kx)
    Iy = convolution_padding(image, Ky)

    return Ix, Iy


def gradient_intensity(Ix, Iy):
    """
    Calculate the gradient intensity of the image.

    Parameters:
    Ix (numpy.ndarray): The image of the horizontal edges
    Iy (numpy.ndarray): The image of the vertical edges

    Returns:
    tuple: The gradient magnitude and the gradient direction
    """
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)


def non_maximum_suppression(image, D):
    """
    Apply non-maximum suppression to an image.

    Parameters:
    image (numpy.ndarray): The input image
    D (numpy.ndarray): The gradient direction of the image

    Returns:
    numpy.ndarray: The image after non-maximum suppression
    """
    M, N = image.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                # angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                # angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]

                if (image[i,j] >= q) and (image[i,j] >= r):
                    Z[i,j] = image[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    """
    Apply double threshold to an image.

    Parameters:
    img (numpy.ndarray): The input image
    lowThresholdRatio (float): The low threshold ratio
    highThresholdRatio (float): The high threshold ratio

    Returns:
    tuple: The image after double threshold, the weak pixel value, and the strong pixel value
    """
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    """
    Apply hysteresis thresholding to an image.

    Parameters:
    img (numpy.ndarray): The input image
    weak (int): The weak pixel value
    strong (int): The strong pixel value

    Returns:
    numpy.ndarray: The image after hysteresis thresholding
    """
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong)
                            or (img[i+1, j] == strong)
                            or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong)
                            or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong)
                            or (img[i-1, j] == strong)
                            or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def hysteresis_(img, weak, strong=255):
    """
    Apply hysteresis thresholding to an image.

    Parameters:
    img (numpy.ndarray): The input image
    weak (int): The weak pixel value
    strong (int): The strong pixel value

    Returns:
    numpy.ndarray: The image after hysteresis thresholding
    """
    M, N = img.shape
    strong_pixels = [(i,j) for i in range(M) for j in range(N) if img[i,j] == strong]

    while strong_pixels:
        i, j = strong_pixels.pop(0)
        for x in range(max(0, i-1), min(M, i+2)):
            for y in range(max(0, j-1), min(N, j+2)):
                if img[x, y] == weak:
                    img[x, y] = strong
                    strong_pixels.append((x, y))

    # Remove all remaining weak pixels
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                img[i, j] = 0
    return img


# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('--lowThresholdRatio', type=float, default=0.05,
                    help='Low threshold ratio for edge detection')
parser.add_argument('--highThresholdRatio', type=float, default=0.09,
                    help='High threshold ratio for edge detection')
parser.add_argument('image_path', type=str,
                    help='The path to the image file')

# Parse the arguments
args = parser.parse_args()

# Extract the image path from the command line arguments
image_path = args.image_path

# Check the validity of image path
if not check_image_path(image_path):
    sys.exit(1)

original_image = cv2.imread(image_path)
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur
gk = gaussian_kernel(5, 1)
blurred = convolution_padding(gray_image, gk)

# Use Sobel filter to extract horizontal and vertical edges
Ix, Iy = sobel_filter(blurred)

# Gradient intensity
G, theta = gradient_intensity(Iy, Ix)

# Non-maximum suppression
nms = non_maximum_suppression(G, theta)
cv2.imwrite("nonmax.jpg", nms)
# Double threshold
dt, weak, strong = threshold(nms, args.lowThresholdRatio, args.highThresholdRatio)

cv2.imwrite("DT.jpg", dt)

# Hysteresis
edges = hysteresis_(dt.copy(), weak, strong)
cv2.imwrite("edges.jpg", edges)

# Plotting both images in one graph
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
# Adapt the window size to fully display the plot
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2, wspace=0.2)

axs[0][0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axs[0][0].set_title('Original Image')
axs[0][0].axis('off')  # Hide axes for better visualization

axs[0][1].imshow(gray_image, cmap='gray')
axs[0][1].set_title('Grayscale Image')
axs[0][1].axis('off')  # Hide axes for better visualization

axs[0][2].imshow(blurred, cmap='gray')
axs[0][2].set_title('Blurred Image')
axs[0][2].axis('off')  # Hide axes for better visualization

axs[1][0].imshow(Ix, cmap='gray')
axs[1][0].set_title('Vertical Edges')
axs[1][0].axis('off')  # Hide axes for better visualization

axs[1][1].imshow(Iy, cmap='gray')
axs[1][1].set_title('Horizontal Edges')
axs[1][1].axis('off')  # Hide axes for better visualization

axs[1][2].imshow(G, cmap='gray')
axs[1][2].set_title('Gradient Magnitude')
axs[1][2].axis('off')  # Hide axes for better visualization

axs[2][0].imshow(nms, cmap='gray')
axs[2][0].set_title('Non Maximum Suppression')
axs[2][0].axis('off')  # Hide axes for better visualization

axs[2][1].imshow(dt, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
axs[2][1].set_title('Double Threshold')
axs[2][1].axis('off')  # Hide axes for better visualization

axs[2][2].imshow(edges, cmap='gray')
axs[2][2].set_title('Hysteresis')
axs[2][2].axis('off')  # Hide axes for better visualization

plt.show()
