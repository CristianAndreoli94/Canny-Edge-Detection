Here's the markdown you requested:

# Canny Edge Detection

This program implements the Canny Edge Detection algorithm to detect edges in an image. It follows the main steps of the algorithm, including Gaussian smoothing, Sobel filtering, non-maximum suppression, double thresholding, and edge tracking by hysteresis.

## Dependencies

To run this program successfully, you must have the following packages installed:

1. Python (3.x recommended)
2. OpenCV (cv2)
   ```bash
   pip install opencv-python
   ```
3. NumPy
   ```bash
   pip install numpy
   ```
4. Matplotlib
   ```bash
   pip install matplotlib
   ```

## Key Features

The program consists of several functions that perform the different steps of the Canny Edge Detection algorithm:

- `gaussian_kernel`: Creates a Gaussian filter with a given size and standard deviation.
- `convolution_padding`: Performs manual convolution of an image with a kernel, using padding to handle border pixels.
- `sobel_filter`: Applies the Sobel filter to an image to extract horizontal and vertical edges.
- `gradient_intensity`: Calculates the gradient magnitude and direction of the image.
- `non_maximum_suppression`: Applies non-maximum suppression to the image to thin the edges.
- `threshold`: Applies double thresholding to the image using low and high threshold ratios.
- `hysteresis`: Performs edge tracking by hysteresis to connect weak edges to strong edges.

## Usage

The program can be run from the command line with the following arguments:

- `--lowThresholdRatio`: Low threshold ratio for edge detection (default: 0.05)
- `--highThresholdRatio`: High threshold ratio for edge detection (default: 0.09)
- `image_path`: The path to the input image file

Example usage:

```bash
python CannyEdgeDetection.py --lowThresholdRatio 0.05 --highThresholdRatio 0.09 path/to/image.jpg
```

## Output

The program generates several output images at different stages of the algorithm:

- `nonmax.jpg`: The image after applying non-maximum suppression
- `DT.jpg`: The image after applying double thresholding
- `edges.jpg`: The final image with the detected edges

Additionally, the program displays a plot with the original image, grayscale image, blurred image, vertical edges, horizontal edges, gradient magnitude, non-maximum suppression, double threshold, and final edges.

## Examples

Here are some example results of running the Canny Edge Detection program:

```bash
    python3 bombo.JPG
```

![Alt text](output_image/bombo_output.png)
![Alt text](output_image/bombo_nonmax.png)
![Alt text](output_image/bombo_DT.png)
![Alt text](output_image/bombo_edges.png)

```bash
    python3 donna.PNG
```

![Alt text](output_image/donna_output.png)
![Alt text](output_image/donna_nonmax.png)
![Alt text](output_image/donna_DT.png)
![Alt text](output_image/donna_edges.png)

For more details on the implementation and theory behind the Canny Edge Detection algorithm, please refer to the source code and the accompanying documentation.