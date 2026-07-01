import cv2
import numpy as np
from .geometric import crop
from .io import convert_color
#blur()
def blur(
        image,
        method="average",
        kernel_size=5,
        sigma=0,
        border_type="reflect"
):
    """
    Apply a blur filter to an image.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or color image.

    method : str, default="average"
        Blur method.

        Available options:

        - "average"
        - "gaussian"
        - "median"

    kernel_size : int, default=5
        Size of the filtering kernel.

        Must be:
        - Positive
        - Odd integer

        Used by:
        - Average blur
        - Gaussian blur
        - Median blur

    sigma : float, default=0
        Standard deviation for Gaussian blur.

        Used only when:
        method="gaussian"

    border_type : str, default="reflect"
        Border handling strategy.

        Available options:

        - "constant"
        - "replicate"
        - "reflect"
        - "reflect_101"

        Note:
        Applied only to average and gaussian blur.

    Returns
    -------
    np.ndarray
        Blurred image.

    Raises
    ------
    TypeError
        If parameter types are invalid.

    ValueError
        If parameter values are invalid.
    """

    # -------------------------
    # Validate image
    # -------------------------

    if image is None:
        raise ValueError("Image cannot be None.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(image).__name__}."
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            "Expected a 2D (grayscale) or 3D (color) image."
        )

    # -------------------------
    # Validate method
    # -------------------------

    if not isinstance(method, str):
        raise TypeError(
            f"Expected method to be a string, got {type(method).__name__}."
        )

    method = method.lower()

    valid_methods = {
        "average",
        "gaussian",
        "median"
    }

    if method not in valid_methods:
        raise ValueError(
            f"Unsupported method. Available options: {list(valid_methods)}."
        )

    # -------------------------
    # Validate kernel size
    # -------------------------

    if type(kernel_size) is not int:
        raise TypeError(
            "kernel_size must be an integer."
        )

    if kernel_size <= 0:
        raise ValueError(
            "kernel_size must be greater than zero."
        )

    if kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be an odd integer."
        )

    # -------------------------
    # Validate sigma
    # -------------------------

    if not isinstance(sigma, (int, float)):
        raise TypeError(
            "sigma must be numeric."
        )

    if sigma < 0:
        raise ValueError(
            "sigma must be greater than or equal to zero."
        )

    # -------------------------
    # Validate border type
    # -------------------------

    border_map = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
        "reflect_101": cv2.BORDER_REFLECT_101
    }

    if not isinstance(border_type, str):
        raise TypeError(
            f"Expected border_type to be a string, got {type(border_type).__name__}."
        )

    border_type = border_type.lower()

    if border_type not in border_map:
        raise ValueError(
            f"Unsupported border_type. "
            f"Available options: {list(border_map.keys())}."
        )

    # -------------------------
    # Average Blur
    # -------------------------

    if method == "average":

        blurred_image = cv2.blur(
            image,
            (kernel_size, kernel_size),
            borderType=border_map[border_type]
        )

    # -------------------------
    # Gaussian Blur
    # -------------------------

    elif method == "gaussian":

        blurred_image = cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            sigmaX=sigma,
            borderType=border_map[border_type]
        )

    # -------------------------
    # Median Blur
    # -------------------------

    else:   # method == "median"

        blurred_image = cv2.medianBlur(
            image,
            kernel_size
        )

    return blurred_image
#roi_blur()
def roi_blur(
        image,
        bbox,
        kernel_size
):
    """
    Apply Gaussian blur to a Region of Interest (ROI).

    The ROI is specified using a bounding box of the form
    `(x, y, width, height)`.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or color image.

    bbox : tuple[int, int, int, int]
        Bounding box defining the ROI as
        `(x, y, width, height)`.

    kernel_size : int
        Size of the Gaussian kernel.

        Must be:
        - Positive
        - Odd

    Returns
    -------
    np.ndarray
        Image with the selected ROI blurred.

    Raises
    ------
    TypeError
        If input types are invalid.

    ValueError
        If the bounding box or kernel size is invalid.
    """

    # -------------------------
    # Validate image
    # -------------------------

    if image is None:
        raise ValueError("Image cannot be None.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(image).__name__}"
        )

    # -------------------------
    # Validate bbox
    # -------------------------

    if not isinstance(bbox, (tuple, list)):
        raise TypeError(
            f"Expected tuple or list for bbox, got {type(bbox).__name__}"
        )

    if len(bbox) != 4:
        raise ValueError(
            "bbox must contain exactly four values: (x, y, width, height)."
        )

    x, y, width, height = bbox

    for name, value in zip(
        ("x", "y", "width", "height"),
        (x, y, width, height)
    ):
        if type(value) is not int:
            raise TypeError(
                f"{name} must be an int, got {type(value).__name__}"
            )

    # -------------------------
    # Validate kernel size
    # -------------------------

    if type(kernel_size) is not int:
        raise TypeError(
            f"kernel_size must be an int, got {type(kernel_size).__name__}"
        )

    if kernel_size <= 0:
        raise ValueError(
            "kernel_size must be greater than zero."
        )

    if kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be an odd integer."
        )

    # -------------------------
    # Extract ROI
    # -------------------------

    roi = crop(
        image,
        x=x,
        y=y,
        width=width,
        height=height
    )

    # -------------------------
    # Blur ROI
    # -------------------------

    blurred_roi = blur(
        roi,
        method="gaussian",
        kernel_size=kernel_size
    )

    # -------------------------
    # Copy original image
    # -------------------------

    output = image.copy()

    # -------------------------
    # Paste blurred ROI
    # -------------------------

    output[
        y:y + height,
        x:x + width
    ] = blurred_roi

    return output

#detect_edges()
def detect_edge(
        image,
        method="sobel",
        kernel_size=3,
        threshold1=100,
        threshold2=200
):
    """
    Detects edges in an image usng dfferent adge detection methods.

    Parameters
    ----------
    image : np.ndarray
        Input image.

        Supported formats:
        - Grayscale : (H, W)
        - Color : (H, W, 3)
    method : str, default="sobel"
        Edge detection method.

        Available options:
        - "sobel"
        - "canny"
        - "laplacian"
    kernal_size : int, default = 3
        Kernal size used by Sobel and Laplacian.
        Must be:
        - Positive
        - Odd integer
    threshold1 : int or float, default=100
        Lower threshold for Canny edge detection.
    threshold2 : int or float, default=200
        Upper threshold for Canny edge detection.

    Returns
    -------
    np.ndarray
        Image with detected edges.

    Raises 
    ------
    TypeError 
       If inout types are invalid
    ValueError
       If paramter values are invalid.
    """
    #validate image
    if image is None:
        raise ValueError("Image cannot be None.")
    if not isinstance(image,np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}.")
    if image.ndim not in [2,3]:
        raise ValueError("Expected a 2D or 3D image")
    
    #validate method

    if not isinstance(method,str):
        raise TypeError(f"Expected method to be a string, got {type(method)}.")
    
    method = method.lower()

    valid_methods = {
        "sobel",
        "canny",
        "laplacian"
    }

    if method not in valid_methods:
        raise ValueError("Unsupported method. Available options: {list(valid_methods)}.")
    
    #validate kernel_size

    if type(kernel_size) is not int:
        raise TypeError(f"Expected kernel_size to be an integer, got {type(kernel_size)}." )
    if kernel_size <= 0:
        raise ValueError("kernel_size must be greater than 0.")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer.")
    # -----------------------------
    # Validate Canny thresholds
    # -----------------------------

    if not isinstance(threshold1, (int, float)):
        raise TypeError(
            f"Expected threshold1 to be int or float, got {type(threshold1)}."
        )

    if not isinstance(threshold2, (int, float)):
        raise TypeError(
            f"Expected threshold2 to be int or float, got {type(threshold2)}."
        )

    if threshold1 < 0:
        raise ValueError(
            "threshold1 must be non-negative."
        )

    if threshold2 < 0:
        raise ValueError(
            "threshold2 must be non-negative."
        )

    if method == "canny" and threshold1 >= threshold2:
        raise ValueError(
            "threshold1 must be smaller than threshold2."
        )
        
    #convert to graayscale
    if image.ndim == 3:
        gray = convert_color(image,src="rgb",dst="gray")
    else:
        gray = image
    #sobel
    if method == "sobel":
        gx = cv2.Sobel(
            gray, 
            cv2.CV_64F,
            1,
            0,
            ksize=kernel_size
        )
        gy = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=kernel_size
        )
        edges = cv2.magnitude(
            gx,
            gy
        )
        edges = cv2.convertScaleAbs(edges)
    
    #laplacian
    elif method == "laplacian":
        edges = cv2.Laplacian(
            gray,
            cv2.CV_64F,
            ksize=kernel_size
        )
        edges = cv2.convertScaleAbs(edges)
    
    #canny
    else:
        edges = cv2.Canny(
            gray,
            threshold1,
            threshold2
        )
    return edges    
 
                        
#blend()
    
def blend(image1, image2, alpha):

    """
    Blend two images together using weighted averaging.

    Parameters
    ---------
    image1 : np.ndarray
        First input image.  
    image2 : np.ndarray
        Second input image.
    alpha : float,default = 0.5
        Weight of image1
        Must be between 0 and 1.

        Examples:
        alpha = 0.8 -> 80%  image1, 20% image2
        alpha = 0.5 -> Equal blend
        alpha = 0.2 -> 20% image1, 80% image2

    Returns
    -------
    np.ndarray
        Blended image.
    Raises
    ------
    TypeError
        If inputs have invalid image.
    ValueError
        If image shape do not match or alpha is invalid.
    """
    #validate image
    if image1 is None or image2 is None:
        raise ValueError("Image cannot be None.")
    if not isinstance(image1,np.ndarray) or not isinstance(image2,np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image1)} and {type(image2)}.")
    if image1.shape != image2.shape:
        raise ValueError("Image shapes do not match.")
    #validate alpha
    if not isinstance(alpha, (int,float)):
        raise TypeError(f"Expected alpha to be int or float, got {type(alpha)}.")
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1.")
    #blend image
    beta=1-alpha
    blended_image = cv2.addWeighted(
        image1,
        alpha,
        image2,
        beta,
        0.0
    )
    return blended_image
    

    








