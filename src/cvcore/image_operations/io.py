from pathlib import Path
import cv2
import numpy as np


# load_image()

def load_image(image_path,mode='color'):
    """
    Loads an image from a file path and converts it to the proper color space.

    Args:
        image_path : str or pathlib.Path
            Path to the local image file.
        mode(str):How to load the image('color', 'grayscale, or 'unchanged')

    Returns:
        np.ndarray: The loaded image as a Numpy array.
    """
    mode_map={
        "color":cv2.IMREAD_COLOR,
        "grayscale":cv2.IMREAD_GRAYSCALE,
        "unchanged":cv2.IMREAD_UNCHANGED
    
    }

    if mode not in mode_map:
        raise ValueError(f"Invalid mode {mode}. Choose from : {list(mode_map.keys())}")
    from pathlib import Path

    if not isinstance(image_path, (str, Path)):
        raise TypeError(
            "image_path must be a string or pathlib.Path object.")
    
    string_path=str(image_path)
    image=cv2.imread(string_path, mode_map[mode])

    if image is None:
        raise FileNotFoundError(f"Could not ope or find the image at: {string_path}")
    if mode == "color":
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    elif mode == "unchanged" and image.shape[-1] == 4:
        image=cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif mode == "unchanged" and image.shape[-1] == 3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
'''Summary of what happens by default:
Default Mode: "color"

Default Channel Order Matrix: RGB

Default Matrix Dimensions: 3D array (Height, Width, 3)
'''


# save_image()

def save_image(image_path,image):
    """
    Saves a Numpy image array to a file path, handling color space and data types.

    Args:
        image_path (str or pathlib.Path): The path to save the image to.
        image (np.ndarray): The image array to save.(supports unit8 or float32)
    
    Returns:
        bool: True if the image waas successfully saved.
    
    """
    if not isinstance(image_path, (str,Path)):
        raise TypeError(f"Expected image to be a np.ndarray, but got {type(image)}")
    string_path=str(image_path)

    if np.issubdtype(image.dtype, np.floating):
        save_ready_image = (image*255.0).astype(np.unit8)
    else:
        save_ready_image=image.copy()
    if len(save_ready_image.shape)==2:
        success = cv2.imwrite(string_path, save_ready_image)
    else:
        channels = save_ready_image.shape[-1]
        if channels == 3:
            save_ready_image = cv2.cvtColor(save_ready_image, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            save_ready_image = cv2.cvtColor(save_ready_image, cv2.COLOR_RGBA2BGRA)
    success=cv2.imwrite(string_path, save_ready_image)
    if not success:
        return IOError(f"Failed to write image to diks at: {string_path}")
    return True




# image_info()

def image_info(image):
    """
    Analyzes an image matrix and returns its core structural metadata.

    Args:
        image(np.ndarray): The image array to analyze.
    Returns:
         dict: A dictionary containing dimensions,channels,daata type, and pixel range.

    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be a np.ndarray, but got {type(image)}")
    shape= image.shape
    height = shape[0]
    width = shape[1]
    if len(shape) == 2:
        channels = 1
    else:
        channels = shape[2]

    min_val = float(image.min())
    max_val = float(image.max())
    
    info_dict= {
        "height":height,
        "width":width,
        "channels":channels,
        "data_type":image.dtype,
        "min_value":min_val,
        "max_value":max_val,
        "size_bytes":image.nbytes
    
    
    }
    return info_dict

#convert_color()
def convert_color(
        image,
        src,
        dst
):
    """
    Convert an image from one color space to another.

    Parameters
    ----------
    image : np.ndarray

        Input image.

    src : str

        Source color space.

        Supported:

        "rgb"
        "bgr"
        "gray"
        "hsv"
        "lab"

    dst : str

        Target color space.

    Returns
    -------
    np.ndarray

        Converted image.

    Raises
    ------
    ValueError
        If conversion is unsupported or image shape is invalid.

    TypeError
        If input types are invalid.
    """

    # -----------------------------
    # Validate image
    # -----------------------------

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if not isinstance(image,np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(image)}."
        )


    # -----------------------------
    # Validate spaces
    # -----------------------------

    if not isinstance(src,str):
        raise TypeError(
            "from_space must be a string."
        )

    if not isinstance(dst,str):
        raise TypeError(
            "to_space must be a string."
        )


    src=src.lower()
    dst=dst.lower()


    supported_spaces={

        "rgb",

        "bgr",

        "gray",

        "hsv",

        "lab"

    }


    if src not in supported_spaces:

        raise ValueError(

            f"Unsupported from_space "

            f"'{src}'."

        )


    if dst not in supported_spaces:

        raise ValueError(

            f"Unsupported to_space "

            f"'{dst}'."

        )


    # -----------------------------
    # Validate image dimensions
    # -----------------------------

    if src=="gray":

        if image.ndim != 2:

            raise ValueError(

                "Expected grayscale image "

                "with shape (H,W)."

            )

    else:

        if image.ndim != 3:

            raise ValueError(

                f"Expected {src} image "

                "with shape (H,W,3)."

            )

        if image.shape[2] != 3:

            raise ValueError(

                f"Expected 3 channels, "

                f"got {image.shape[2]}."

            )


    # -----------------------------
    # Same conversion
    # -----------------------------

    if src == dst:

        return image.copy()


    # -----------------------------
    # Conversion map
    # -----------------------------

    conversion_map={

        ("rgb","gray"):cv2.COLOR_RGB2GRAY,

        ("gray","rgb"):cv2.COLOR_GRAY2RGB,


        ("rgb","bgr"):cv2.COLOR_RGB2BGR,

        ("bgr","rgb"):cv2.COLOR_BGR2RGB,


        ("rgb","hsv"):cv2.COLOR_RGB2HSV,

        ("hsv","rgb"):cv2.COLOR_HSV2RGB,


        ("rgb","lab"):cv2.COLOR_RGB2LAB,

        ("lab","rgb"):cv2.COLOR_LAB2RGB,


        ("gray","bgr"):cv2.COLOR_GRAY2BGR,

        ("bgr","gray"):cv2.COLOR_BGR2GRAY,


        ("bgr","hsv"):cv2.COLOR_BGR2HSV,

        ("hsv","bgr"):cv2.COLOR_HSV2BGR,


        ("bgr","lab"):cv2.COLOR_BGR2LAB,

        ("lab","bgr"):cv2.COLOR_LAB2BGR,

    }


    key=(src,dst)


    if key not in conversion_map:

        raise ValueError(

            f"Conversion from "

            f"{src}"

            f" to "

            f"{dst}"

            f" is not supported."

        )


    converted_image=cv2.cvtColor(

        image,

        conversion_map[key]

    )


    return converted_image

