import numpy as np 
import cv2
#crop()
def crop(
    image,
    x,
    y,
    width,
    height,
    pad=0.0
):
    """
    Extract a Region of Interest (ROI) from an image.

    The ROI is defined by its top-left corner `(x, y)` and its
    dimensions `(width, height)`. Optional padding can be added
    around the ROI before cropping.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or color image.

    x : int
        Left coordinate of the ROI.

    y : int
        Top coordinate of the ROI.

    width : int
        Width of the ROI in pixels.

    height : int
        Height of the ROI in pixels.

    pad : float, default=0.0
        Fractional padding added equally around the ROI.

        Must satisfy:

            0 <= pad <= 1

        Example
        -------
        pad=0.2

        Expands the crop region by 20%.

    Returns
    -------
    np.ndarray
        Cropped image as a detached copy.

    Raises
    ------
    TypeError
        If input types are invalid.

    ValueError
        If coordinates or padding values are invalid.
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

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D image, got {image.ndim}D."
        )

    # -------------------------
    # Validate coordinates
    # -------------------------

    values = {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }

    for name, value in values.items():
        # bool is a subclass of int
        if type(value) is not int:
            raise TypeError(
                f"{name} must be an int, got {type(value).__name__}"
            )

    if x < 0 or y < 0:
        raise ValueError("x and y must be non-negative.")

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero.")

    img_h, img_w = image.shape[:2]

    # Convert dimensions to end coordinates
    x2 = x + width
    y2 = y + height

    if x2 > img_w:
        raise ValueError(
            f"Crop exceeds image width ({img_w})."
        )

    if y2 > img_h:
        raise ValueError(
            f"Crop exceeds image height ({img_h})."
        )

    # -------------------------
    # Validate padding
    # -------------------------

    if not isinstance(pad, (int, float)):
        raise TypeError(
            f"pad must be numeric, got {type(pad).__name__}"
        )

    pad = float(pad)

    if not (0 <= pad <= 1):
        raise ValueError(
            "pad must lie between 0 and 1."
        )

    # -------------------------
    # Apply padding
    # -------------------------

    if pad > 0:

        pad_x = int(width * pad / 2)
        pad_y = int(height * pad / 2)

        x = max(0, x - pad_x)
        y = max(0, y - pad_y)

        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)

    # -------------------------
    # Crop ROI
    # -------------------------

    return image[y:y2, x:x2].copy()

# resize()
def resize(
    image,
    size,
    interpolation="linear"
):
    """
    Resize an image using explicit dimensions or by specifying
    the desired length of its longest side.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or color image.

    size : tuple[int, int] or int
        Resize specification.

        - If a tuple ``(width, height)`` is provided, the image is
          resized to those exact dimensions.

        - If an integer is provided, the longest side of the image
          is resized to that value while preserving the aspect ratio.

    interpolation : str, default="linear"
        Interpolation method.

        Available options:

        - "nearest"
        - "linear"
        - "cubic"
        - "area"

    Returns
    -------
    np.ndarray
        Resized image.
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

    original_height, original_width = image.shape[:2]

    # -------------------------
    # Validate size
    # -------------------------

    if isinstance(size, tuple):

        if len(size) != 2:
            raise ValueError(
                "size must contain exactly (width, height)."
            )

        new_width, new_height = size

        if type(new_width) is not int or type(new_height) is not int:
            raise TypeError(
                "Width and height must be integers."
            )

        if new_width <= 0 or new_height <= 0:
            raise ValueError(
                "Width and height must be positive."
            )

    elif type(size) is int:

        if size <= 0:
            raise ValueError(
                "size must be positive."
            )

        longest_side = max(original_width, original_height)
        resize_scale = size / longest_side

        new_width = round(original_width * resize_scale)
        new_height = round(original_height * resize_scale)

    else:
        raise TypeError(
            "size must be either an int or a tuple (width, height)."
        )

    # -------------------------
    # Validate interpolation
    # -------------------------

    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }

    if interpolation not in interpolation_map:
        raise ValueError(
            f"Unsupported interpolation. "
            f"Choose from {list(interpolation_map.keys())}."
        )

    # -------------------------
    # Resize image
    # -------------------------

    resized_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation_map[interpolation],
    )

    return resized_image


#rotate

def rotate(
        image,
        angle,
        expand=True,
        interpolation="linear",
        border_value=(0,0,0)
):
    """
    Rotate an image by a given angle.

    Parameters:

    image : np.ndarray
        Input image array.
    angle : int or float
        Rotation angle in degrees.

        Positive -> Counter-clockwise
        Negative -> Clockwise
    expand : bool, default = True
        if True:
            Expand canvas to avoid cropping
        if False:
            keep original image size.
    interpolation : str, default='linear'
        Interpolation method - 
         
        -nearest
        -linear
        -cubic

    border_value  : tuple
        RGB color used to fill empty regions.

        Example :
        (0,0,0) -> Black
        (255,255,255) ->WHITE

    Returns:
    np.ndarray
        Rotated image.

    Interpolation estimates pixel values because rotated pixels rarely land exactly on the image grid.

    """
    if image is None:
        raise ValueError(
            "Image cannot be None"
        )
    if not isinstance(image,np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(image)}."
        )
    if image.ndim not in [2,3]:
        raise ValueError(
            "Expected a 2D(grayscale)"
            "or 3D (color) image."
        )
    if type(angle) is bool:
        raise TypeError("angle cannot be booolean.")
    if type(angle) not in [int,float]:
        raise TypeError("angle must be an integer or float")
    if angle == 0:
        return image.copy()
    if type(expand) is not bool:
        raise TypeError("expand must be a boolean")
    interpolation_map={
        "nearest" : cv2.INTER_NEAREST,
        "linear" : cv2.INTER_LINEAR,
        "cubic" : cv2.INTER_CUBIC
    }
    if interpolation not in interpolation_map:
        raise ValueError(
            f"Unsupported interpolation."
            f"Available: {list(interpolation_map.keys())}"
        )
    if image.ndim == 3:
        if not isinstance(border_value, tuple) or len(border_value)!=3:
            raise TypeError("border value must be a tuple of 3 integers")   
    h,w=image.shape[:2]
    center = (w/2, h/2) 

    M=cv2.getRotationMatrix2D(center,angle,1.0)

    if expand:
        cos_theta = abs(M[0,0])
        sin_theta = abs(M[0,1])
        new_w = int((h * sin_theta) + (w * cos_theta))
        new_h = int((h * cos_theta) + (w * sin_theta))
        M[0,2] += (new_w/2) - center[0]
        M[1,2] += (new_h/2) - center[1]
    else:
        new_h=h
        new_w=w

    rotated_image=cv2.warpAffine(
        image,
        M,
        (new_w,new_h),
        flags=interpolation_map[interpolation],
        borderValue=border_value
    )
    return rotated_image

#flip
def flip(image,type="horizontal"):
    """
    Flip an image along a specified axis.

    Parameters
    ----------
    image : np.ndarray
    Input Image.

    Supported formats
    -grayscale : (h,w)
    -RGB : (h,w,c) where c=3

    type : str, default="horizontal"
    Direction of flipping

    AVALIABLE OPTIONS:
    -"horizontal" : Flip left <-> right
    -"vertical" : Flip top <-> bottom
    -"both" : Flip horizontally and vertically

    Returns
     np.ndarray
        Flipped image aas a new Numpy array.

    Raises:
    
    ValueError:
    -If mode is not supported.
    -IF image is None
    -If image dimensions are invalid.
    TypeError:
    -If image is not a numpy array.
    """

    #validate image
    if image is None:
        raise ValueError("Image cannot be None.")

    if image.ndim not in [2,3]:
        raise ValueError("Expected a 2D or 3D image.")
    if not isinstance(image,np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}.")
    
    #valdate mode

    if not isinstance(type,str):
        raise TypeError(f"Expected mode to be a string, got {type(type)}.")
    flip_map={
        "horizontal" : 1,
        "vertical" : 0,
        "both" : -1
    }
    if type not in flip_map:
        raise ValueError(f"UNsupported mode. Avaliable optionss:{list(flip_map.keys())}.")
    #perform flippping

    flipped_image=cv2.flip(image,flip_map[type])

    #cv2.fllip returns a new image with indepenndent memory.

    return flipped_image

# translate
def translate(image,tx,ty,border_value=(0,0,0)):
    """
    Translate(shift) an image horizontally and verticaaly.
    this function shifts every pixel in the image acooding 
    to the given translation values.
    x'=x+tx
    y'=y+ty

    The transformation is performed using OpenCV's
    affine transformation function : cv2.wrapAffine().

    Parameters 
    ----------
    image : np.ndarray
        Input image.
    tx : int or float 
        Translation along the x-axis (horizontal shift).
        Positive -> move right
        Negative -> move left
    ty : int or float
        Translation along the y-axis (vertical shift).
        Positive -> move down
        Negative -> move up  

    border_value : int,float or tuple, default=0
        value used to fill the newly created empty regions.
        Examples:
        Grayscale :
            border_value = 0
            border_value = 255
        RGB :
            border_value = (0,0,0) -> Black
            border_value = (255,255,255) -> White
    
    Returns
    -------
    np.ndarray
        Translated image as a new Numpy array.

    Raises
    ------
    ValueError:
        - If image is None.
        - If image dimensions are invalid.
        - If RGB border tuple does not have 3 values.
    TypeError:
        - If image is not a numpy array.
        - If tx or ty are not int or float.
        - If border_value is not int, float, or tuple.
    """

    #validate image
    if image is None:
        raise ValueError("Image cannot be None.")
    if not isinstance(image,np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}.")
    if image.ndim not in [2,3]:
        raise ValueError("Expected a 2D or 3D image.")
    
    #validate tx 
    if type(tx) is bool:
        raise TypeError("tx cannot be boolean.")
    if not isinstance(tx,(int,float)):
        raise TypeError(f"Expected tx to be int or float, got {type(tx)}.")

    #validate ty
    if type(ty) is bool:
        raise TypeError("ty cannot be boolean.")
    if not isinstance(ty,(int,float)):
        raise TypeError(f"Expected ty to be int or float, got {type(ty)}.")

    #validate border_value
    if type(border_value) is bool:
        raise TypeError("border_value cannot be boolean.")
    #rgb image
    if image.ndim == 3:
        if isinstance(border_value, tuple):
            if len(border_value) != 3:
                raise TypeError("For RGB images, border_value must be a tuple of 3 values.")
            for value in border_value:
                if type(value) not in [int,float]:
                    raise TypeError("Each value in border_value tuple must be int or float.")
        elif not isinstance(border_value,(int,float)):
            raise TypeError("For RGB images, border_value must be an int, float, or a tuple of 3 values.")
    #grayscale image
    else:
        if not isinstance(border_value,(int,float)):
            raise TypeError("For grayscale images, border_value must be an int or float.")
    
    #get image dimensions
    height,width=image.shape[:2]

    #create translation matrix
    M=np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    #apply translation using cv2.warpAffine
    translated_image=cv2.warpAffine(
        image,
        M,
        (width,height),
        borderValue=border_value
    )

    return translated_image
#pad()
def pad(image,top,bottom,left,right,pad_type="constant",value=0):
    """
    Add padding (extra pixels) around an image.
    Parameters :
    -----------
    image : np.ndarray
        Input image.
    top : int
        Number of pixels to pad on the top.
    bottom : int
        Number of pixels to pad on the bottom.
    left : int
        Number of pixels to pad on the left.
    right : int
        Number of pixels to pad on the right.
    pad_type : str, default="constant"
        Padding type. Options are "constant", "edge", "reflect", or "symmetric".
    value : int, float, or tuple, default=0
        used only wehn pad_type="constant".
        Grayscale :
            value = 0
            value = 255
        RGB :
            value = (0,0,0) -> Black
            value = (255,255,255) -> White  
    Returns
    -------
    np.ndarray
        Padded image as a new Numpy array.
    """

    #validate imaage
    if image is None:
        raise ValueError("Image cannot be None.")
    if not isinstance(image,np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}.")
    if image.ndim not in [2,3]:
        raise ValueError("Expected a 2D or 3D image.")
    #validate padding values
    paddings = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right
    }
    for name,value_pad in paddings.items():
        if type(value_pad) is not int:
            raise TypeError(f"{name} must be an integer, got {type(value_pad)}.")
        if value_pad < 0:
            raise ValueError(f"{name} must be non-negative, got {value_pad}.")
        
    #validate mode
        pad_map = {

        "constant": cv2.BORDER_CONSTANT,

        "replicate": cv2.BORDER_REPLICATE,

        "reflect": cv2.BORDER_REFLECT,

        "wrap": cv2.BORDER_WRAP
    }

    if not isinstance(pad_type, str):

        raise TypeError(

            "mode must "

            "be a string."

        )

    if pad_type not in pad_map:

        raise ValueError(

            f"Unsupported mode."

            f" Available options: "

            f"{list(pad_map.keys())}"

        )
    #validate constant value
    if pad_type == "constant" :
        if type(value) is bool:
            raise TypeError("value cannot be boolean.")
        #rgb

        if image.ndim == 3:
            if isinstance(value, tuple):
                if len(value) != 3:
                    raise TypeError("For RGB images, value must be a tuple of 3 values.")
                for v in value:
                    if type(v) not in [int,float]:
                        raise TypeError("Each value in the value tuple must be int or float.")
            elif not isinstance(value,(int,float)):
                raise TypeError("For RGB images, value must be an int, float, or a tuple of 3 values.")
        #grayscale
        else:
            if not isinstance(value,(int,float)):
                raise TypeError("For grayscale images, value must be an int or float.")
    #add padding 
    padded_image=cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        pad_map[pad_type],   
        value=value
    )
    return padded_image
