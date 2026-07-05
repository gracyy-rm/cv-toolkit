import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from ..image_operations import blend

from ..image_operations import load_image

#plotting-plot - done 
#plot_image_row - change -done
def image_row(**images):
    """
    Display multiple images in a single horizontal row.

    Parameters
    ----------
    **images : dict
        Keyword arguments where:

        - Keyword : Image title.
        - Value   : Image as a NumPy array or a local image path
                    (str or pathlib.Path).

    Raises
    ------
    ValueError
        - If no images are provided.
        - If an image is empty.
        - If an image is not 2D or 3D.
        - If a color image does not have 3 (RGB) or 4 (RGBA) channels.

    TypeError
        If an image is not a NumPy array, string path,
        or pathlib.Path object.

    Returns
    -------
    None
        Displays all images in a single horizontal row.
    """

    if not images:
        raise ValueError("At least one image must be provided.")

    n_images = len(images)

    WIDTH_PER_IMAGE = 5
    ROW_HEIGHT = 5
    MAX_ROW_WIDTH = 25

    figure_width = min(
        n_images * WIDTH_PER_IMAGE,
        MAX_ROW_WIDTH
    )

    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(figure_width, ROW_HEIGHT)
    )

    # Make axes iterable when only one image is provided
    if n_images == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, images.items()):

        # Load image if a path is provided
        if isinstance(image, (str, Path)):
            image = load_image(image)

        # Validate input type
        elif not isinstance(image, np.ndarray):
            raise TypeError(
                f"Invalid input for '{title}'. "
                "Expected a NumPy ndarray, str, or pathlib.Path."
            )

        # Validate image is not empty
        if image.size == 0:
            raise ValueError(
                f"Image '{title}' is empty."
            )

        # Validate image dimensions
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Image '{title}' must be a 2D or 3D NumPy array."
            )

        # Validate channel count for color images
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError(
                f"Image '{title}' must have 3 (RGB) or 4 (RGBA) channels."
            )
        is_grayscale = (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)
        cmap_val = "gray" if is_grayscale else None

        # Squeeze out the trailing channel dimension if it's (H, W, 1) so matplotlib accepts it
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.squeeze(image, axis=2)

        ax.imshow(image, cmap=cmap_val)
        # --------------------------------
        
        ax.set_title(title)
        ax.axis("off")
        

    plt.tight_layout()
    plt.show()

# image_grid() - done
#grid_size = rows,cols-done

def image_grid(images, rows, cols, titles=None):
    """
    Display multiple images in a grid layout.

    Parameters
    ----------
    images : list or tuple
        Collection of images to display. Each image can be:
        - numpy.ndarray
        - str (image path)
        - pathlib.Path

    rows : int
        Number of rows in the image grid.

    cols : int
        Number of columns in the image grid.

    titles : list or tuple of str, optional
        Titles corresponding to each image.
        If None, no titles are displayed.

    Raises
    ------
    TypeError
        - If images is not a list or tuple.
        - If titles is provided but is not a list or tuple.
        - If rows or cols are not integers.
        - If an image is not a NumPy ndarray, str, or pathlib.Path.

    ValueError
        - If images is empty.
        - If titles length does not match images.
        - If rows or cols are less than or equal to zero.
        - If the grid cannot accommodate all images.
        - If an image is empty.
        - If an image is not 2D or 3D.
        - If a color image does not have 3 (RGB) or 4 (RGBA) channels.

    Returns
    -------
    None
        Displays all images in the specified grid.
    """

    # Validate images
    if not isinstance(images, (list, tuple)):
        raise TypeError(
            "'images' must be a list or tuple."
        )

    if len(images) == 0:
        raise ValueError(
            "'images' cannot be empty."
        )

    # Validate rows
    if not isinstance(rows, int):
        raise TypeError(
            "'rows' must be an integer."
        )

    if rows <= 0:
        raise ValueError(
            "'rows' must be greater than zero."
        )

    # Validate columns
    if not isinstance(cols, int):
        raise TypeError(
            "'cols' must be an integer."
        )

    if cols <= 0:
        raise ValueError(
            "'cols' must be greater than zero."
        )

    # Validate titles
    if titles is None:
        titles = [""] * len(images)

    elif not isinstance(titles, (list, tuple)):
        raise TypeError(
            "'titles' must be a list or tuple."
        )

    elif len(titles) != len(images):
        raise ValueError(
            "'titles' and 'images' must have the same length."
        )

    # Validate grid capacity
    if len(images) > rows * cols:
        raise ValueError(
            "The specified grid is too small to accommodate all images."
        )

    WIDTH_PER_IMAGE = 5
    HEIGHT_PER_IMAGE = 5
    MAX_FIGURE_WIDTH = 25
    MAX_FIGURE_HEIGHT = 20

    figure_width = min(
        cols * WIDTH_PER_IMAGE,
        MAX_FIGURE_WIDTH
    )

    figure_height = min(
        rows * HEIGHT_PER_IMAGE,
        MAX_FIGURE_HEIGHT
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figure_width, figure_height)
    )

    # Convert the subplot axes into a 1D NumPy array so every subplot
    # can be accessed using a single loop, regardless of the grid shape.
    axes = np.array(axes).flatten()

    for ax, image, title in zip(axes, images, titles):

        # Load image from path
        if isinstance(image, (str, Path)):
            image = load_image(image)

        # Validate image type
        elif not isinstance(image, np.ndarray):
            raise TypeError(
                f"Invalid image for '{title}'. "
                "Expected a NumPy ndarray, str, or pathlib.Path."
            )

        # Validate image
        if image.size == 0:
            raise ValueError(
                f"Image '{title}' is empty."
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                f"Image '{title}' must be a 2D or 3D NumPy array."
            )

        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError(
                f"Image '{title}' must have 3 (RGB) or 4 (RGBA) channels."
            )
        is_grayscale = (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)
        cmap_val = "gray" if is_grayscale else None

        # Squeeze out single trailing channel dimensions if it's (H, W, 1)
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.squeeze(image, axis=2)

        ax.imshow(image, cmap=cmap_val)

        ax.set_title(title)
        ax.axis("off")

    # Hide unused subplots
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# bbox()


def bbox(image, bboxes, labels):
    """
    Display an image with annotated bounding boxes.

    Parameters
    ----------
    image : numpy.ndarray, str, or pathlib.Path
        Input image. Can be:
        - numpy.ndarray
        - str (image path)
        - pathlib.Path

    bboxes : list or tuple
        Collection of bounding boxes in the format:
        [[x, y, width, height], ...]

    labels : list or tuple
        Labels corresponding to each bounding box.

    Raises
    ------
    TypeError
        - If image is not a NumPy ndarray, str, or pathlib.Path.
        - If bboxes is not a list or tuple.
        - If labels is not a list or tuple.
        - If a bounding box is not a list or tuple.
        - If a bounding box contains non-numeric values.

    ValueError
        - If bboxes is empty.
        - If labels is empty.
        - If the number of labels does not match the number of bounding boxes.
        - If the image is empty.
        - If the image is not 2D or 3D.
        - If a color image does not have 3 (RGB) or 4 (RGBA) channels.
        - If a bounding box does not contain exactly four values.
        - If a bounding box has non-positive width or height.

    Returns
    -------
    None
        Displays the annotated image.
    """

    # Load image from path
    if isinstance(image, (str, Path)):
        image = load_image(image)

    # Validate image type
    elif not isinstance(image, np.ndarray):
        raise TypeError(
            "Expected image to be a NumPy ndarray, str, or pathlib.Path."
        )

    # Validate image
    if image.size == 0:
        raise ValueError(
            "Image is empty."
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            "Image must be a 2D or 3D NumPy array."
        )

    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError(
            "Image must have 3 (RGB) or 4 (RGBA) channels."
        )

    # Validate bboxes
    if not isinstance(bboxes, (list, tuple)):
        raise TypeError(
            "'bboxes' must be a list or tuple."
        )

    if len(bboxes) == 0:
        raise ValueError(
            "'bboxes' cannot be empty."
        )

    # Validate labels
    if not isinstance(labels, (list, tuple)):
        raise TypeError(
            "'labels' must be a list or tuple."
        )

    if len(labels) == 0:
        raise ValueError(
            "'labels' cannot be empty."
        )

    if len(labels) != len(bboxes):
        raise ValueError(
            "'labels' and 'bboxes' must have the same length."
        )

    # Create a copy so the original image remains unchanged
    annotated_image = image.copy()

    for index, (box, label) in enumerate(zip(bboxes, labels)):

        # Validate bounding box type
        if not isinstance(box, (list, tuple)):
            raise TypeError(
                f"Bounding box at index {index} must be a list or tuple."
            )

        # Validate bounding box length
        if len(box) != 4:
            raise ValueError(
                f"Bounding box at index {index} must contain exactly four values: "
                "[x, y, width, height]."
            )

        # Validate numeric values
        if not all(isinstance(value, (int, float)) for value in box):
            raise TypeError(
                f"Bounding box at index {index} must contain only numeric values."
            )

        x, y, width, height = box

        # Validate dimensions
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Bounding box at index {index} must have positive width and height."
            )

        # Convert coordinates to integers for OpenCV
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # Draw label above the bounding box
        cv2.putText(
            annotated_image,
            str(label),
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
    is_grayscale = (annotated_image.ndim == 2) or (annotated_image.ndim == 3 and annotated_image.shape[2] == 1)
    cmap_val = "gray" if is_grayscale else None

    # Strip out single channel wrappers like (H, W, 1) so matplotlib renders cleanly
    if annotated_image.ndim == 3 and annotated_image.shape[2] == 1:
        annotated_image = np.squeeze(annotated_image, axis=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_image, cmap=cmap_val)

    plt.axis("off")
    plt.show()

# segments
#bbox_flag=true - more acuurate position of the mask 
def segment(image, masks, labels, bbox_flag=False):
    """
    Overlay color-coded segmentation masks and annotations onto an image.

    Parameters
    ----------
    image : numpy.ndarray, str, or pathlib.Path
        Input image (2D grayscale, 3D RGB, or 3D RGBA).
    masks : list or tuple of (numpy.ndarray, str, or pathlib.Path)
        Sequence of 2D binary segmentation masks matching the image dimensions.
    labels : list or tuple of str
        Text labels matching each mask in the `masks` sequence.
    bbox_flag : bool, optional
        If True, draws bounding boxes around segmented objects. Default is False.

    Returns
    -------
    None
        Renders the final visual output directly via Matplotlib.

    Raises
    ------
    TypeError
        If input types for image, masks, labels, or bbox_flag are invalid.
    ValueError
        If shapes mismatch, files are empty, or sequence lengths do not align.
    
    """
    # --- 1. Input Validation ---
    if isinstance(image, (str, Path)):
        image = load_image(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError("Expected image to be a NumPy ndarray, str, or pathlib.Path.")

    if image.size == 0:
        raise ValueError("Image is empty.")
        
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("Image must be a 2D or a 3-4 channel 3D NumPy array.")

    if not isinstance(masks, (list, tuple)) or not isinstance(labels, (list, tuple)):
        raise TypeError("'masks' and 'labels' must be a list or tuple.")

    if len(masks) == 0 or len(labels) == 0 or len(masks) != len(labels):
        raise ValueError("'masks' and 'labels' cannot be empty and must match in length.")
        
    if not isinstance(bbox_flag, bool):
        raise TypeError("'bbox_flag' must be a boolean.")

    # --- 2. Setup Canvas & Palettes ---
    segmented_image = image.copy()
    overlay = np.zeros_like(segmented_image)
    
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    # We will cache the extracted coordinates here to avoid reloading files or recalculating arrays
    processed_objects = []

    # --- 3. Pass 1: Process Masks & Construct Overlay ---
    for index, (mask_input, label) in enumerate(zip(masks, labels)):
        # Load mask if it's a path, otherwise use it directly
        mask = load_image(mask_input, mode="grayscale") if isinstance(mask_input, (str, Path)) else mask_input
        
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError(f"Mask '{label}' must be a valid 2D NumPy array matching the image dimensions.")
            
        binary_mask = mask > 0
        ys, xs = np.where(binary_mask)
        
        # Skip if the mask contains no true object pixels
        if len(xs) == 0:
            continue
            
        color = COLORS[index % len(COLORS)]
        
        # Color the overlay mask
        overlay[binary_mask] = color
        
        # Calculate bounding box parameters once
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # Find label placement: the minimum X coordinate belonging to the minimum Y coordinate
        label_x = xs[ys == y_min].min()
        
        # Store for the drawing phase
        processed_objects.append({
            'label': str(label),
            'color': color,
            'label_pos': (label_x, max(y_min - 10, 0)),
            'bbox': (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        })

    # --- 4. The Single Blend Operation ---
    # Blends all masks simultaneously so colors remain crisp and clean
    segmented_image = blend(image1=segmented_image, image2=overlay, alpha=0.75)

    # --- 5. Pass 2: Annotate the Blended Image ---
    for obj in processed_objects:
        # Draw text label
        cv2.putText(
            segmented_image, obj['label'], obj['label_pos'],
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj['color'], 2, cv2.LINE_AA
        )
        
        # Draw bounding box if flagged
        if bbox_flag:
            x, y, w, h = obj['bbox']
            cv2.rectangle(
                segmented_image, (x, y), (x + w - 1, y + h - 1), obj['color'], 2
            )

    # --- 6. Render Final Figure ---
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.show()
    