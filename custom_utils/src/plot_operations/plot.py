import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

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

        ax.imshow(image)
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

        ax.imshow(image)
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

    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()

# segments

def segment(image, masks, labels):
    """
    Display an image with segmented object masks overlaid.

    Parameters
    ----------
    image : numpy.ndarray, str, or pathlib.Path
        Input image. Can be:
        - numpy.ndarray
        - str (image path)
        - pathlib.Path

    masks : list or tuple
        Collection of segmentation masks. Each mask can be:
        - numpy.ndarray
        - str (mask path)
        - pathlib.Path

        Every mask must:
        - be a 2D array.
        - have the same height and width as the image.

    labels : list or tuple
        Labels corresponding to each segmentation mask.

    Raises
    ------
    TypeError
        - If image is not a NumPy ndarray, str, or pathlib.Path.
        - If masks is not a list or tuple.
        - If labels is not a list or tuple.
        - If a mask is not a NumPy ndarray, str, or pathlib.Path.

    ValueError
        - If image is empty.
        - If image is not a 2D or 3D array.
        - If a color image does not have 3 (RGB) or 4 (RGBA) channels.
        - If masks is empty.
        - If labels is empty.
        - If labels and masks have different lengths.
        - If a mask is empty.
        - If a mask is not a 2D array.
        - If a mask size does not match the image.

    Returns
    -------
    None
        Displays the segmented image.
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

    # Validate masks
    if not isinstance(masks, (list, tuple)):
        raise TypeError(
            "'masks' must be a list or tuple."
        )

    if len(masks) == 0:
        raise ValueError(
            "'masks' cannot be empty."
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

    if len(labels) != len(masks):
        raise ValueError(
            "'labels' and 'masks' must have the same length."
        )

    # Create copy of image
    segmented_image = image.copy()

    # Fixed RGB color palette
    COLORS = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
    ]

    ALPHA = 0.5

    for index, (mask, label) in enumerate(zip(masks, labels)):

        # Load mask from path
        if isinstance(mask, (str, Path)):
            mask = load_image(mask, mode="grayscale")

        # Validate mask type
        elif not isinstance(mask, np.ndarray):
            raise TypeError(
                f"Invalid mask for '{label}'. "
                "Expected a NumPy ndarray, str, or pathlib.Path."
            )

        # Validate mask
        if mask.size == 0:
            raise ValueError(
                f"Mask '{label}' is empty."
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Mask '{label}' must be a 2D NumPy array."
            )

        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask '{label}' must have the same height and width as the image."
            )

        # Convert mask to binary
        binary_mask = mask > 0

        # Select color
        color = COLORS[index % len(COLORS)]

        # Create overlay for current mask
        overlay = np.zeros_like(segmented_image)

        # Paint object pixels
        overlay[binary_mask] = color

        # Blend overlay with image
        segmented_image = cv2.addWeighted(
            segmented_image,
            1.0,
            overlay,
            ALPHA,
            0
        )

        # Find top-left pixel of object
        points = np.argwhere(binary_mask)

        if len(points) > 0:

            y = points[:, 0].min()
            x = points[points[:, 0] == y][:, 1].min()

            cv2.putText(
                segmented_image,
                str(label),
                (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.show()


