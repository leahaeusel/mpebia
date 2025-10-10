"""Utility functions for positioning of plot elements."""


def get_axis_bbox(ax, fig):
    """Get the bounding box of an axis in the figure.

    Args:
        ax (Axes): The axis object for which the bounding box is to be retrieved.
        fig (Figure): The figure object that contains the axis.

    Returns:
        Bbox: The bounding box of the axis in figure coordinates.
    """
    # Get the bounding box of the axis including labels
    bbox_wrong_coords = ax.get_tightbbox(fig.canvas.get_renderer())
    # Transform the bounding box to figure coordinates
    bbox = bbox_wrong_coords.transformed(fig.transFigure.inverted())
    return bbox
