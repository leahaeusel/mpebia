"""Utility objects and functions for styling plots."""

from mpebia.plotting import colors

arrowprops = dict(
    arrowstyle="simple, head_length=0.8, head_width=0.8",
    color=colors.ARROWS,
    mutation_scale=15,
    clip_on=False,
)


def get_label_kwargs(ax=None, fontsize=12):
    """Get kwargs for matplotlib annotations.

    Args:
        ax (Axes): Matplotlib axes object.
        fontsize (int, optional): Font size for the annotation. Defaults to 12.

    Returns:
        dict: Dictionary of label kwargs.
    """
    label_kwargs = dict(
        fontsize=fontsize,
        ha="center",
        va="center",
        zorder=12,
        annotation_clip=False,
    )
    if ax is not None:
        label_kwargs["xycoords"] = ax.transAxes
        label_kwargs["textcoords"] = ax.transAxes

    return label_kwargs


def get_arrow_kwargs(ax):
    """Get kwargs for arrow annotations.

    Args:
        ax (Axes): Matplotlib axes object.

    Returns:
        dict: Dictionary of arrow label kwargs.
    """
    arrow_label_kwargs = get_label_kwargs(ax, 14)
    arrow_label_kwargs["arrowprops"] = arrowprops
    return arrow_label_kwargs
