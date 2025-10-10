"""RC parameters for matplotlib."""

from matplotlib import rc, rcParams, rcParamsDefault

rcParams.update(rcParamsDefault)
rc(
    "font",
    family="serif",
    serif=["Computer Modern Roman"],
    monospace=["Computer Modern Typewriter"],
    size="12.0",
)
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}  \usepackage{bm}")
rc("figure", dpi=300)
rc("axes", labelpad=6.0)
