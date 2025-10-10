"""Parameters for the relative increase in information gain on a grid."""


class ParametersRIIGGrid:
    """Wrapper for parameters for evaluating the RIIG on a grid."""

    num_obs_I_max_exp = 11

    num_snrs = 50
    snr_min = 50
    snr_max = int(1.0e5)

    seed_noise = 42
