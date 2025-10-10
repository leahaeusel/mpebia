"""Parameters for the relative increase in information gain on a grid."""


class ParametersRIIGGrid:
    """Wrapper for parameters for evaluating the RIIG on a grid."""

    num_snrs_1 = 50
    snr_1_min = 10
    snr_1_max = 1.0e6

    num_snrs_2 = 50
    snr_2_min = 10
    snr_2_max = 1.0e6

    num_kjs = 10
    kj_min = -1.5
    kj_max = 0.0

    fixed_snr_1 = snr_1_min
    fixed_snr_2 = snr_2_max
