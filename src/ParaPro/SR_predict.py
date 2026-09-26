import numpy as np

# Symbolic Regresion parameters refitted
P0 = [
    1.11094,  # A
    1.468,  # B
    0.650881,  # C
    0.786251,  # D
    1.03437,  # E
    -0.588886,  # F
]

A, B, C, D, E, F = P0


def SR_predict(
    X_in,
    A=A,
    B=B,
    C=C,
    D=D,
    E=E,
    F=F,  # eta, R, delta, Cp
):
    """
    Formula from the Symbolic Regression model refitted to the training data. Return the predicted J value based on the input features.

    Parameters
    ----------
    X_in : tuple
        A tuple containing the input features (eta, R, delta, Cp).
    *args: float
        Model parameters (A, B, C, D, E, F) used in the formula.

    Returns
    -------
    float
        The predicted J value based on the input features and model parameters.


    The original formula is:
    ```log(R / ((((delta / 1.3897328) + (1.3516902/ ((((log_R / Cp) + sqrt_eta_over_R) * R) ^0.7988522))) / (sqrt_eta_over_R - -0.5891432)) +((delta * inv_eta) ^ 1.0286156))) + (sqrt(log(Cp))* -0.66001064)```
    """
    eta, R, delta, Cp = X_in
    eps = 1e-9  # small value to avoid division by zero or log of zero

    # build component that get used multiple times
    sqrt_eta_over_R = np.sqrt(eta / R)

    theta = (((np.log(R) / (Cp + eps)) + sqrt_eta_over_R) * R) ** D

    # building psi
    t1 = delta / (A + eps)  # first frac in psi
    t2 = B / (theta + eps)  # second frac in psi

    denom = sqrt_eta_over_R + C  # middle term in psi
    denom = np.where(np.abs(denom) < 1e-6, 1e-6 * np.sign(denom), denom)

    t3 = (delta * (1.0 / eta)) ** E  # final term in psi

    psi = (t1 + t2) / denom + t3

    inner_val = R / (psi + eps)

    log_J = np.log(inner_val + eps) + F * np.sqrt(np.log(Cp + eps))

    return float(np.exp(np.clip(log_J, -50, 20)))
