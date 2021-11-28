from functools import lru_cache

import numpy as np
from numba import vectorize
from scipy.special import jn as Jn

from .correlations import Pk, k

twopi2 = 2 * np.pi ** 2


@vectorize("float64(float64)")
def j0(x):
    if x < 1e-2:
        x2 = x ** 2
        return 1 - x2 / 6 + x2 ** 2 / 120
    else:
        return np.sin(x) / x


@vectorize("float64(float64)")
def j1(x):
    if x < 1e-1:
        x2 = x ** 2
        tmp = x * (1 / 3 - x2 / 30)
        if x > 1e-3:
            tmp += x ** 5 / 840
        if x > 1e-2:
            tmp += -(x ** 7) / 45360
        return tmp
    else:
        return (np.sin(x) / x - np.cos(x)) / x


@vectorize("float64(float64)")
def j1_o_x(x):
    x2 = x ** 2
    if x < 0.5:
        return (
            1 / 3
            - x2 / 30
            + x2 ** 2 / 840
            - x2 ** 3 / 45360
            + x2 ** 4 / 3991680
            - x2 ** 5 / 518918400
        )
    else:
        return (np.sin(x) / x - np.cos(x)) / x2


@vectorize("float64(float64)")
def j2(x):
    x2 = x ** 2
    if x < 0.4:
        tmp = x2 / 15 - x2 ** 2 / 210
        if x > 1e-3:
            tmp += x2 ** 3 / 7560
        if x > 1e-2:
            tmp += -(x2 ** 4) / 498960 + x2 ** 5 / 51891840
        return tmp
    else:
        return (3 / x2 - 1) * np.sin(x) / x - 3 * np.cos(x) / x2


def J1_o_x(x):
    return Jn(1, x) / x


@vectorize("float64(float64)")
def xj2_mj1_o_x(x):
    if x < 1e-4:
        x2 = x ** 2
        return -(x2 ** 2) / 168 + x2 / 10 - 1 / 3
    else:
        return (x * j2(x) - j1(x)) / x


@vectorize("float64(float64)")
def j2_o_x(x):
    if x < 1e-3:
        x2 = x ** 2
        return x * (1 / 15 - x2 / 210 + x2 ** 2 / 7560)
    else:
        return j2(x) / x


@vectorize("float64(float64)")
def j2_o_x2(x):
    if x < 1e-3:
        x2 = x ** 2
        return 1 / 15 - x2 / 210 + x2 ** 2 / 7560
    else:
        return j2(x) / x ** 2


@vectorize("float64(float64)")
def _2j2_mxj1_o_x(x):
    if x < 1e-3:
        return -(x / 5) + x ** 3 / 42 - x ** 5 / 1080
    else:
        return (2 * j2(x) - x * j1(x)) / x


@vectorize("float64(float64)")
def _8j2_m4xj1_x2j0_o_x2(x):
    x2 = x ** 2
    if x < 0.01:
        return 1 / 5 - x2 / 14 + x2 ** 2 / 216 - x2 ** 3 / 7920
    else:
        return (8 * j2(x) - 4 * x * j1(x) + x2 * j0(x)) / x2


@vectorize("float64(float64)")
def m4j2_j1_o_x2(x):
    x2 = x ** 2
    if x < 0.01:
        return 1 / 15 - x2 / 70 + x2 ** 2 / 1512 - x2 ** 3 / 71280
    else:
        return (-4 * j2(x) + x * j1(x)) / x2


############################################################
# Correlation functions
############################################################


@lru_cache(None)
def Gamma_dd_gTH(R1, R2, dx=0):
    r"""Density (gaussian)-density (TH).

    $$
        \frac{k^2P(k)}{(2\pi)^3}
        \int_0^\pi\sin\theta\mathrm{d}\theta\int_0^{2\pi}\mathrm{d}\phi
            W_{g}(kR_1) W_{th}(k R_2) e^{ikr\cos \theta}

        = \frac{3 k P(k) e^{-\frac{1}{2} k^2 R_1^2} j_0(kr) j_1(k R_2)}{2 \pi ^2 R_2}
    $$"""
    kR1 = k * R1
    kR2 = k * R2
    kr = k * dx
    intgd = 3 * k * Pk * np.exp(-1 / 2 * kR1 ** 2) * j0(kr) * j1_o_x(kR2) / twopi2 / R2
    return np.trapz(intgd, k)


@lru_cache(None)
def Gamma_dgradd_gg(R1, R2, dx=0):
    """Density-gradient density(gaussian).

    $$
    \\left\\langle \\delta_g \nabla_x \\delta_g(\vec{r}) \right\rangle
    = -\frac{k^3 P(k) e^{-\frac{1}{2} k^2 (R_1^2+R_2^2)}}{2 \\pi ^2} j_1(kr).
    $$
    """
    if dx == 0:
        return 0
    r = dx
    intgd = (
        -(k ** 3)
        * Pk
        * np.exp(-0.5 * k ** 2 * (R1 ** 2 + R2 ** 2))
        / twopi2
        * j1(k * r)
    )
    return np.trapz(intgd, k)


@lru_cache(None)
def Gamma_dhess_gg(R1, R2, dx=0, element=None):
    r"""Density-hessian (gaussian).
    $$
    \left\langle \delta_g \nabla_{rr} \delta_g(\vec{r}) \right\rangle
        = \frac{k^3 P(k) e^{-\frac{1}{2} k^2 \left(R_1^2 + R_2^2\right)}}
               {2 \pi ^2 r} (k r j_2(kr) - j_1(kr)),\\
    \left\langle \delta_g \nabla_{\theta\theta} \delta_g(\vec{r}) \right\rangle
    = \frac{1}{2}{\left\langle \delta_g \nabla_{rr} \delta_g(\vec{r}) \right\rangle}, \
        \left\langle \delta_g \nabla_{\theta\phi} \delta_g(\vec{r}) \right\rangle
    = \left\langle \delta_g \nabla_{r\theta} \delta_g(\vec{r}) \right\rangle
    = \left\langle \delta_g \nabla_{r\phi} \delta_g(\vec{r})\right\rangle = 0.
    $$
    """
    r = dx
    kr = k * r
    if element == "r r":
        intgd = (
            k ** 4
            * Pk
            * np.exp(-0.5 * k ** 2 * (R1 ** 2 + R2 ** 2))
            / twopi2
            * xj2_mj1_o_x(kr)
        )
        return np.trapz(intgd, k)
    elif element in ("theta theta", "phi phi"):
        intgd = (
            -(k ** 4)
            * Pk
            * np.exp(-0.5 * k ** 2 * (R1 ** 2 + R2 ** 2))
            / twopi2
            * j1_o_x(kr)
        )
        return np.trapz(intgd, k)
    elif element in ("r theta", "r phi", "theta phi"):
        return 0
    else:
        raise NotImplementedError


@lru_cache(None)
def _Gamma_graddhess_gg_helper(R1, R2, dx, el):
    k2R1R2 = k ** 2 * (R1 ** 2 + R2 ** 2)
    common = k ** 5 * Pk * np.exp(-k2R1R2 / 2) / (2 * np.pi ** 2)
    if el == "r r r":
        integrand = common * _2j2_mxj1_o_x(k * dx)
        return np.trapz(integrand, k)
    elif el in ("r theta theta", "r phi phi"):
        print(el)
        integrand = -common * j2_o_x(k * dx)
        return np.trapz(integrand, k)
    elif el in (
        "r r theta",
        "r r phi",
        "r theta phi",
        "theta theta theta",
        "theta theta phi",
        "theta phi phi",
        "phi phi phi",
    ):
        return 0
    else:
        raise NotImplementedError(R1, R2, el)


def Gamma_graddhess_gg(R1, R2, dx=0, element=None):
    """Gradient-hessian (gaussian)."""
    el = " ".join(sorted(element.split(), key=lambda e: ["r", "theta", "phi"].index(e)))

    return _Gamma_graddhess_gg_helper(R1, R2, dx, el)


@lru_cache(None)
def Gamma_hesshess_gg_helper(R1, R2, dx, el):
    k2R1R2 = k ** 2 * (R1 ** 2 + R2 ** 2)
    common = k ** 6 * Pk * np.exp(-k2R1R2 / 2) / (2 * np.pi ** 2)
    kr = k * dx
    if el == "r r r r":
        integrand = common * _8j2_m4xj1_x2j0_o_x2(kr)
    elif el in ("r r theta theta", "r r phi phi"):
        integrand = common * m4j2_j1_o_x2(kr)
    elif el in ("theta theta theta theta", "phi phi phi phi"):
        integrand = common * 3 * j2_o_x2(kr)
    elif el in ("theta theta phi phi",):
        integrand = common * j2_o_x2(kr)
    else:
        return 0
    return np.trapz(integrand, k)


def Gamma_hesshess_gg(R1, R2, dx=0, element=None):
    """Hessian-hessian (gaussian)."""
    el = " ".join(sorted(element.split(), key=lambda e: ["r", "theta", "phi"].index(e)))

    return Gamma_hesshess_gg_helper(R1, R2, dx, el)
