import dolfin
import ufl_legacy as ufl


def subplus(x):
    r"""
    Ramp function
    .. math::
       \max\{x,0\}
    """

    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)


def heaviside(x):
    r"""
    Heaviside function
    .. math::
       \mathcal{H}(x) = \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}
    """

    return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)


def neo_hookean(F: ufl.Coefficient, mu: float = 15.0) -> ufl.Coefficient:
    r"""Neo Hookean model
    .. math::
        \Psi(F) = \frac{\mu}{2}(I_1 - 3)
    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    mu : float, optional
        Material parameter, by default 15.0
    Returns
    -------
    ufl.Coefficient
        Strain energy density
    """
    C = F.T * F
    J = ufl.det(F)
    I1 = pow(J, -2 / 3) * dolfin.tr(C)
    return 0.5 * mu * (I1 - 3)


def transverse_holzapfel_ogden(
    F: ufl.Coefficient,
    f0: dolfin.Function,
    a: float = 2.280,
    b: float = 9.726,
    a_f: float = 1.685,
    b_f: float = 15.779,
) -> ufl.Coefficient:
    r"""Transverse isotropic version of the model from Holzapfel and Ogden [1]_.
    The strain energy density function is given by
    .. math::
        \Psi(I_1, I_{4\mathbf{f}_0}, I_{4\mathbf{s}_0}, I_{8\mathbf{f}_0\mathbf{s}_0})
        = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
        + \frac{a_f}{2 b_f} \mathcal{H}(I_{4\mathbf{f}_0} - 1)
        \left( e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} -1 \right)
       
    where
    .. math::
        (x)_+ = \max\{x,0\}
    and
    .. math::
        \mathcal{H}(x) = \begin{cases}
            1, & \text{if $x > 0$} \\
            0, & \text{if $x \leq 0$}
        \end{cases}
    is the Heaviside function.
    .. [1] Holzapfel, Gerhard A., and Ray W. Ogden.
        "Constitutive modelling of passive myocardium:
        a structurally based framework for material characterization.
        "Philosophical Transactions of the Royal Society of London A:
        Mathematical, Physical and Engineering Sciences 367.1902 (2009):
        3445-3475.
    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    f0 : dolfin.Function
        Fiber direction
    a : float, optional
        Material parameter, by default 2.280
    b : float, optional
        Material parameter, by default 9.726
    a_f : float, optional
        Material parameter, by default 1.685
    b_f : float, optional
        Material parameter, by default 15.779
    Returns
    -------
    ufl.Coefficient
        Strain energy density
    """
    J = ufl.det(F)
    C = F.T * F
    I1 = pow(J, -2 / 3) * dolfin.tr(C)
    I4f = pow(J, -2 / 3) * dolfin.inner(C * f0, f0)

    return (a / (2.0 * b)) * (dolfin.exp(b * (I1 - 3)) - 1.0) + (
        a_f / (2.0 * b_f)
    ) * heaviside(I4f - 1) * (dolfin.exp(b_f * subplus(I4f - 1) ** 2) - 1.0)


def active_stress_energy(
    F: ufl.Coefficient, f0: dolfin.Function, Ta: dolfin.Constant
) -> ufl.Coefficient:
    """Active stress energy
    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    f0 : dolfin.Function
        Fiber direction
    Ta : dolfin.Constant
        Active tension
    Returns
    -------
    ufl.Coefficient
        Active stress energy
    """

    I4f = dolfin.inner(F * f0, F * f0)
    return 0.5 * Ta * (I4f - 1)


def compressibility(F: ufl.Coefficient, kappa: float = 1e3) -> ufl.Coefficient:
    r"""Penalty for compressibility
    .. math::
        \kappa (J \mathrm{ln}J - J + 1)
    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    kappa : float, optional
        Parameter for compressibility, by default 1e3
    Returns
    -------
    ufl.Coefficient
        Energy for compressibility
    """
    J = dolfin.det(F)
    return kappa * (J * dolfin.ln(J) - J + 1)


def von_mises(T: ufl.Coefficient) -> ufl.Coefficient:
    r"""Compute the von Mises stress tensor :math`\sigma_v`, with

    .. math::

        \sigma_v^2 = \frac{1}{2} \left(
            (\mathrm{T}_{11} - \mathrm{T}_{22})^2 +
            (\mathrm{T}_{22} - \mathrm{T}_{33})^2 +
            (\mathrm{T}_{33} - \mathrm{T}_{11})^2 +
        \right) - 3 \left(
            \mathrm{T}_{12} + \mathrm{T}_{23} + \mathrm{T}_{31}
        \right)

    Parameters
    ----------
    T : ufl.Coefficient
        Cauchy stress tensor

    Returns
    -------
    ufl.Coefficient
        The von Mises stress tensor
    """
    von_Mises_squared = 0.5 * (
        (T[0, 0] - T[1, 1]) ** 2 + (T[1, 1] - T[2, 2]) ** 2 + (T[2, 2] - T[0, 0]) ** 2
    ) + 3 * (T[0, 1] + T[1, 2] + T[2, 0])

    return ufl.sqrt(abs(von_Mises_squared))
