from typing import Literal, Sequence
import dolfin
import numpy as np
import matplotlib.pyplot as plt


from geometries import EllipsoidRadius, get_ellipsoid_geometry, get_sphere_geometry
from utils import laplace


def remove_3d_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_sphere_geo(
    radii,
    widths,
    default_width,
    default_radius,
    psize_ref,
):
    fig = plt.figure(figsize=(10, 6))

    geo = get_sphere_geometry(
        radius=default_radius, width=default_width, psize_ref=psize_ref
    )
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_xlim(-2.0, 0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)
    remove_3d_ticks(ax)

    dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
    ax.azim = 0
    ax.dist = 7
    ax.elev = -20

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_xlim(-2.0, 0)
    ax.set_ylim(-1.0, 2.5)
    ax.set_zlim(-2.0, 1.5)
    remove_3d_ticks(ax)

    dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
    ax.azim = -60
    ax.dist = 7
    ax.elev = 30
    fig.savefig("figures/sphere_mesh_default.png", dpi=300)

    fig = plt.figure(figsize=(12, 4))
    for i, radius in enumerate(radii, start=1):
        geo = get_sphere_geometry(
            radius=radius, width=default_width, psize_ref=psize_ref
        )
        ax = fig.add_subplot(1, len(radii), i, projection="3d")
        ax.set_xlim(-3.0, 0)
        ax.set_ylim(-2.5, 2.5)
        ax.set_zlim(-2.5, 2.5)
        remove_3d_ticks(ax)

        dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
        ax.set_title(f"$r : 1 = {radius / default_radius:.2f}$")
        ax.azim = 0
        ax.dist = 7
        ax.elev = -20
        # if i != len(radii):
        #     ax.set_xticks([])
        # if i != 1:
        #     ax.set_zticks([])

    fig.savefig("figures/sphere_mesh_radius.png", dpi=300)

    fig = plt.figure(figsize=(12, 4))
    for i, width in enumerate(widths, start=1):
        geo = get_sphere_geometry(
            radius=default_radius, width=width, psize_ref=psize_ref
        )
        ax = fig.add_subplot(1, len(radii), i, projection="3d")
        ax.set_xlim(-3.0, 0)
        ax.set_ylim(-2.5, 2.5)
        ax.set_zlim(-2.5, 2.5)
        remove_3d_ticks(ax)

        dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
        ax.set_title(f"$r:w = {int(default_radius / width)}:1$")
        ax.azim = 0
        ax.dist = 7
        ax.elev = -20
        # if i != len(radii):
        #     ax.set_xticks([])
        # if i != 1:
        #     ax.set_zticks([])

    fig.savefig("figures/sphere_mesh_width.png", dpi=300)


def plot_ellipsoid_geo(
    radii,
    widths,
    default_width,
    default_radius,
    psize_ref,
):
    fig = plt.figure(figsize=(10, 6))

    geo = get_ellipsoid_geometry(
        radius=default_radius, width=default_width, psize_ref=psize_ref
    )
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_xlim(-2.0, 0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    remove_3d_ticks(ax)

    dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
    ax.azim = 0
    ax.dist = 7
    ax.elev = -20

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_xlim(-2.0, 0)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-1.5, 0.5)
    remove_3d_ticks(ax)

    dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
    ax.azim = -60
    ax.dist = 7
    ax.elev = 30
    fig.savefig("figures/ellipsoid_mesh_default.png", dpi=300)

    fig = plt.figure(figsize=(12, 4))
    for i, radius in enumerate(radii, start=1):
        geo = get_ellipsoid_geometry(
            radius=radius, width=default_width, psize_ref=psize_ref
        )
        ax = fig.add_subplot(1, len(radii), i, projection="3d")
        ax.set_xlim(-1.5, 0)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
        ax.set_title(f"$long:short = {radius}$")
        ax.azim = 0
        ax.dist = 7
        ax.elev = -20
        remove_3d_ticks(ax)
        # if i != len(radii):
        #     ax.set_xticks([])
        # if i != 1:
        #     ax.set_zticks([])

    fig.savefig("figures/ellipsoid_mesh_radius.png", dpi=300)

    fig = plt.figure(figsize=(12, 4))
    for i, width in enumerate(widths, start=1):
        geo = get_ellipsoid_geometry(
            radius=default_radius, width=width, psize_ref=psize_ref
        )
        ax = fig.add_subplot(1, len(widths), i, projection="3d")
        ax.set_xlim(-1.5, 0)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
        ax.set_title(f"$short:w = {int(default_radius.short / width)}:1$")
        ax.azim = 0
        ax.dist = 7
        ax.elev = -20
        remove_3d_ticks(ax)
        # if i != len(widths):
        #     ax.set_xticks([])
        # if i != 1:
        #     ax.set_zticks([])

    fig.savefig("figures/ellipsoid_mesh_width.png", dpi=300)


def plot_aha(default_radius, default_width, psize_ref):
    import fenics_plotly

    fenics_plotly.set_renderer("png")

    geo = get_ellipsoid_geometry(
        radius=default_radius, width=default_width, psize_ref=psize_ref
    )

    fig = fenics_plotly.plot(geo.cfun)
    fig.figure.write_image("figures/ellipsoid_mesh_aha.png")


def plot_laplace(
    ax,
    width: float | Sequence[float],
    radius: float | Sequence[float] | EllipsoidRadius | Sequence[EllipsoidRadius],
    pressure: float | Sequence[float],
    by: Literal["width", "radius", "pressure"],
    factor: float = 2.0,
    color="k",
) -> None:
    # Check a few special cases first
    extra_label = ""
    if not np.isclose(factor, 2.0):
        extra_label = f" (factor={factor})"

    if by == "radius":
        if len(radius) > 0 and isinstance(radius[0], EllipsoidRadius):
            ax.plot(
                [r.ratio for r in radius],
                laplace(
                    pressure=pressure,
                    radius=[r.long for r in radius],
                    width=width,
                    factor=factor,
                ),
                color=color,
                linestyle="--",
                label="Laplace (radius long)" + extra_label,
            )
            ax.plot(
                [r.ratio for r in radius],
                laplace(
                    pressure=pressure,
                    radius=[r.short for r in radius],
                    width=width,
                    factor=factor,
                ),
                color=color,
                linestyle=":",
                label="Laplace (radius short)" + extra_label,
            )
            return
        else:
            x = radius
    elif by == "width":
        x = width
    elif by == "pressure":
        x = pressure
    else:
        raise ValueError(f"Unknown x value {by}")

    if isinstance(radius, EllipsoidRadius):
        ax.plot(
            x,
            laplace(
                pressure=pressure,
                radius=radius.long,
                width=width,
                factor=factor,
            ),
            color=color,
            linestyle="--",
            label="Laplace (radius long)" + extra_label,
        )
        ax.plot(
            x,
            laplace(
                pressure=pressure,
                radius=radius.short,
                width=width,
                factor=factor,
            ),
            color=color,
            linestyle=":",
            label="Laplace (radius short)" + extra_label,
        )
        return

    ax.plot(
        x,
        laplace(
            pressure=pressure,
            radius=radius,
            width=width,
            factor=factor,
        ),
        color=color,
        linestyle="--",
        label="Laplace" + extra_label,
    )


def plot_stress_curves(
    y,
    default_width,
    default_radius,
    default_pressure,
    df_pressure,
    df_width,
    df_radius,
    postfix,
    pressures,
    pressure_labels,
    widths,
    width_labels,
    radii,
    radius_labels,
    prefix,
    width_xlabel="width",
    radius_xlabel="radius",
    pressure_xlabel="pressure",
    extra_factor: float | None = None,
):
    ax = df_pressure.sort_values(by="pressure").plot(
        x="pressure",
        y=y,
    )
    plot_laplace(
        ax,
        pressure=pressures,
        radius=default_radius,
        width=default_width,
        by="pressure",
    )

    if extra_factor is not None:
        plot_laplace(
            ax,
            pressure=pressures,
            radius=default_radius,
            width=default_width,
            by="pressure",
            factor=extra_factor,
            color="gray",
        )

    ax.legend()
    ax.set_yticks([])
    ax.set_ylabel("Stress")
    ax.set_xticks(pressures)
    ax.set_xticklabels(pressure_labels)
    ax.set_xlabel(pressure_xlabel)
    # ax.set_title(f"$r={default_radius}$, $w={default_width}$")
    plt.savefig(f"figures/{prefix}_pressure_{postfix}.png")

    ax = df_width.sort_values(by="width").plot(
        x="width",
        y=y,
        marker="o",
    )
    plot_laplace(
        ax,
        pressure=default_pressure,
        radius=default_radius,
        width=widths,
        by="width",
    )

    if extra_factor is not None:
        plot_laplace(
            ax,
            pressure=default_pressure,
            radius=default_radius,
            width=widths,
            by="width",
            factor=extra_factor,
            color="gray",
        )

    ax.legend()
    ax.set_yticks([])
    ax.set_xticks(widths)
    ax.set_xticklabels(width_labels)
    ax.set_xlabel(width_xlabel)
    ax.set_ylabel("Stress")
    # ax.set_title(f"$r={default_radius}$, $p=0.3$")

    plt.savefig(f"figures/{prefix}_width_{postfix}.png")

    ax = df_radius.sort_values(by="radius").plot(
        x="radius",
        y=y,
    )

    plot_laplace(
        ax,
        pressure=default_pressure,
        radius=radii,
        width=default_width,
        by="radius",
    )
    if extra_factor is not None:
        plot_laplace(
            ax,
            pressure=default_pressure,
            radius=radii,
            width=default_width,
            by="radius",
            factor=extra_factor,
            color="gray",
        )

    if isinstance(radii[0], EllipsoidRadius):
        ax.set_xticks([r.ratio for r in radii])
    else:
        ax.set_xticks(radii)
    ax.set_xticklabels(radius_labels)
    ax.legend()
    ax.set_yticks([])
    ax.set_ylabel("Stress")
    ax.set_xlabel(radius_xlabel)
    # ax.set_title(f"$w={default_width}$, $p=0.3$")
    plt.savefig(f"figures/{prefix}_radius_{postfix}.png")
