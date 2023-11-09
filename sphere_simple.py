import dolfin
import ufl_legacy as ufl
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils import laplace
import mechanics
from geometries import get_sphere_geometry


dolfin.parameters["form_compiler"]["quadrature_degree"] = 6
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["optimize"] = True


pressures = (0.0, 0.1, 0.2, 0.3, 0.4)
widths = (0.1, 0.2, 0.3, 0.5)
radii = (0.3, 0.5, 1.0, 1.5)
default_width = 0.5
default_radius = 1.0
psize_ref = 0.1


def solve(geo, pressures: tuple[float, ...], output: Path | str):
    # Scaled variables
    mu = 1
    lmbda = 1.0

    # Create mesh and define function space
    V = dolfin.VectorFunctionSpace(geo.mesh, "P", 1)
    # Fix base
    bcs = dolfin.DirichletBC(
        V, dolfin.Constant((0, 0, 0)), geo.ffun, geo.markers["BASE"][0]
    )

    # Define strain and stress
    def epsilon(u):
        return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(d) + 2 * mu * epsilon(u)

    # Define variational problem
    u = dolfin.TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = ufl.TestFunction(V)

    F = ufl.grad(u) + ufl.Identity(d)
    N = ufl.FacetNormal(geo.mesh)
    pressure = dolfin.Constant(0.0)
    # n = pressure * ufl.cofac(F) * N
    # ds = ufl.ds(subdomain_data=geo.ffun)
    # endo = geo.markers["ENDO"][0]

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(-pressure * N, v) * ufl.ds(
        subdomain_data=geo.ffun, subdomain_id=geo.markers["ENDO"][0]
    )

    # Compute solution
    u = dolfin.Function(V)
    W = dolfin.FunctionSpace(geo.mesh, "DG", 0)
    for i, p in enumerate(pressures):
        pressure.assign(p)

        dolfin.solve(a == L, u, bcs)

        # Plot stress
        s = sigma(u) - (1.0 / 3) * ufl.tr(sigma(u)) * ufl.Identity(
            d
        )  # deviatoric stress
        von_Mises = dolfin.project(ufl.sqrt(3.0 / 2 * ufl.inner(s, s)), W)

        F = ufl.grad(u) + ufl.Identity(d)
        f0 = F * geo.fiber["0-0"]["f0"]
        f1 = F * geo.fiber["60-60"]["f0"]
        f2 = F * geo.fiber["90-60"]["f0"]

        circ = dolfin.project(ufl.inner(s * f0, f0), W)
        f6060 = dolfin.project(ufl.inner(s * f1, f1), W)
        f9060 = dolfin.project(ufl.inner(s * f2, f2), W)
        with dolfin.XDMFFile(Path(output).with_suffix(".xdmf").as_posix()) as xdmf:
            xdmf.write_checkpoint(u, "u", float(i), dolfin.XDMFFile.Encoding.HDF5, True)
            xdmf.write_checkpoint(
                von_Mises, "von_Mises", float(i), dolfin.XDMFFile.Encoding.HDF5, True
            )
            xdmf.write_checkpoint(
                circ, "circ", float(i), dolfin.XDMFFile.Encoding.HDF5, True
            )
            xdmf.write_checkpoint(
                f6060, "f6060", float(i), dolfin.XDMFFile.Encoding.HDF5, True
            )
            xdmf.write_checkpoint(
                f9060, "f9060", float(i), dolfin.XDMFFile.Encoding.HDF5, True
            )


def main():
    psize_ref = 0.1
    outdir = Path("results_sphere_simple")

    for radius in radii:
        print(f"{radius = }")
        geo = get_sphere_geometry(
            radius=radius,
            width=default_width,
            psize_ref=psize_ref,
            fiber={"0-0": (0, 0), "60-60": (60, -60), "90-60": (90, -60)},
        )
        output = outdir / f"radius{radius}_width{default_width}_psize{psize_ref}.xdmf"
        if output.is_file():
            continue
        solve(
            geo=geo,
            pressures=pressures,
            output=output,
        )

    for width in widths:
        print(f"{width =}")
        geo = get_sphere_geometry(
            radius=default_radius,
            width=width,
            psize_ref=psize_ref,
            fiber={"0-0": (0, 0), "60-60": (60, -60), "90-60": (90, -60)},
        )
        output = outdir / f"radius{default_radius}_width{width}_psize{psize_ref}.xdmf"
        if output.is_file():
            continue
        solve(
            geo=geo,
            pressures=pressures,
            output=output,
        )


def postprocess_item(radius, width, outdir, psize_ref=psize_ref):
    geo = get_sphere_geometry(
        radius=radius,
        width=width,
        psize_ref=psize_ref,
        fiber={"0-0": (0, 0), "60-60": (60, -60), "90-60": (90, -60)},
    )
    V = dolfin.FunctionSpace(geo.mesh, "DG", 0)
    von_Mises = dolfin.Function(V)
    circ = dolfin.Function(V)
    f6060 = dolfin.Function(V)
    f9060 = dolfin.Function(V)
    # dsendo = ufl.ds(
    #     subdomain_data=geo.ffun,
    #     domain=geo.mesh,
    #     subdomain_id=geo.markers["ENDO"][0],
    # )
    # dsepi = ufl.ds(
    #     subdomain_data=geo.ffun, domain=geo.mesh, subdomain_id=geo.markers["EPI"][0]
    # )
    dx = ufl.dx(domain=geo.mesh)

    # endoarea = dolfin.assemble(dolfin.Constant(1.0) * dsendo)
    # epiarea = dolfin.assemble(dolfin.Constant(1.0) * dsepi)
    volume = dolfin.assemble(dolfin.Constant(1.0) * dx)

    output = outdir / f"radius{radius}_width{width}_psize{psize_ref}.xdmf"
    if not output.is_file():
        return []

    data = []
    with dolfin.XDMFFile(Path(output).with_suffix(".xdmf").as_posix()) as xdmf:
        for i, p in enumerate(pressures):
            xdmf.read_checkpoint(von_Mises, "von_Mises", i)
            xdmf.read_checkpoint(circ, "circ", i)
            xdmf.read_checkpoint(f6060, "f6060", i)
            xdmf.read_checkpoint(f9060, "f9060", i)

            avg_von_mises = dolfin.assemble(von_Mises * dx) / volume
            avg_circ = dolfin.assemble(circ * dx) / volume
            avg_f6060 = dolfin.assemble(f6060 * dx) / volume
            avg_f9060 = dolfin.assemble(f9060 * dx) / volume
            # endo = dolfin.assemble(von_Mises * dsendo) / endoarea
            # epi = dolfin.assemble(von_Mises * dsepi) / epiarea
            data.append(
                {
                    "radius": radius,
                    "pressure": p,
                    "width": width,
                    "von mises": avg_von_mises,
                    "circumferential": avg_circ,
                    "(60, -60)": avg_f6060,
                    "(90, -60)": avg_f9060,
                }
            )
    return data


def plot_geo():
    fig = plt.figure(figsize=(10, 6))

    geo = get_sphere_geometry(
        radius=default_radius, width=default_width, psize_ref=psize_ref
    )
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_xlim(-2.0, 0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)

    dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
    ax.azim = 0
    ax.dist = 7
    ax.elev = -20

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_xlim(-2.0, 0)
    ax.set_ylim(-1.0, 2.5)
    ax.set_zlim(-2.0, 1.5)

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
        ax.set_ylim(-3.0, 3.0)
        ax.set_zlim(-3.0, 3.0)

        dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
        ax.set_title(f"$r = {radius}$")
        ax.azim = 0
        ax.dist = 7
        ax.elev = -20
        if i > 0:
            ax.set_yticks([])

    fig.savefig("figures/sphere_mesh_radius.png", dpi=300)

    fig = plt.figure(figsize=(12, 4))
    for i, width in enumerate(widths, start=1):
        geo = get_sphere_geometry(
            radius=default_radius, width=width, psize_ref=psize_ref
        )
        ax = fig.add_subplot(1, len(radii), i, projection="3d")
        ax.set_xlim(-3.0, 0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_zlim(-3.0, 3.0)

        dolfin.common.plotting.mplot_mesh(ax=ax, mesh=geo.mesh)
        ax.set_title(f"$w = {width}$")
        ax.azim = 0
        ax.dist = 7
        ax.elev = -20

    fig.savefig("figures/sphere_mesh_width.png", dpi=300)


def _plot(
    y, df_pressure, df_width, df_radius, postfix, extra_factor: float | None = None
):
    ax = df_pressure.sort_values(by="pressure").plot(
        x="pressure",
        y=y,
    )
    ax.plot(
        pressures,
        laplace(
            pressure=np.array(pressures), radius=default_radius, width=default_width
        ),
        "k--",
        label="Laplace",
    )
    if extra_factor is not None:
        ax.plot(
            pressures,
            laplace(
                pressure=np.array(pressures),
                radius=default_radius,
                width=default_width,
                factor=extra_factor,
            ),
            "k:",
            label=f"Laplace (factor={extra_factor})",
        )
    ax.legend()
    ax.set_yticks([])
    ax.set_ylabel("Stress")
    ax.set_title(f"$r={default_radius}$, $w={default_width}$")
    plt.savefig(f"figures/sphere_pressure_{postfix}.png")

    ax = df_width.sort_values(by="width").plot(
        x="width",
        y=y,
        marker="o",
    )
    ax.plot(
        widths,
        laplace(pressure=0.3, radius=default_radius, width=np.array(widths)),
        "k--",
        label="Laplace",
    )
    if extra_factor is not None:
        ax.plot(
            widths,
            laplace(
                pressure=0.3,
                radius=default_radius,
                width=np.array(widths),
                factor=extra_factor,
            ),
            "k:",
            label=f"Laplace (factor={extra_factor})",
        )
    ax.legend()
    ax.set_yticks([])
    ax.set_ylabel("Stress")
    ax.set_title(f"$r={default_radius}$, $p=0.3$")

    plt.savefig(f"figures/sphere_width_{postfix}.png")

    ax = df_radius.sort_values(by="radius").plot(
        x="radius",
        y=y,
    )
    ax.plot(
        radii,
        laplace(pressure=0.3, radius=np.array(radii), width=default_width),
        "k--",
        label="Laplace",
    )
    if extra_factor is not None:
        ax.plot(
            radii,
            laplace(
                pressure=0.3,
                radius=np.array(radii),
                width=default_width,
                factor=extra_factor,
            ),
            "k:",
            label=f"Laplace (factor={extra_factor})",
        )
    ax.legend()
    ax.set_yticks([])
    ax.set_ylabel("Stress")
    ax.set_title(f"$w={default_width}$, $p=0.3$")
    plt.savefig(f"figures/sphere_radius_{postfix}.png")


def plot_von_mises_only(df_pressure, df_width, df_radius):
    return _plot(
        df_pressure=df_pressure,
        df_width=df_width,
        df_radius=df_radius,
        y=["von mises"],
        postfix="von_mises_only",
    )


def plot_circ_and_von_mises(df_pressure, df_width, df_radius):
    return _plot(
        df_pressure=df_pressure,
        df_width=df_width,
        df_radius=df_radius,
        y=["von mises", "circumferential"],
        postfix="circ",
        extra_factor=8,
    )


def plot_all(df_pressure, df_width, df_radius):
    return _plot(
        df_pressure=df_pressure,
        df_width=df_width,
        df_radius=df_radius,
        y=["von mises", "circumferential", "(60, -60)", "(90, -60)"],
        postfix="all",
    )


def postprocess():
    outdir = Path("results_sphere_simple")

    data = []
    for radius in radii:
        print(f"{radius= }")
        data.extend(postprocess_item(radius=radius, width=default_width, outdir=outdir))

    for width in widths:
        print(f"{width = }")
        if width == default_width:
            # Already covered above
            continue
        data.extend(postprocess_item(radius=default_radius, width=width, outdir=outdir))

    df = pd.DataFrame(data)

    df_pressure = df[(df["radius"] == default_radius) & (df["width"] == default_width)]
    df_width = df[(df["radius"] == default_radius) & (df["pressure"] == 0.3)]
    df_radius = df[(df["width"] == default_width) & (df["pressure"] == 0.3)]
    plot_von_mises_only(df_pressure, df_width, df_radius)
    plot_circ_and_von_mises(df_pressure, df_width, df_radius)
    plot_all(df_pressure, df_width, df_radius)


if __name__ == "__main__":
    # main()
    postprocess()
    # plot_geo()
