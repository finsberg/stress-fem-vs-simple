import dolfin
import ufl_legacy as ufl
import pandas as pd
import numpy as np
from pathlib import Path


import plot
from geometries import get_sphere_geometry


dolfin.parameters["form_compiler"]["quadrature_degree"] = 6
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["optimize"] = True


pressures = (0.0, 0.1, 0.2, 0.3, 0.4)
widths = (0.1, 0.2, 0.3, 0.5)
radii = (0.25, 0.5, 1.0, 1.5)
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


def plot_von_mises_only(df_pressure, df_width, df_radius):
    return plot.plot_stress_curves(
        df_pressure=df_pressure,
        df_width=df_width,
        df_radius=df_radius,
        y=["von mises"],
        postfix="von_mises_only",
        default_radius=default_radius,
        default_width=default_width,
        default_pressure=0.3,
        pressures=pressures,
        widths=widths,
        radii=radii,
        pressure_labels=(f"{xi:.2f}" for xi in np.array(pressures) / max(pressures)),
        width_labels=(f"{xi:.2f}" for xi in default_radius / np.array(widths)),
        radius_labels=(f"{xi:.2f}" for xi in np.array(radii) / default_radius),
        prefix="sphere",
        width_xlabel="radius / width",
        radius_xlabel="radius / default radius",
        pressure_xlabel="pressure / max pressure",
    )


def plot_circ_and_von_mises(df_pressure, df_width, df_radius):
    return plot.plot_stress_curves(
        df_pressure=df_pressure,
        df_width=df_width,
        df_radius=df_radius,
        y=["von mises", "circumferential"],
        postfix="circ",
        extra_factor=8,
        default_radius=default_radius,
        default_width=default_width,
        default_pressure=0.3,
        pressures=pressures,
        widths=widths,
        radii=radii,
        pressure_labels=(f"{xi:.2f}" for xi in np.array(pressures) / max(pressures)),
        width_labels=(f"{xi:.2f}" for xi in default_radius / np.array(widths)),
        radius_labels=(f"{xi:.2f}" for xi in np.array(radii) / default_radius),
        width_xlabel="radius / width",
        radius_xlabel="radius / default radius",
        pressure_xlabel="pressure / max pressure",
        prefix="sphere",
    )


def plot_all(df_pressure, df_width, df_radius):
    return plot.plot_stress_curves(
        df_pressure=df_pressure,
        df_width=df_width,
        df_radius=df_radius,
        y=["von mises", "circumferential", "(60, -60)", "(90, -60)"],
        postfix="all",
        default_radius=default_radius,
        default_width=default_width,
        default_pressure=0.3,
        pressures=pressures,
        widths=widths,
        radii=radii,
        pressure_labels=(f"{xi:.2f}" for xi in np.array(pressures) / max(pressures)),
        width_labels=(f"{xi:.2f}" for xi in default_radius / np.array(widths)),
        radius_labels=(f"{xi:.2f}" for xi in np.array(radii) / default_radius),
        width_xlabel="radius / width",
        radius_xlabel="radius / default radius",
        pressure_xlabel="pressure / max pressure",
        prefix="sphere",
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
    # postprocess()
    plot.plot_sphere_geo(
        radii=radii,
        widths=widths,
        default_width=default_width,
        default_radius=default_radius,
        psize_ref=psize_ref,
    )
