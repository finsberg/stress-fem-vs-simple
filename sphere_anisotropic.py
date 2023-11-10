import dolfin
import ufl_legacy as ufl
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils import laplace
import plot

import mechanics
import pulse2
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
    geo = pulse2.LVGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        ffun=geo.ffun,
        cfun=geo.cfun,
        f0=geo.fiber["90-60"]["f0"],
        s0=geo.fiber["90-60"]["s0"],
        n0=geo.fiber["90-60"]["n0"],
    )

    material_params = pulse2.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse2.HolzapfelOgden(f0=geo.f0, s0=geo.s0, parameters=material_params)

    Ta = dolfin.Constant(0.0)
    active_model = pulse2.ActiveStress(geo.f0, activation=Ta)
    comp_model = pulse2.Incompressible()

    model = pulse2.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )
    problem = pulse2.LVProblem(
        model=model, geometry=geo, parameters={"bc_type": "fix_base"}
    )

    # Compute solution

    W = dolfin.FunctionSpace(geo.mesh, "DG", 1)

    for i, pi in enumerate(pressures):
        pulse2.itertarget.itertarget(
            problem,
            target_end=pi,
            target_parameter="pressure",
            control_parameter="pressure",
            control_mode="pressure",
        )

        u, p = problem.state.split(deepcopy=True)

        F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
        J = ufl.det(F)
        P = ufl.diff(model.strain_energy(F, p), F)
        T = (1 / J) * P * F.T

        f = F * geo.f0
        von_Mises = dolfin.project(mechanics.von_mises(T), W)

        devT = T - (1 / 3) * ufl.tr(T) * ufl.Identity(3)
        fiber_stress = dolfin.project(ufl.inner(devT * f, f), W)

        with dolfin.XDMFFile(Path(output).with_suffix(".xdmf").as_posix()) as xdmf:
            xdmf.write_checkpoint(u, "u", float(i), dolfin.XDMFFile.Encoding.HDF5, True)
            xdmf.write_checkpoint(
                von_Mises, "von_Mises", float(i), dolfin.XDMFFile.Encoding.HDF5, True
            )
            xdmf.write_checkpoint(
                fiber_stress,
                "fiber_stress",
                float(i),
                dolfin.XDMFFile.Encoding.HDF5,
                True,
            )


def main():
    psize_ref = 0.1
    outdir = Path("results_sphere_anisotropic")

    for radius in radii:
        print(f"{radius = }")
        geo = get_sphere_geometry(
            radius=radius,
            width=default_width,
            psize_ref=psize_ref,
            fiber={"90-60": (90, -60)},
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
            fiber={"90-60": (90, -60)},
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
        fiber={"90-60": (90, -60)},
    )
    V = dolfin.FunctionSpace(geo.mesh, "DG", 1)
    von_Mises = dolfin.Function(V)
    Tf = dolfin.Function(V)

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
            xdmf.read_checkpoint(Tf, "fiber_stress", i)

            avg_von_mises = dolfin.assemble(von_Mises * dx) / volume
            avg_Tf = dolfin.assemble(Tf * dx) / volume

            data.append(
                {
                    "radius": radius,
                    "pressure": p,
                    "width": width,
                    "von mises": avg_von_mises,
                    "fiber_stress": avg_Tf,
                }
            )
    return data


def postprocess():
    outdir = Path("results_sphere_anisotropic")

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

    return plot.plot_stress_curves(
        df_pressure=df_pressure,
        df_width=df_width,
        df_radius=df_radius,
        y=["von mises", "fiber_stress"],
        postfix="anisotropic",
        default_radius=default_radius,
        default_width=default_width,
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


if __name__ == "__main__":
    main()
    postprocess()
