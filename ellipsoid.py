import dolfin
import ufl_legacy as ufl
import pandas as pd
import numpy as np
from pathlib import Path

import plot

import mechanics
import pulse2
from geometries import get_ellipsoid_geometry, EllipsoidRadius


dolfin.parameters["form_compiler"]["quadrature_degree"] = 6
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["optimize"] = True


pressures = (0.0, 0.1, 0.2, 0.3, 0.4)
widths = (0.1, 0.25, 0.5)
radii = (
    EllipsoidRadius(short=0.5, long=0.5),
    EllipsoidRadius(short=0.5, long=1.0),
    EllipsoidRadius(short=0.5, long=1.5),
    EllipsoidRadius(short=0.5, long=2.5),
)
default_width = 0.25
default_radius = EllipsoidRadius(short=0.5, long=1.5)


def solve(geo, pressures: tuple[float, ...], output: Path | str):
    geo = pulse2.LVGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        ffun=geo.ffun,
        cfun=geo.cfun,
        f0=geo.f0,
        s0=geo.s0,
        n0=geo.n0,
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
        model=model,
        geometry=geo,
        parameters={"bc_type": "fix_base"},
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
                von_Mises,
                "von_Mises",
                float(i),
                dolfin.XDMFFile.Encoding.HDF5,
                True,
            )
            xdmf.write_checkpoint(
                fiber_stress,
                "fiber_stress",
                float(i),
                dolfin.XDMFFile.Encoding.HDF5,
                True,
            )


def main():
    outdir = Path("results_ellipsoid")

    for radius in radii:
        print(radius)
        geo = get_ellipsoid_geometry(
            radius=radius,
            width=default_width,
        )
        output = (
            outdir
            / f"radius_long{radius.long}_radius_short{radius.short}_width{default_width}.xdmf"
        )
        if output.is_file():
            continue
        solve(
            geo=geo,
            pressures=pressures,
            output=output,
        )

    for width in widths:
        print(width)
        geo = get_ellipsoid_geometry(
            radius=default_radius,
            width=width,
        )
        output = outdir / (
            f"radius_long{default_radius.long}_"
            f"radius_short{default_radius.short}_width{width}.xdmf"
        )
        if output.is_file():
            continue
        solve(
            geo=geo,
            pressures=pressures,
            output=output,
        )


def postprocess_item(radius, width, outdir):
    geo = get_ellipsoid_geometry(
        radius=radius,
        width=width,
    )
    V = dolfin.FunctionSpace(geo.mesh, "DG", 1)
    von_Mises = dolfin.Function(V)
    Tf = dolfin.Function(V)
    dx = ufl.dx(domain=geo.mesh)
    volume = dolfin.assemble(dolfin.Constant(1.0) * dx)

    output = (
        outdir
        / f"radius_long{radius.long}_radius_short{radius.short}_width{width}.xdmf"
    )
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
                    "radius": radius.ratio,
                    "radius_long": radius.long,
                    "radius_short": radius.short,
                    "pressure": p,
                    "width": width,
                    "von mises": avg_von_mises,
                    "fiber_stress": avg_Tf,
                },
            )
    return data


def postprocess():
    outdir = Path("results_ellipsoid")

    data = []
    for radius in radii:
        print(radius)
        data.extend(postprocess_item(radius=radius, width=default_width, outdir=outdir))

    for width in widths:
        print(width)
        if width == default_width:
            # Already covered above
            continue
        data.extend(postprocess_item(radius=default_radius, width=width, outdir=outdir))

    df = pd.DataFrame(data)

    df_pressure = df[
        (df["radius"] == default_radius.ratio) & (df["width"] == default_width)
    ]
    df_width = df[(df["radius"] == default_radius.ratio) & (df["pressure"] == 0.3)]
    df_radius = df[(df["width"] == default_width) & (df["pressure"] == 0.3)]

    return plot.plot_stress_curves(
        df_pressure=df_pressure,
        df_width=df_width,
        df_radius=df_radius,
        y=["von mises", "fiber_stress"],
        postfix="anisotropic",
        default_radius=default_radius,
        default_width=default_width,
        default_pressure=0.3,
        pressures=pressures,
        widths=widths,
        radii=radii,
        pressure_labels=(f"{xi:.2f}" for xi in np.array(pressures) / max(pressures)),
        width_labels=(f"{xi:.2f}" for xi in default_radius.short / np.array(widths)),
        radius_labels=(f"{xi:.2f}" for xi in np.array([r.ratio for r in radii])),
        width_xlabel="radius short / width",
        radius_xlabel="radius ratio long : short",
        pressure_xlabel="pressure / max pressure",
        prefix="ellipsoid",
    )


if __name__ == "__main__":
    main()
    postprocess()

    plot.plot_ellipsoid_geo(
        radii=radii,
        widths=widths,
        default_width=default_width,
        default_radius=default_radius,
    )
    plot.plot_aha(
        default_width=default_width,
        default_radius=default_radius,
    )
