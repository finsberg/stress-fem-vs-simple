import dolfin
import ufl_legacy as ufl
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import mechanics
import pulse2
from geometries import EllipsoidRadius, load_geometry_from_folder
import cardiac_geometries


dolfin.parameters["form_compiler"]["quadrature_degree"] = 6
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["optimize"] = True


p_atrium = 0.2
p_ED = 1.2
p_aortic = 17.0
ESV = 70.0
default_width = 0.25
default_radius = EllipsoidRadius(short=0.5, long=1.5)


def main(geofolder):
    output = Path("results_ellipsoid") / "pv_loop.xdmf"
    output_pv_loop = output.with_suffix(".npy")

    geo_ = load_geometry_from_folder(geofolder)
    geo = pulse2.LVGeometry(
        mesh=geo_.mesh,
        markers=geo_.markers,
        ffun=geo_.ffun,
        cfun=geo_.cfun,
        f0=geo_.f0,
        s0=geo_.s0,
        n0=geo_.n0,
    )
    # Scale mesh to realistic volume
    geo.mesh.coordinates()[:] /= 3

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
    volumes = []
    pressures = []
    Tas = []

    for i, (
        target_end,
        target_parameter,
        control_parameter,
        control_mode,
        control_step,
    ) in enumerate(
        [
            (0.0, "pressure", "pressure", "pressure", 0.02),
            (p_atrium, "pressure", "pressure", "pressure", 0.02),
            (p_ED, "pressure", "pressure", "pressure", 0.02),
            (p_aortic, "pressure", "gamma", "volume", 0.01),
            (ESV, "volume", "gamma", "pressure", 0.01),
            (p_atrium, "pressure", "gamma", "volume", -0.01),
            (0.0, "gamma", "gamma", "pressure", -0.01),
        ],
    ):
        pulse2.itertarget.itertarget(
            problem,
            target_end=target_end,
            target_parameter=target_parameter,
            control_parameter=control_parameter,
            control_mode=control_mode,
            control_step=control_step,
        )
        u, p, *extra = problem.state.split(deepcopy=True)

        volumes.append(geo.inner_volume(u=u))
        pressures.append(problem.pendo)
        Tas.append(problem.get_control_parameter("gamma"))
        np.save(output_pv_loop, np.vstack([volumes, pressures, Tas]))

        F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
        J = ufl.det(F)
        P = ufl.diff(model.strain_energy(F, p), F)
        T = (1 / J) * P * F.T

        f = F * geo.f0
        von_Mises = dolfin.project(mechanics.von_mises(T), W)

        devT = T - (1 / 3) * ufl.tr(T) * ufl.Identity(3)
        fiber_stress = dolfin.project(ufl.inner(devT * f, f), W)
        det = dolfin.project(J, W)

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
            xdmf.write_checkpoint(
                det,
                "det",
                float(i),
                dolfin.XDMFFile.Encoding.HDF5,
                True,
            )


def kPa2mmHg(p):
    return p * 7.5


def postprocess(geofolder):
    output = Path("results_ellipsoid") / "pv_loop.xdmf"
    output_pv_loop = output.with_suffix(".npy")

    geo = load_geometry_from_folder(geofolder)

    V = dolfin.FunctionSpace(geo.mesh, "DG", 1)
    von_Mises = dolfin.Function(V)
    Tf = dolfin.Function(V)
    dx = ufl.dx(domain=geo.mesh, subdomain_data=geo.cfun)
    volume = dolfin.assemble(dolfin.Constant(1.0) * dx)
    volumes = {
        k: dolfin.assemble(dolfin.Constant(1.0) * dx(v[0]))
        for k, v in geo.markers.items()
        if v[1] == 3
    }

    pv_data = np.load(output_pv_loop)
    fig, ax = plt.subplots()
    ax.plot(pv_data[0, :], kPa2mmHg(pv_data[1, :]))
    ax.set_xlabel("Volume [mL]")
    ax.set_ylabel("Pressure [mmHg]")
    fig.savefig("figures/pv_loop.png", dpi=300)

    if not output.is_file():
        return []

    data = []

    def filter_zero(a, b):
        try:
            value = a / b
        except ZeroDivisionError:
            value = 0.0

        return value

    with dolfin.XDMFFile(Path(output).with_suffix(".xdmf").as_posix()) as xdmf:
        for i, p in enumerate(pv_data[0, :]):
            xdmf.read_checkpoint(von_Mises, "von_Mises", i)
            xdmf.read_checkpoint(Tf, "fiber_stress", i)

            avg_von_mises = dolfin.assemble(von_Mises * dx) / volume
            avg_Tf = dolfin.assemble(Tf * dx) / volume

            regional_von_mises = {
                f"von_mises_{'_'.join(k.lower().split('-'))}": filter_zero(
                    dolfin.assemble(von_Mises * dx(v[0])),
                    volumes[k],
                )
                for k, v in geo.markers.items()
                if v[1] == 3
            }
            regional_Tf = {
                f"fiber_stress_{'_'.join(k.lower().split('-'))}": filter_zero(
                    dolfin.assemble(Tf * dx(v[0])),
                    volumes[k],
                )
                for k, v in geo.markers.items()
                if v[1] == 3
            }
            item = {
                "pressure": p,
                "von mises": avg_von_mises,
                "fiber_stress": avg_Tf,
            }
            item.update(regional_Tf)
            item.update(regional_von_mises)
            data.append(item)

        df = pd.DataFrame(data)

        regions = [
            k.replace("von_mises_", "") for k in df.keys() if k.startswith("von_mises_")
        ]

        fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
        lines = []

        x = np.arange(pv_data.shape[1])
        for r in regions:
            ax[0].plot(x, df[f"von_mises_{r}"].to_numpy())
            (l,) = ax[1].plot(x, df[f"fiber_stress_{r}"].to_numpy())
            lines.append(l)

        ax[0].plot(x, df["von mises"].to_numpy(), "k--")
        (l,) = ax[1].plot(x, df["fiber_stress"].to_numpy(), "k--")
        lines.append(l)

        for i, axi in enumerate(ax):
            axi.set_yticks([])

        ax[0].set_ylabel("Von Mises Stress")
        ax[1].set_ylabel("Fiber Stress")
        ax[2].set_ylabel("Pressure [mmHg]")
        ax[3].set_ylabel("Active stress")
        ax[1].set_xticks(x)

        ax[2].plot(x, kPa2mmHg(pv_data[1, :]))
        ax[3].plot(x, pv_data[2, :])

        lgd = fig.legend(
            lines,
            regions + ["global"],
            loc="center right",
        )
        fig.subplots_adjust(right=0.75)
        fig.savefig(
            "figures/regional_stresses_contract",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.close("all")


if __name__ == "__main__":
    geofolder = Path("lv")
    if not geofolder.is_dir():
        cardiac_geometries.create_lv_ellipsoid(
            geofolder,
            create_fibers=True,
        )
    main(geofolder=geofolder)
    postprocess(geofolder=geofolder)
