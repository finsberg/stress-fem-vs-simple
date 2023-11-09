from pathlib import Path
import cardiac_geometries
import ldrb
import dolfin
import math


def get_sphere_geometry(
    radius, width, psize_ref=0.1, fiber: dict[str, tuple[int, int]] | None = None
):
    folder = Path("spheres") / f"radius{radius}_width{width}_psize{psize_ref}"

    if not folder.is_dir():
        # Create geometry
        cardiac_geometries.create_lv_ellipsoid(
            folder,
            r_short_endo=radius,
            r_short_epi=radius + width,
            r_long_endo=radius,
            r_long_epi=radius + width,
            psize_ref=psize_ref,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.pi / 2,
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.pi / 2,
        )

    geo = cardiac_geometries.geometry.Geometry.from_folder(folder)
    if fiber is None:
        return geo

    geo.fiber = {}

    for k, v in fiber.items():
        fiber_path = folder / f"fiber_{k}.xdmf"

        if not fiber_path.is_file():
            system = ldrb.dolfin_ldrb(
                mesh=geo.mesh,
                ffun=geo.ffun,
                fiber_space="DG_1",
                markers={
                    "base": geo.markers["BASE"][0],
                    "lv": geo.markers["ENDO"][0],
                    "epi": geo.markers["EPI"][0],
                },
                alpha_endo_lv=v[0],
                alpha_epi_lv=v[1],
            )

            for name in ["fiber", "sheet", "sheet_normal"]:
                with dolfin.XDMFFile(fiber_path.as_posix()) as xdmf:
                    xdmf.write_checkpoint(
                        getattr(system, name),
                        name,
                        0.0,
                        dolfin.XDMFFile.Encoding.HDF5,
                        True,
                    )

        V = dolfin.VectorFunctionSpace(geo.mesh, "DG", 1)
        geo.fiber[k] = {
            "f0": dolfin.Function(V),
            "s0": dolfin.Function(V),
            "n0": dolfin.Function(V),
        }
        for f, name in [
            (geo.fiber[k]["f0"], "fiber"),
            (geo.fiber[k]["s0"], "sheet"),
            (geo.fiber[k]["n0"], "sheet_normal"),
        ]:
            with dolfin.XDMFFile(fiber_path.as_posix()) as xdmf:
                xdmf.read_checkpoint(f, name, 0)
    return geo


if __name__ == "__main__":
    geo = get_sphere_geometry(radius=1.0, width=0.5, fiber=(0, 0))
    breakpoint()
