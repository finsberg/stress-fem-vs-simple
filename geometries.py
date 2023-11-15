from pathlib import Path
from typing import NamedTuple
import cardiac_geometries
import ldrb
import dolfin
import math


class EllipsoidRadius(NamedTuple):
    short: float
    long: float

    @property
    def ratio(self) -> int:
        return int(self.long / self.short)

    def __str__(self) -> str:
        return f"{self.ratio}:1"


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


def get_ellipsoid_geometry(
    radius: EllipsoidRadius,
    width: float,
    psize_ref=0.1,
):
    folder = (
        Path("ellipsoids")
        / f"radius_long{radius.long}_radius_short{radius.short}_width{width}_psize{psize_ref}"
    )

    if not folder.is_dir():
        # Create geometry
        cardiac_geometries.create_lv_ellipsoid(
            folder,
            r_short_endo=radius.short,
            r_short_epi=radius.short + width,
            r_long_endo=radius.long,
            r_long_epi=radius.long + width,
            psize_ref=psize_ref,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.pi / 2,
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.pi / 2,
            create_fibers=True,
            fiber_space="DG_1",
            fiber_angle_endo=90,
            fiber_angle_epi=-60,
        )
    schema = cardiac_geometries.geometry.Geometry.default_schema()
    cfun = schema["cfun"]._asdict()
    cfun["fname"] = "cfun.xdmf"
    schema["cfun"] = cardiac_geometries.geometry.H5Path(**cfun)
    return cardiac_geometries.geometry.Geometry.from_folder(folder, schema=schema)


if __name__ == "__main__":
    # geo = get_sphere_geometry(radius=1.0, width=0.5, fiber=(0, 0))
    # breakpoint()
    get_ellipsoid_geometry(radius_long=1.5, radius_short=0.5, width=0.3, psize_ref=0.2)
