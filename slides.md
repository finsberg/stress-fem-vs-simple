---
theme: default
paginate: true
header: 'From the Law of Laplace to FEM-based stress estimation'
footer: 'Work-seminar 30.11.23 - Henrik Finsberg'
size: 16:9
style: |
  .small-text {
    font-size: 0.55rem;
  }
html: true
marp: true
---

# From the Law of Laplace to FEM-based stress estimation

Henrik Finsberg
Work seminar
November 30th 2023


---

# Goal

How well does the law of Laplace approximate the stresses in the heart (left ventricle)

---

## What is the law of Laplace

The law of laplace states that

$$
\text{ws} = \frac{p \times r}{2 \times w}
$$

![bg right fit](figures/laplace.jpg)

---

## We can test this in a finite element model

![w:700 right](figures/sphere_stress_default.png)

Here using $r = 1$, $w = 1$ and $p = 0.4$ in which case $\text{ws} = 0.4$

---

# We can also test this law using different radii

By average the stress over the entire geometry

![w:900 center](figures/sphere_mesh_radius.png)
![bg right:35% fit](figures/sphere_radius_von_mises_only.png)

---


# With different widths

![w:900 center](figures/sphere_mesh_width.png)
![bg right:35% fit](figures/sphere_width_von_mises_only.png)


---

# Or with different pressures

![bg right:35% fit](figures/sphere_pressure_von_mises_only.png)
* But let us try to make our spherical ventricle a bit more realistic


---

## Lets try to add some fibers in our geometry

And let us first make the fibers circumferential
* We will now compute the stress in the direction of the fibers

![bg right:60% fit](figures/sphere_circ_fibers.png)

---

## Stress has nine components

We orient a cube to so that fibers are aligned with one of the axes.

![bg right:50% fit](figures/Schematic-representation-of-myocardial-fiber-orientation-and-deformation-in-3-orthogonal_W640.jpg)

![w:400 center](figures/Components_stress_tensor_cartesian.svg.png)


<p class="small-text">Taken from 10.1186/s12968-016-0258-x under [CC BY 4.0 DD](https://creativecommons.org/licenses/by/4.0/) and from https://en.wikipedia.org/wiki/Cauchy_stress_tensor#/media/File:Components_stress_tensor_cartesian.svg  under [CC BY-SA 3.0 DEED](https://creativecommons.org/licenses/by-sa/3.0/deed.en)</p>

---

![bg fit](figures/sphere_pressure_circ.png)
![bg fit](figures/sphere_radius_circ.png)
![bg fit](figures/sphere_width_circ.png)

---

![bg fit](figures/sphere_fibers.png)

---

![bg fit](figures/sphere_pressure_all.png)
![bg fit](figures/sphere_radius_all.png)
![bg fit](figures/sphere_width_all.png)

---

## The myocardium is anisotropic

In the calculations so far we have assumed that the heart is linear elastic, isotropic and undergoes small deformation (e.g steel). However, the myocardium is nonlinear, anisotropic and undergoes large deformations.

![bg right:40% fit](figures/microstructure.png)

---

<style scoped>section { justify-content: start; }</style>
## Using an anisotropic material model

![bg fit](figures/sphere_pressure_anisotropic.png)
![bg fit](figures/sphere_radius_anisotropic.png)
![bg fit](figures/sphere_width_anisotropic.png)

---

## The left ventricle is not a sphere


