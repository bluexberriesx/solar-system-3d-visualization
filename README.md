# Solar System 3D Visualization

An interactive 3D visualization of the Solar System built with Python and Matplotlib.

This project displays the eight planets orbiting the Sun using Keplerian elliptical orbits, rendered in three dimensions with orbital inclinations, planet spheres, Saturn’s rings, and a starfield background. The visualization is designed to be physically motivated while remaining visually clear.

---

## Features

- Keplerian elliptical orbits
- 3D orbital inclinations and orientations
- Nonlinear distance scaling for visibility
- Planetary spheres with reflective shading
- Emissive Sun with glow
- Saturn’s rings
- Starfield background
- Interactive rotation and zoom

---

## Notes on Accuracy and Scaling

- Orbital shapes are based on Keplerian mechanics.
- Distances are **nonlinearly scaled** to allow inner and outer planets to be visible in a single view.
- Neptune (30 AU) is used as the reference point for distance scaling.
- Planet sizes are **not to scale** and are chosen for visual clarity.
- Orbital elements are approximate and intended for visualization purposes, not precise ephemerides.

These choices are made explicitly to balance physical meaning and visual interpretability.

---

## Controls

- Mouse drag: rotate the view
- Mouse scroll wheel or `+` / `-`: zoom in and out

---

## Requirements

- Python 3
- NumPy
- Matplotlib

---

## Motivation

This project was created as a personal exploration of orbital mechanics and scientific visualization using tools commonly taught in undergraduate physics and computer science courses.

It is intended as a clear, honest visualization rather than a full physical simulation.

---

## License

This project is shared for educational and personal use.
