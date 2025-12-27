"""
Solar System 3D Visualization

A static, interactive 3D visualization of the Solar System built with Python and Matplotlib.

The model uses Keplerian elliptical orbits rendered in three dimensions with orbital
inclinations and orientations. Distances are nonlinearly scaled (using a power-law) to
allow inner and outer planets to be visible in a single view, while planet sizes are
chosen for visual clarity rather than physical scale.

Key features:
- Keplerian orbits with eccentricity
- 3D orbital inclinations and rotations
- Nonlinear distance scaling (Neptune as reference)
- Planetary spheres with reflective shading
- Emissive Sun with glow
- Saturn's rings
- Starfield background

This project is intended as a physically motivated visualization, not a full dynamical
simulation or ephemeris-accurate model.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("dark_background")

def enable_scroll_zoom(ax, base_scale=1.2):
    def on_scroll(event):
        if event.inaxes != ax:
            return
        
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        zmid = (zlim[0] + zlim[1]) / 2

        xrad = (xlim[1] - xlim[0]) / 2
        yrad = (ylim[1] - ylim[0]) / 2
        zrad = (zlim[1] - zlim[0]) / 2

        if event.button == "up":
            scale = 1 / base_scale
        elif event.button == "down":
            scale = base_scale
        else:
            return
        
        ax.set_xlim3d(xmid - xrad * scale, xmid + xrad * scale)
        ax.set_ylim3d(ymid - yrad * scale, ymid + yrad * scale)
        ax.set_zlim3d(zmid - zrad * scale, zmid + zrad * scale)

        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("scroll_event", on_scroll)

def enable_keyboard_zoom(ax, base_scale=1.2):
    def on_key(event):
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        zmid = (zlim[0] + zlim[1]) / 2

        xrad = (xlim[1] - xlim[0]) / 2
        yrad = (ylim[1] - ylim[0]) / 2
        zrad = (zlim[1] - zlim[0]) / 2

        key = (event.key or "").lower()

        if key in ["+"]:
            scale = 1 / base_scale
        elif key in ["-"]:
            scale = base_scale
        else:
            return

        ax.set_xlim3d(xmid - xrad * scale, xmid + xrad * scale)
        ax.set_ylim3d(ymid - yrad * scale, ymid + yrad * scale)
        ax.set_zlim3d(zmid - zrad * scale, zmid + zrad * scale)

        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("key_press_event", on_key)

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")

enable_scroll_zoom(ax)
enable_keyboard_zoom(ax)

ax.grid(False)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((0, 0, 0, 0))
    axis.pane.set_edgecolor((0, 0, 0, 0))

ax.xaxis.line.set_color((0, 0, 0, 0))
ax.yaxis.line.set_color((0, 0, 0, 0))
ax.zaxis.line.set_color((0, 0, 0, 0))

gamma = 0.6
L = 100

light = plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45)

INC_EXAG = 10.0

STAR_RADIUS = 3 * L
N_STARS = 1200

phi = np.random.uniform(0, 2*np.pi, N_STARS)
costheta = np.random.uniform(-1, 1, N_STARS)
theta = np.arccos(costheta)

r = STAR_RADIUS * np.random.uniform(0.95, 1.05, N_STARS)

xs = r * np.sin(theta) * np.cos(phi)
ys = r * np.sin(theta) *np.sin(phi)
zs = r * np.cos(theta)

NEPTUNE_AU  = 30.0
C = L / (NEPTUNE_AU ** gamma)

def scale_distance(r_au):
    return C * (r_au ** gamma)

def rot_x(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rot_z(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def orbit_curve_3d(a_au, e, i_deg, Omega_deg, omega_deg, n_points=800):
    nu = np.linspace(0, 2*np.pi, n_points)

    r_au = (a_au * (1 - e**2)) / (1 + e * np.cos(nu))
    r_plot = scale_distance(r_au)

    x = r_plot * np.cos(nu)
    y = r_plot * np.sin(nu)
    z = np.zeros_like(x)

    i = np.deg2rad(i_deg)
    Omega = np.deg2rad(Omega_deg)
    omega = np.deg2rad(omega_deg)

    R = rot_z(Omega) @ rot_x(i) @ rot_z(omega)

    pts = np.vstack([x, y, z])
    pts3 = R @ pts

    return pts3[0], pts3[1], pts3[2]

def planet_point_3d(a_au, e, i_deg, Omega_deg, omega_deg, nu_deg):
    nu = np.deg2rad(nu_deg)

    r_au = (a_au * (1 - e**2)) / (1+ e * np.cos(nu))
    r_plot = scale_distance(r_au)

    px0 = r_plot * np.cos(nu)
    py0 = r_plot * np.sin(nu)
    pz0 = 0.0

    R = rot_z(np.deg2rad(Omega_deg)) @ rot_x(np.deg2rad(i_deg)) @ rot_z(np.deg2rad(omega_deg))
    px, py, pz = R @ np.array([px0, py0, pz0])

    return px, py, pz

def draw_rings(ax, center_xyz, R, r_inner, r_outer, n_points=300):
    t = np.linspace(0, 2*np.pi, n_points)

    x_out = r_outer * np.cos(t)
    y_out = r_outer * np.sin(t)
    z_out = np.zeros_like(x_out)

    x_in = r_inner * np.cos(t)
    y_in = r_inner * np.sin(t)
    z_in = np.zeros_like(x_in)

    outer = R @ np.vstack([x_out, y_out, z_out])
    inner = R @ np.vstack([x_in, y_in, z_in])

    cx, cy, cz = center_xyz
    outer[0] += cx; outer[1] += cy; outer[2] += cz
    inner[0] += cx; inner[1] += cy; inner[2] += cz

    ax.plot(outer[0], outer[1], outer[2], color="#7a7a7a", linewidth=1.2, alpha=0.85)
    ax.plot(inner[0], inner[1], inner[2], color="#7a7a7a", linewidth=1.0, alpha=0.55)

def draw_sphere(ax, center_xyz, radius, color, n_lat=18, n_lon=24, alpha=1.0, shaded=True):
    cx, cy, cz = center_xyz

    u = np.linspace(0, 2*np.pi, n_lon)
    v = np.linspace(0, np.pi, n_lat)

    x = cx + radius * np.outer(np.cos(u), np.sin(v))
    y = cy + radius * np.outer(np.sin(u), np.sin(v))
    z = cz + radius * np.outer(np.ones_like(u), np.cos(v))

    base = np.array(plt.matplotlib.colors.to_rgb(color))

    if shaded:
        nx = (x - cx) / radius
        ny = (y - cy) / radius
        nz = (z - cz) / radius

        az = np.deg2rad(light.azdeg)
        alt = np.deg2rad(light.altdeg)
        lx = np.cos(alt) * np.cos(az)
        ly = np.cos(alt) * np.sin(az)
        lz = np.sin(alt)

        intensity = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)

        ambient = 0.25
        diffuse = 0.85
        shade_val = ambient + diffuse * intensity
        shaded_rgb = np.clip(base * shade_val[..., None], 0, 1)
    else:
        shaded_rgb = np.ones(x.shape + (3,)) * base

    facecolors = np.dstack([shaded_rgb, np.full(shaded_rgb.shape[:2], alpha)])

    ax.plot_surface(
        x, y, z,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=10,
    )

planets = [
    {"name": "Mercury", "a": 0.387, "e": 0.206, "i": 7.0,   "Omega": 48.3,  "omega": 29.1,  "color": "#b1b1b1", "size": 40, "radius": 0.4, "nu": 20},
    {"name": "Venus",   "a": 0.723, "e": 0.007, "i": 3.4,   "Omega": 76.7,  "omega": 54.9,  "color": "#e6c27a", "size": 90, "radius": 1.2, "nu": 110},
    {"name": "Earth",   "a": 1.000, "e": 0.017, "i": 0.0,   "Omega": 0.0,   "omega": 0.0,   "color": "#2a5caa", "size": 100, "radius": 1.3, "nu": 150},
    {"name": "Mars",    "a": 1.524, "e": 0.093, "i": 1.85,  "Omega": 49.6,  "omega": 286.5, "color": "#c45a3a", "size": 70, "radius": 0.9, "nu": 250},
    {"name": "Jupiter", "a": 5.203, "e": 0.049, "i": 1.30,  "Omega": 100.6, "omega": 273.9, "color": "#d8b38a", "size": 240, "radius": 3.6, "nu": 70},
    {"name": "Saturn",  "a": 9.537, "e": 0.056, "i": 2.49,  "Omega": 113.7, "omega": 339.4, "color": "#d9cf9b", "size": 210, "radius": 3.2, "nu": 10},
    {"name": "Uranus",  "a": 19.191,"e": 0.047, "i": 0.77,  "Omega": 74.0,  "omega": 96.7,  "color": "#8ad1d1", "size": 150, "radius": 2.2, "nu": 190},
    {"name": "Neptune", "a": 30.068,"e": 0.009, "i": 1.77,  "Omega": 131.8, "omega": 273.2, "color": "#3b5bdc", "size": 145, "radius": 2.1, "nu": 310},
]

for p in planets:
    ox, oy, oz = orbit_curve_3d(p["a"], p["e"], p["i"] * INC_EXAG , p["Omega"], p["omega"])
    ax.plot(ox, oy, oz, color="#7a7a7a", linewidth=1.0, alpha=0.6)

    px, py, pz = planet_point_3d(p["a"], p["e"], p["i"] * INC_EXAG , p["Omega"], p["omega"], p["nu"])

    if p["name"] == "Saturn":
        R_sat = rot_z(np.deg2rad(p["Omega"])) @ rot_x(np.deg2rad(p["i"] * INC_EXAG )) @ rot_z(np.deg2rad(p["omega"]))

        r_outer = 7.5
        r_inner = 5.2

        draw_rings(ax, (px, py, pz), R_sat, r_inner, r_outer)

    draw_sphere(ax, (px, py, pz), p["radius"], p["color"], alpha=1.0)
    ax.text(px, py, pz, f" {p['name']}", fontsize=8)

SUN_R = 3.0

draw_sphere(ax, (0, 0, 0), radius=SUN_R, color="#ffcc33", alpha=1.0, shaded=False)

for k, a in [(1.35, 0.12), (1.75, 0.06), (2.25, 0.03)]:
    draw_sphere(ax, (0, 0, 0), radius=SUN_R * k, color="#ffcc33", alpha=a, shaded=False)

ax.text(0, 0, 0, " Sun", fontsize=9)

ax.scatter(xs, ys, zs, s=1, color="white", alpha=0.6, depthshade=False)

ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-L, L)

ax.set_box_aspect((1, 1, 1))

ax.set_title("Solar System Model")

scale_text = (
    "Distance scale:\n"
    "Neptune (30 AU) → 100 units\n"
    "r_plot ∝ r_AU^γ,  γ = 0.6\n"
    "Planet sizes not to scale"
)

ax.text2D(
    0.02, 0.02,
    scale_text,
    transform=ax.transAxes,
    fontsize=9,
    color="white",
    alpha=0.85
)
plt.show()