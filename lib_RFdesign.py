import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from cqmore import Workplane
import cadquery as cq
from cadquery import exporters
from pathlib import Path

class Convex:
    def __init__(self, model_path: Path):
        """
        Base class for Convex Hull operations.
        run_dir: Pathlib object from the backbone instance.
        """
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            raise ValueError("model_path cannot be None. A valid directory Path is required.")  

    def plotConvex3D(self, hull):
        """
        Visualizes the convex hull in 3D.
        Maintains the original visual logic and aspect ratio setting.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Configure 3D pane appearance
        ax.xaxis.pane.set_edgecolor('k')
        ax.yaxis.pane.set_edgecolor('k')
        ax.zaxis.pane.set_edgecolor('k')
        ax.xaxis.pane.set_facecolor("w")
        ax.yaxis.pane.set_facecolor("w")
        ax.zaxis.pane.set_facecolor("w")

        ax.grid(False)
        ax.view_init(azim=50, elev=30)

        # 1. Plot point cloud
        pts = hull.points
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='k', s=10, alpha=0.5)

        # 2. Draw faces
        faces = [pts[s] for s in hull.simplices]
        poly = Poly3DCollection(faces, alpha=0.3, facecolors='gray', edgecolors='k')
        ax.add_collection3d(poly)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Maintains the original aspect ratio command
        ax.set_aspect('equal')
        
        plt.show()

    def plotProfile2D(self, curve_pts,):

        curve_pts = np.asarray(curve_pts, dtype=float)

        pts = curve_pts

        fig, ax = plt.subplots(figsize=(7, 6))

        # Plot the polyline (ordered)
        ax.plot(pts[:, 0], pts[:, 1], lw=1.5, c="k")

        # Explicitly close the loop if close_pts is provided
        ax.plot([pts[-1, 0], pts[0, 0]], [pts[-1, 1], pts[0, 1]], lw=1.5, c="k")

        # Optionally show points for debugging
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c="k", alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, c='k')

        plt.show()

    def plotStepBackshort3D(self, step_info):
        """Visualizes a step-backshort metadata dict returned by genStepBackshort."""
        boxes = step_info.get('boxes', [])
        if not boxes:
            raise ValueError('step_info must contain a non-empty boxes list.')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.pane.set_edgecolor('k')
        ax.yaxis.pane.set_edgecolor('k')
        ax.zaxis.pane.set_edgecolor('k')
        ax.xaxis.pane.set_facecolor('w')
        ax.yaxis.pane.set_facecolor('w')
        ax.zaxis.pane.set_facecolor('w')
        ax.grid(False)
        ax.view_init(azim=50, elev=30)

        all_vertices = []
        for box in boxes:
            x0, x1 = box['x_min'], box['x_max']
            y0, y1 = box['y_min'], box['y_max']
            z0, z1 = box['z_min'], box['z_max']
            vertices = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ])
            all_vertices.append(vertices)
            faces = [
                vertices[[0, 1, 2, 3]],
                vertices[[4, 5, 6, 7]],
                vertices[[0, 1, 5, 4]],
                vertices[[1, 2, 6, 5]],
                vertices[[2, 3, 7, 6]],
                vertices[[3, 0, 4, 7]],
            ]
            poly = Poly3DCollection(faces, alpha=0.25, facecolors='gray', edgecolors='k')
            ax.add_collection3d(poly)
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='k', s=10, alpha=0.4)

        pts = np.vstack(all_vertices)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
        ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
        ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
        ax.set_aspect('equal')
        plt.show()

    def plotProbehorn3D(self, horn_info):
        """Visualizes a probe-horn metadata dict returned by genProbehorn."""
        sections = horn_info.get('sections', [])
        if not sections:
            raise ValueError('horn_info must contain a non-empty sections list.')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.pane.set_edgecolor('k')
        ax.yaxis.pane.set_edgecolor('k')
        ax.zaxis.pane.set_edgecolor('k')
        ax.xaxis.pane.set_facecolor('w')
        ax.yaxis.pane.set_facecolor('w')
        ax.zaxis.pane.set_facecolor('w')
        ax.grid(False)
        ax.view_init(azim=50, elev=30)

        all_pts = []
        n_pts = len(sections[0]['pts'])
        for sec in sections:
            ring = np.asarray(sec['pts'], dtype=float)
            z = float(sec['z'])
            xs = np.append(ring[:, 0], ring[0, 0])
            ys = np.append(ring[:, 1], ring[0, 1])
            zs = np.full(xs.shape, z)
            ax.plot(xs, ys, zs, c='k', lw=0.8, alpha=0.7)
            all_pts.append(np.column_stack((ring[:, 0], ring[:, 1], np.full(len(ring), z))))

        # Longitudinal ridge lines connecting corresponding vertices
        stack = np.stack(all_pts)  # (n_sections, n_pts, 3)
        for i in range(n_pts):
            ax.plot(stack[:, i, 0], stack[:, i, 1], stack[:, i, 2], c='k', lw=0.5, alpha=0.4)

        pts = np.vstack(all_pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='k', s=2, alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
        plt.show()

class ConvexBackshort(Convex):
    """
    Specific class for Backshort generation, inheriting general Convex tools.
    """
    def genBackshort(self, a=9.525, b=4.7625, c=-7.725, k=6, grid_res=30, shifts=(0, -4.7625, -0.34575)):
        """
        Generates the original smooth backshort geometry and exports it to STEP.
        Maintains the original mathematical formulas and function names.
        """
        shift_x, shift_y, shift_z = shifts

        # 1. Generate point cloud
        x = np.linspace(-a, a, grid_res)
        y = np.linspace(-b, b, grid_res)
        X, Y = np.meshgrid(x, y)

        # Super-ellipsoid style surface calculation logic
        Z = c * np.sqrt(np.maximum(0, 1 - (X / a)**(2*k)) * np.maximum(0, 1 - (Y / b)**(2*k)))

        # 2. Apply translations
        X += shift_x
        Y += shift_y
        Z += shift_z

        # Create coordinate array
        raw_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # 3. Compute Convex Hull
        hull_data = ConvexHull(raw_points)

        # 4. Create CadQuery solid and export
        result = Workplane().polyhedron(hull_data.points, hull_data.simplices)

        exporters.export(result, str(self.model_path))
        return hull_data

    def genStepBackshort(
        self,
        a=9.525,
        b=4.7625,
        step_heights=(2.0, 2.0, 2.0),
        shrink=1.5,
        shifts=(0, -4.7625, -0.34575),
    ):
        """
        Generates a negative-Z step-backshort by stacking shrinking boxes.

        Parameters
        ----------
        a, b : float
            Base half-widths in X/Y.
        step_heights : sequence[float]
            Per-step thickness values. Each entry is stacked along negative Z.
        shrink : float
            XY shrink factor applied automatically for each higher step.
        shifts : tuple[float, float, float]
            Final translation applied to the stacked solid.
        """
        shift_x, shift_y, shift_z = shifts
        heights = [float(h) for h in step_heights if float(h) > 0]
        if not heights:
            raise ValueError('step_heights must contain at least one positive thickness.')
        if shrink <= 1.0:
            raise ValueError('shrink must be greater than 1.0.')

        solid = None
        z_cursor = 0.0
        for i, height in enumerate(heights):
            half_x = float(a) / (shrink ** i)
            half_y = float(b) / (shrink ** i)
            width_x = 2.0 * half_x
            width_y = 2.0 * half_y
            z_min = -(z_cursor + height)

            box = (
                cq.Workplane('XY')
                .box(width_x, width_y, height, centered=(True, True, False))
                .translate((0.0, 0.0, z_min))
            )
            solid = box if solid is None else solid.union(box)
            z_cursor += height

        boxes = []
        z_cursor = 0.0
        for i, height in enumerate(heights):
            half_x = float(a) / (shrink ** i)
            half_y = float(b) / (shrink ** i)
            z_min = -(z_cursor + height)
            boxes.append({
                'x_min': -half_x + shift_x,
                'x_max': half_x + shift_x,
                'y_min': -half_y + shift_y,
                'y_max': half_y + shift_y,
                'z_min': z_min + shift_z,
                'z_max': z_min + height + shift_z,
            })
            z_cursor += height

        solid = solid.translate((shift_x, shift_y, shift_z))
        exporters.export(solid, str(self.model_path))
        return {
            'type': 'stepbackshort',
            'base_half_width': (float(a), float(b)),
            'step_heights': heights,
            'n_steps': len(heights),
            'shrink': float(shrink),
            'total_depth': float(sum(heights)),
            'boxes': boxes,
        }


    def genStepBackshortCont(
        self,
        a=9.525,
        b=4.7625,
        step_heights=(2.0, 2.0, 2.0, 2.0, 2.0),
        shrink_params=(1.0, 0.2, 0.2, 0.2, 0.2),
        shifts=(0, -4.7625, -0.34575),
    ):
        """
        Generates a 5-step negative-Z step-backshort with monotonic per-step XY shrink factors.

        Parameters
        ----------
        a, b : float
            Base half-widths in X/Y.
        step_heights : sequence[float]
            Per-step thickness values (5 steps). Each entry is stacked along negative Z.
        shrink_params : sequence[float]
            Re-parameterized shrink controls used to guarantee monotonic ordering.
            The tuple must be (s1, s2, s3, s4, s5), where all values > 0.
            Per-step shrink factors are reconstructed cumulatively as:
              shrink_1 = s1
              shrink_2 = s1 + s2
              shrink_3 = s1 + s2 + s3
              shrink_4 = s1 + s2 + s3 + s4
              shrink_5 = s1 + s2 + s3 + s4 + s5
            This guarantees shrink_1 < shrink_2 < shrink_3 < shrink_4 < shrink_5.
        shifts : tuple[float, float, float]
            Final translation applied to the stacked solid.
        """
        shift_x, shift_y, shift_z = shifts

        heights = [float(h) for h in step_heights]
        sp = [float(v) for v in shrink_params]

        if len(heights) != 5:
            raise ValueError('step_heights must contain exactly 5 values.')
        if len(sp) != 5:
            raise ValueError('shrink_params must contain exactly 5 values: (s1, s2, s3, s4, s5).')
        if any(h <= 0.0 for h in heights):
            raise ValueError('All step_heights values must be positive.')

        s1, s2, s3, s4, s5 = sp
        if any(v < 0.0 for v in (s1, s2, s3, s4, s5)):
            raise ValueError('s1..s5 must be positive.')

        shrink_vals = [
            s1,
            s1 + s2,
            s1 + s2 + s3,
            s1 + s2 + s3 + s4,
            s1 + s2 + s3 + s4 + s5,
        ]

        solid = None
        z_cursor = 0.0
        for height, shrink in zip(heights, shrink_vals):
            if shrink < 1e-5:
                half_x = float(a)
                half_y = float(b)
            else:
                half_x = float(a) / shrink
                half_y = float(b) / shrink
            width_x = 2.0 * half_x
            width_y = 2.0 * half_y
            z_min = -(z_cursor + height)

            box = (
                cq.Workplane('XY')
                .box(width_x, width_y, height, centered=(True, True, False))
                .translate((0.0, 0.0, z_min))
            )
            solid = box if solid is None else solid.union(box)
            z_cursor += height

        boxes = []
        z_cursor = 0.0
        for height, shrink in zip(heights, shrink_vals):
            if shrink < 1e-5:
                half_x = float(a)
                half_y = float(b)
            else:
                half_x = float(a) / shrink
                half_y = float(b) / shrink
            z_min = -(z_cursor + height)
            boxes.append({
                'x_min': -half_x + shift_x,
                'x_max': half_x + shift_x,
                'y_min': -half_y + shift_y,
                'y_max': half_y + shift_y,
                'z_min': z_min + shift_z,
                'z_max': z_min + height + shift_z,
            })
            z_cursor += height

        solid = solid.translate((shift_x, shift_y, shift_z))
        exporters.export(solid, str(self.model_path))
        return {
            'type': 'stepbackshort_cont',
            'base_half_width': (float(a), float(b)),
            'step_heights': heights,
            'n_steps': len(heights),
            'shrink_params': sp,
            'shrinks': shrink_vals,
            'total_depth': float(sum(heights)),
            'boxes': boxes,
        }



class ConvexFinshape(Convex):
    
    def genFinshape(self, a=6, b=6, k=2, grid_res=400, shifts=(0.0, -1.0)):
      
        shift_x, shift_y = shifts
      
        t = np.linspace(0.0, np.pi, grid_res)

        x = a * np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** (2.0 / k)) + shift_x
        y = b * (np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** (2.0 / k)) - 1) + shift_y

        # Ordered boundary points on the arc
        curve_pts = list(zip(x, y))

        # -----------------------
        # Build a planar Face by closing the arc with a straight baseline
        wp = cq.Workplane("XY").polyline(curve_pts).close()

        # Turn the closed polyline into a wire, then create a planar face (sheet)
        wire = wp.wire().val()
        face = cq.Face.makeFromWires(wire)

        # export
        exporters.export(face, str(self.model_path))

        return curve_pts

    

class ConvexProbehorn(Convex):
    """
    Specific class for Probe-horn generation, inheriting general Convex tools.

    The horn tapers from a knife-edge WR-5 rectangular aperture (tip, at z = 0)
    down to an N-gon (circle approximation) base along negative Z, followed by
    an optional straight N-gon prism body. The taper silhouette is controlled
    by a single power-law parameter, so both convex and concave (pencil-like)
    profiles are covered:
        r(theta, t) = r_tip(theta) + (r_base(theta) - r_tip(theta)) * t**p
    with t = 0 at the aperture (tip) and t = 1 at the taper base.
        p = 1  : straight cone / pyramid
        p < 1  : outward-bulging silhouette
        p > 1  : concave, pencil-tip / trumpet-like silhouette
    """

    @staticmethod
    def _sectionAngles(n_sides, wr_a, wr_b, theta_offset):
        """
        Builds the common set of boundary directions used for every section.
        Uniform N-gon directions plus the 4 rectangle-corner directions, so the
        tip section reproduces the WR-5 rectangle exactly (sharp corners).
        """
        base = np.mod(theta_offset + np.arange(n_sides) * 2.0 * np.pi / n_sides, 2.0 * np.pi)
        corner = np.arctan2(wr_b, wr_a)
        corners = np.mod(np.array([corner, np.pi - corner, np.pi + corner, -corner]), 2.0 * np.pi)
        angles = np.sort(np.concatenate([base, corners]))
        # Drop near-duplicate directions
        keep = np.ones(angles.shape, dtype=bool)
        keep[1:] = np.diff(angles) > 1e-9
        if angles[-1] - angles[0] > 2.0 * np.pi - 1e-9:
            keep[-1] = False
        return angles[keep]

    @staticmethod
    def _radiusRect(angles, wr_a, wr_b):
        """Radial distance from origin to the WR-5 rectangle boundary."""
        cos_t = np.abs(np.cos(angles))
        sin_t = np.abs(np.sin(angles))
        with np.errstate(divide='ignore'):
            r_x = np.where(cos_t > 1e-12, (wr_a / 2.0) / cos_t, np.inf)
            r_y = np.where(sin_t > 1e-12, (wr_b / 2.0) / sin_t, np.inf)
        return np.minimum(r_x, r_y)

    @staticmethod
    def _radiusNgon(angles, n_sides, base_radius, theta_offset):
        """
        Radial distance from origin to the boundary of the regular N-gon
        inscribed in a circle of base_radius (flat-side sampling), so the base
        section is exactly the regular N-gon even at intermediate directions.
        """
        apothem = base_radius * np.cos(np.pi / n_sides)
        local = np.mod(angles - theta_offset, 2.0 * np.pi / n_sides) - np.pi / n_sides
        return apothem / np.cos(local)

    def genProbehorn(
        self,
        wr_a=1.2954,
        wr_b=0.6477,
        base_radius=3.0,
        taper_length=8.0,
        body_length=10.0,
        n_sides=20,
        profile_power=1.0,
        n_slices=40,
        theta_offset=None,
        shifts=(0.0, 0.0, 0.0),
    ):
        """
        Generates a probe horn and exports metal body + WR-5 vacuum as a
        two-solid STEP assembly ('horn_metal', 'wr5_vacuum').

        Parameters
        ----------
        wr_a, wr_b : float
            Waveguide aperture inner dimensions [mm]. Defaults: WR-5.
        base_radius : float
            Circumscribed radius of the N-gon at the taper base.
        taper_length : float
            Axial length of the tapered section (aperture at z=0, base at
            z=-taper_length).
        body_length : float
            Axial length of the straight N-gon prism below the taper
            (0 disables the body).
        n_sides : int
            Number of polygon sides approximating the circle (4 -> pyramid).
        profile_power : float
            Power-law exponent p of the taper silhouette (see class docstring).
        n_slices : int
            Number of loft sections along the taper.
        theta_offset : float or None
            Angular offset of the N-gon vertices [rad]. None -> pi/n_sides
            (one flat side faces +X; for n_sides=4 the flats align with the
            WR-5 rectangle sides).
        shifts : tuple[float, float, float]
            Final translation applied to both solids.
        """
        shift_x, shift_y, shift_z = shifts

        if n_sides < 3:
            raise ValueError('n_sides must be at least 3.')
        if profile_power <= 0.0:
            raise ValueError('profile_power must be positive.')
        if taper_length <= 0.0:
            raise ValueError('taper_length must be positive.')
        if body_length < 0.0:
            raise ValueError('body_length must be non-negative.')
        if n_slices < 2:
            raise ValueError('n_slices must be at least 2.')
        if theta_offset is None:
            theta_offset = np.pi / n_sides

        # 1. Common boundary directions and tip/base radial profiles
        angles = self._sectionAngles(n_sides, wr_a, wr_b, theta_offset)
        r_tip = self._radiusRect(angles, wr_a, wr_b)
        r_base = self._radiusNgon(angles, n_sides, base_radius, theta_offset)

        if np.any(r_base <= r_tip):
            raise ValueError(
                'Base N-gon does not enclose the WR-5 aperture in every '
                'direction. Increase base_radius (or n_sides).'
            )

        # 2. Section rings from tip (t=0, z=0) to base (t=1, z=-taper_length)
        sections = []
        for j in range(n_slices + 1):
            t = j / n_slices
            r = r_tip + (r_base - r_tip) * t ** profile_power
            pts = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
            sections.append({'z': -t * taper_length, 'pts': [tuple(p) for p in pts]})

        # 3. Loft the taper through all section wires
        wires = []
        for sec in sections:
            wp = cq.Workplane('XY', origin=(0.0, 0.0, sec['z'])).polyline(sec['pts']).close()
            wires.append(wp.wire().val())
        outer = cq.Solid.makeLoft(wires, True)

        # 4. Straight N-gon prism body below the taper
        total_length = taper_length + body_length
        if body_length > 0.0:
            base_pts = sections[-1]['pts']
            body = (
                cq.Workplane('XY', origin=(0.0, 0.0, -total_length))
                .polyline(base_pts)
                .close()
                .extrude(body_length)
                .val()
            )
            outer = outer.fuse(body)

        # 5. WR-5 rectangular channel (vacuum), running the full length
        vacuum = cq.Solid.makeBox(
            wr_a, wr_b, total_length,
            pnt=cq.Vector(-wr_a / 2.0, -wr_b / 2.0, -total_length),
        )
        metal = outer.cut(vacuum)

        # 6. Apply translations and export as a named two-solid STEP assembly
        metal = metal.translate(cq.Vector(shift_x, shift_y, shift_z))
        vacuum = vacuum.translate(cq.Vector(shift_x, shift_y, shift_z))

        assy = cq.Assembly()
        assy.add(metal, name='horn_metal')
        assy.add(vacuum, name='wr5_vacuum')
        assy.save(str(self.model_path))

        # Shifted sections for plotting/inspection
        sections_shifted = [
            {
                'z': sec['z'] + shift_z,
                'pts': [(x + shift_x, y + shift_y) for (x, y) in sec['pts']],
            }
            for sec in sections
        ]

        return {
            'type': 'probehorn',
            'wr_aperture': (float(wr_a), float(wr_b)),
            'base_radius': float(base_radius),
            'taper_length': float(taper_length),
            'body_length': float(body_length),
            'total_length': float(total_length),
            'n_sides': int(n_sides),
            'profile_power': float(profile_power),
            'n_slices': int(n_slices),
            'theta_offset': float(theta_offset),
            'sections': sections_shifted,
        }
