import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

# -------------------------------------------
# Step 1: Define environment
# -------------------------------------------
boundary_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
boundary_poly = Polygon(boundary_coords)

obstacles = [
    Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
    Polygon([(6, 6), (7, 6), (7, 7), (6, 7)])
]

N = 3
robot_positions = np.array([
    [2.0, 2.0],
    [8.0, 2.0],
    [5.0, 8.0]
])

max_iterations = 50
tolerance = 0.01
num_rays = 360


def is_line_intersecting_polygons(line, polygons):
    intersections = []
    for poly in polygons:
        if line.intersects(poly):
            intersections.append(line.intersection(poly))
    # Check boundary
    if line.intersects(boundary_poly):
        intersections.append(line.intersection(boundary_poly))

    start = np.array(line.coords[0])
    closest_point = None
    min_dist = float('inf')
    for inter in intersections:
        if inter.geom_type == 'MultiPoint':
            for pt in inter:
                dist = np.linalg.norm(np.array(pt.coords[0]) - start)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = pt
        elif inter.geom_type == 'Point':
            dist = np.linalg.norm(np.array(inter.coords[0]) - start)
            if dist < min_dist:
                min_dist = dist
                closest_point = inter
        elif inter.geom_type == 'LineString':
            pts = list(inter.coords)
            for cpt in pts:
                dist = np.linalg.norm(np.array(cpt) - start)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = Point(cpt)

    return closest_point


def compute_visibility_polygon(robot_pos, boundary_poly, obstacles, num_rays=360):
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    visible_points = []
    polygons_to_check = [boundary_poly] + obstacles

    for ang in angles:
        direction = np.array([np.cos(ang), np.sin(ang)])
        far_point = robot_pos + 1000 * direction
        line = LineString([tuple(robot_pos), tuple(far_point)])
        closest_intersection = is_line_intersecting_polygons(line, polygons_to_check)
        if closest_intersection is not None:
            visible_points.append(closest_intersection.coords[0])

    if not visible_points:
        return None

    vis_poly = Polygon(visible_points)
    obstacle_union = unary_union(obstacles)
    free_space = boundary_poly.difference(obstacle_union)
    visibility_polygon = vis_poly.intersection(free_space)
    return visibility_polygon


def assign_points_to_robots(sample_points, visibility_polygons, robot_positions):
    assignments = {i: [] for i in range(len(robot_positions))}

    for pt in sample_points:
        point_geom = Point(pt)
        visible_robots = []
        for i, vis_poly in enumerate(visibility_polygons):
            if vis_poly is not None and point_geom.within(vis_poly):
                visible_robots.append(i)
        if len(visible_robots) == 1:
            assignments[visible_robots[0]].append(pt)
        elif len(visible_robots) > 1:
            dists = [np.linalg.norm(pt - robot_positions[j]) for j in visible_robots]
            closest_robot = visible_robots[np.argmin(dists)]
            assignments[closest_robot].append(pt)
    return assignments


def polygons_from_assignments(assignments):
    from shapely.geometry import MultiPoint
    vd_polygons = {}
    for i, pts in assignments.items():
        if len(pts) < 3:
            vd_polygons[i] = None
        else:
            multi_pt = MultiPoint(pts)
            vd_polygons[i] = multi_pt.convex_hull
    return vd_polygons


for iteration in range(max_iterations):
    visibility_polygons = []
    for rpos in robot_positions:
        vp = compute_visibility_polygon(rpos, boundary_poly, obstacles, num_rays=num_rays)
        visibility_polygons.append(vp)

    xs = np.linspace(0, 10, 50)
    ys = np.linspace(0, 10, 50)
    grid_points = np.array([(x, y) for x in xs for y in ys])

    obstacle_union = unary_union(obstacles)
    free_space = boundary_poly.difference(obstacle_union)
    free_points = [p for p in grid_points if Point(p).within(free_space)]

    assignments = assign_points_to_robots(free_points, visibility_polygons, robot_positions)
    vd_polygons = polygons_from_assignments(assignments)

    new_positions = []
    for i, pts in assignments.items():
        if len(pts) == 0:
            new_positions.append(robot_positions[i])
        else:
            arr_pts = np.array(pts)
            centroid = np.mean(arr_pts, axis=0)
            new_positions.append(centroid)
    new_positions = np.array(new_positions)

    movement = np.linalg.norm(new_positions - robot_positions, axis=1)
    if np.all(movement < tolerance):
        robot_positions = new_positions
        break
    robot_positions = new_positions

# -------------------------------------------
# Visualization
# -------------------------------------------
fig, ax = plt.subplots()
# Plot boundary
bx, by = boundary_poly.exterior.xy
ax.plot(bx, by, 'k-')

# Plot obstacles
for obs in obstacles:
    ox, oy = obs.exterior.xy
    ax.fill(ox, oy, color='gray', alpha=0.5)

colors = ['red', 'blue', 'green']

# Plot final robot positions
for i, rpos in enumerate(robot_positions):
    ax.plot(rpos[0], rpos[1], 'o', color=colors[i], label=f'Robot {i}')

# Plot visibility polygons for each robot
for i, vp in enumerate(visibility_polygons):
    if vp is not None and not vp.is_empty:
        vx, vy = vp.exterior.xy
        # Slightly different alpha or hatch to distinguish from VVD
        ax.fill(vx, vy, color=colors[i], alpha=0.1, label=f'Robot {i} Visibility')

# Plot VVD polygons for each robot
for i, vdpoly in vd_polygons.items():
    if vdpoly is not None and not vdpoly.is_empty:
        vx, vy = vdpoly.exterior.xy
        # Use a bit darker alpha than visibility
        ax.fill(vx, vy, color=colors[i], alpha=0.3, label=f'Robot {i} VVD')

ax.set_aspect('equal', 'box')
ax.legend()
plt.title("Robots with Visibility and Visibility-based Voronoi Diagrams")
plt.show()
