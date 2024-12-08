import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon


def generate_ray(robot_pos, angle, max_distance=100):
    """Generate a ray extending from the robot's position at a specific angle."""
    x_start, y_start = robot_pos
    x_end = x_start + max_distance * np.cos(np.radians(angle))
    y_end = y_start + max_distance * np.sin(np.radians(angle))
    return LineString([(x_start, y_start), (x_end, y_end)])


def ray_casting(robot_pos, obstacles, map_boundary, n_rays=360, max_distance=100):
    """Cast rays in all directions and find the visible points."""
    visible_points = []
    for angle in np.linspace(0, 360, n_rays, endpoint=False):
        ray = generate_ray((robot_pos.x, robot_pos.y), angle, max_distance)
        closest_intersection = None
        closest_distance = float("inf")

        # Check intersections with obstacles and the map boundary
        for obstacle in obstacles + [map_boundary]:
            intersection = ray.intersection(obstacle)
            if not intersection.is_empty:
                if isinstance(intersection, Point):  # Single intersection point
                    distance = robot_pos.distance(intersection)
                    if distance < closest_distance:
                        closest_intersection = intersection
                        closest_distance = distance
                elif isinstance(
                    intersection, MultiPoint
                ):  # Multiple intersection points
                    for point in intersection.geoms:
                        distance = robot_pos.distance(point)
                        if distance < closest_distance:
                            closest_intersection = point
                            closest_distance = distance
                elif isinstance(intersection, LineString):  # Line segment intersection
                    for point in intersection.coords:
                        point_geom = Point(point)
                        distance = robot_pos.distance(point_geom)
                        if distance < closest_distance:
                            closest_intersection = point_geom
                            closest_distance = distance

        if closest_intersection:
            visible_points.append((closest_intersection.x, closest_intersection.y))
        else:
            # If no obstacle, use ray endpoint
            visible_points.append((ray.boundary.geoms[1].x, ray.boundary.geoms[1].y))

    return visible_points


# Robot position as a Shapely Point
robot_position = Point(5, 5)

# Obstacles defined as polygons
obstacles = [
    Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    Polygon([(7, 7), (9, 7), (9, 9), (7, 9)]),
    Polygon([(6, 2), (8, 2), (8, 4), (6, 4)]),
]

# Define the map boundary as a large rectangle
map_boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

# Cast rays to calculate visibility
visible_points = ray_casting(robot_position, obstacles, map_boundary, n_rays=360)

# Plot the results
fig, ax = plt.subplots(figsize=(8, 8))

# Draw obstacles
for obstacle in obstacles:
    x, y = obstacle.exterior.xy
    ax.fill(x, y, alpha=0.5, fc="gray", ec="black", label="Obstacle")

# Draw the map boundary
x, y = map_boundary.exterior.xy
ax.plot(x, y, "black", linestyle="--", label="Map Boundary")

# Draw the robot's position
ax.plot(robot_position.x, robot_position.y, "ro", label="Robot")

# Draw rays and visible points
for point in visible_points:
    ax.plot([robot_position.x, point[0]], [robot_position.y, point[1]], "b-", alpha=0.3)
ax.scatter(*zip(*visible_points, strict=False), c="blue", s=10, label="Visible Points")

# Set plot limits and labels
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal", adjustable="box")
ax.legend()
plt.title("Robot Visibility with Map Boundary")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
