import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon


# Step 1: Define the S-shaped domain
def create_s_shape():
    outer_boundary = Polygon([(0, 0), (15, 0), (15, 10), (0, 10), (0, 0)])
    inner_obstacle_1 = Polygon(
        [(4, 6), (6, 6), (6, 10), (4, 10), (4, 6)]
    )  # Upper block
    inner_obstacle_2 = Polygon(
        [(9, 0), (11, 0), (11, 4), (9, 4), (9, 0)]
    )  # Lower block
    domain = outer_boundary.difference(inner_obstacle_1).difference(inner_obstacle_2)
    return domain, [inner_obstacle_1, inner_obstacle_2]


# Step 2: Discretize the domain
def discretize_domain(domain, resolution=0.2):
    minx, miny, maxx, maxy = domain.bounds
    x = np.arange(minx, maxx, resolution)
    y = np.arange(miny, maxy, resolution)
    points = [Point(px, py) for px in x for py in y if domain.contains(Point(px, py))]
    return points


# Step 3: Compute visibility polygon for a sensor
def compute_visibility_polygon(sensor, domain, obstacles):
    """Computes the visibility polygon for a sensor, considering obstacles."""
    visibility_polygon = domain
    for obstacle in obstacles:
        visibility_polygon = visibility_polygon.difference(obstacle)
    return visibility_polygon


# Step 4: Assign points to sensors based on visibility and proximity
def compute_vvd(sensors, domain, obstacles, points):
    """Computes the VVD by assigning points to the nearest visible sensor."""
    vvd = {sensor: [] for sensor in sensors}
    for point in points:
        visible_sensors = []
        for sensor in sensors:
            visibility_polygon = compute_visibility_polygon(sensor, domain, obstacles)
            if visibility_polygon.contains(point):
                visible_sensors.append(sensor)
        if visible_sensors:
            nearest_sensor = min(visible_sensors, key=lambda s: point.distance(s))
            vvd[nearest_sensor].append(point)
    return vvd


# Step 5: Plot the VVD
def plot_vvd(domain, sensors, vvd, obstacles):
    fig, ax = plt.subplots()
    # Plot domain
    x, y = domain.exterior.xy
    ax.plot(x, y, color="black")
    # Plot obstacles
    for obs in obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, color="gray")
    # Plot sensors
    for sensor in sensors:
        ax.plot(sensor.x, sensor.y, "ro")
    # Plot VVD
    colors = ["red", "blue", "green", "orange", "purple"]
    for i, (sensor, region_points) in enumerate(vvd.items()):
        x = [p.x for p in region_points]
        y = [p.y for p in region_points]
        ax.scatter(x, y, color=colors[i % len(colors)], s=5)
    plt.axis("equal")
    plt.title("Visibility-Based Voronoi Diagram (VVD)")
    plt.show()


# Step 6: Plot only the boundaries of the S-shaped map
def plot_boundaries(domain, obstacles):
    fig, ax = plt.subplots()
    # Plot domain boundary
    x, y = domain.exterior.xy
    ax.plot(x, y, color="black", label="Domain Boundary")
    # Plot obstacles
    for obs in obstacles:
        x, y = obs.exterior.xy
        ax.plot(x, y, color="gray", label="Obstacle Boundary")
    ax.legend()
    plt.axis("equal")
    plt.title("S-Shaped Map with Boundaries")
    plt.show()


# Main function
if __name__ == "__main__":
    # Create the S-shaped domain
    domain, obstacles = create_s_shape()

    # Define sensors
    sensors = [Point(3, 3), Point(7, 7), Point(12, 3)]

    # Discretize the domain
    points = discretize_domain(domain, resolution=0.2)

    # Compute VVD with visibility constraints
    vvd = compute_vvd(sensors, domain, obstacles, points)

    # Plot the boundaries only
    plot_boundaries(domain, obstacles)

    # Plot VVD
    plot_vvd(domain, sensors, vvd, obstacles)
