import pygame
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# ---------------------- CONFIGURATION ----------------------
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
ROBOT1_POSITION = (300, 300)  # Robot 1 starting position
ROBOT2_POSITION = (500, 300)  # Robot 2 starting position
ROBOT_RADIUS = 5
NUM_RAYS = 360
RAY_LENGTH = 1000  # Max length of rays
OVERLAP_THRESHOLD = 2.0  # Distance threshold to consider hits the "same" point

# Define colors
WHITE = (255, 255, 255)
BLACK = (11, 11, 11)
GRAY = (160, 160, 160)
GREEN = (144, 238, 144)  # Light pastel green
BLUE = (135, 206, 250)   # Light pastel blue
RED = (255, 182, 193)    # Light pastel red

# ---------------------- ENVIRONMENT -------------------------
obstacles = [
    # A rectangle obstacle
    [(200, 200), (300, 200), (300, 250), (200, 250)],
    # A triangle obstacle
    [(500, 100), (550, 200), (450, 200)],
    # Another polygon
    [(600, 400), (650, 450), (600, 500), (550, 450)]
]

def polygon_to_edges(polygon):
    edges = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        edges.append((p1, p2))
    return edges

# Combine all obstacle edges
obstacle_edges = []
for poly in obstacles:
    obstacle_edges.extend(polygon_to_edges(poly))

# Boundary edges
boundary = [
    ((0, 0), (WINDOW_WIDTH, 0)),
    ((WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT)),
    ((WINDOW_WIDTH, WINDOW_HEIGHT), (0, WINDOW_HEIGHT)),
    ((0, WINDOW_HEIGHT), (0, 0))
]

all_edges = obstacle_edges + boundary

# ---------------------- RAY CASTING HELPER FUNCTIONS ----------------------
def line_intersection(p0, p1, p2, p3):
    """
    Returns the intersection point of line segment p0p1 and p2p3 if exists, otherwise None.
    """
    s1_x = p1[0] - p0[0]
    s1_y = p1[1] - p0[1]
    s2_x = p3[0] - p2[0]
    s2_y = p3[1] - p2[1]

    denom = (-s2_x * s1_y + s1_x * s2_y)
    if denom == 0:
        return None  # Parallel or coincident lines

    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / denom
    t = ( s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / denom

    if 0 <= s <= 1 and 0 <= t <= 1:
        ix = p0[0] + (t * s1_x)
        iy = p0[1] + (t * s1_y)
        return (ix, iy)
    return None

def cast_ray(start, angle, edges):
    """
    Casts a ray from start at the given angle and returns the closest intersection point
    with any edge, or None if no intersection.
    """
    end_x = start[0] + RAY_LENGTH * math.cos(angle)
    end_y = start[1] + RAY_LENGTH * math.sin(angle)
    ray_end = (end_x, end_y)

    closest_point = None
    closest_dist = float('inf')

    for edge in edges:
        intersect_pt = line_intersection(start, ray_end, edge[0], edge[1])
        if intersect_pt is not None:
            dist = math.dist(start, intersect_pt)
            if dist < closest_dist:
                closest_dist = dist
                closest_point = intersect_pt

    return closest_point

def cast_all_rays(robot_pos, edges, num_rays=360):
    hits = []
    for i in range(num_rays):
        angle = math.radians(i)
        hit_point = cast_ray(robot_pos, angle, edges)
        hits.append(hit_point)
    return hits

def remove_overlapping_points(robot1_pos, robot2_pos, hits1, hits2, threshold):
    """
    Given two sets of hit points (hits1 and hits2) from two robots,
    eliminate overlaps. If two hits at the same angle are within 'threshold' distance,
    only keep the one corresponding to the robot closer to that point.
    """
    # Both hits arrays are indexed by the same angle (0 to NUM_RAYS-1)
    for i in range(len(hits1)):
        pt1 = hits1[i]
        pt2 = hits2[i]
        if pt1 is not None and pt2 is not None:
            # Check if they are close enough to be considered the same point
            if math.dist(pt1, pt2) < threshold:
                dist1 = math.dist(robot1_pos, pt1)
                dist2 = math.dist(robot2_pos, pt2)
                # Keep only the one closer to its robot
                if dist1 < dist2:
                    # Robot 1 is closer, Robot 2 loses this point
                    hits2[i] = None
                else:
                    # Robot 2 is closer, Robot 1 loses this point
                    hits1[i] = None
    return hits1, hits2

def polygon_to_shapely(polygon):
    """Convert a list of tuples to a Shapely Polygon."""
    return Polygon(polygon)

def shapely_to_polygon(shapely_poly):
    """Convert a Shapely Polygon or valid geometry to a list of points."""
    if shapely_poly.is_empty:
        return []
    if shapely_poly.geom_type == "Polygon":
        return list(shapely_poly.exterior.coords)
    elif shapely_poly.geom_type == "MultiPolygon":
        # Return the exterior of the largest polygon
        largest_poly = max(shapely_poly.geoms, key=lambda p: p.area)
        return list(largest_poly.exterior.coords)
    return []


def divide_intersection(intersection, robot1_pos, robot2_pos):
    """
    Divide the intersection polygon based on proximity to two robots.
    Handles cases where the intersection is not a Polygon.
    """
    # Check if intersection is a valid Polygon
    if not isinstance(intersection, Polygon) or intersection.is_empty:
        return Polygon(), Polygon()

    robot1_vertices = []
    robot2_vertices = []

    # Divide the intersection based on proximity to robots
    for point in intersection.exterior.coords:
        p = Point(point)
        if p.distance(Point(robot1_pos)) < p.distance(Point(robot2_pos)):
            robot1_vertices.append(point)
        else:
            robot2_vertices.append(point)

    # Ensure polygons are valid (require at least 4 coordinates)
    if len(robot1_vertices) < 3:
        robot1_vertices = []
    else:
        robot1_vertices.append(robot1_vertices[0])  # Close the polygon

    if len(robot2_vertices) < 3:
        robot2_vertices = []
    else:
        robot2_vertices.append(robot2_vertices[0])  # Close the polygon

    divided_polygon1 = Polygon(robot1_vertices) if robot1_vertices else Polygon()
    divided_polygon2 = Polygon(robot2_vertices) if robot2_vertices else Polygon()

    return divided_polygon1, divided_polygon2



def draw_polygon(screen, polygon, color):
    """Draw a filled polygon on the screen."""
    if polygon.is_empty:
        return
    pygame.draw.polygon(screen, color, shapely_to_polygon(polygon))

# ---------------------- MAIN SIMULATION LOOP ----------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("2D Two-Robot Ray Casting with Polygon Division")
    clock = pygame.time.Clock()

    running = True
    robot1_pos = list(ROBOT1_POSITION)
    robot2_pos = list(ROBOT2_POSITION)

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Robot movement controls
        keys = pygame.key.get_pressed()
        speed = 1
        if keys[pygame.K_UP]:
            robot2_pos[1] -= speed
        if keys[pygame.K_DOWN]:
            robot2_pos[1] += speed
        if keys[pygame.K_LEFT]:
            robot2_pos[0] -= speed
        if keys[pygame.K_RIGHT]:
            robot2_pos[0] += speed
        if keys[pygame.K_w]:
            robot1_pos[1] -= speed
        if keys[pygame.K_s]:
            robot1_pos[1] += speed
        if keys[pygame.K_a]:
            robot1_pos[0] -= speed
        if keys[pygame.K_d]:
            robot1_pos[0] += speed

        # Define the FoV polygons as circles (simplified for this example)
        fov_robot1 = Point(robot1_pos).buffer(150).simplify(1.0)  # Circle around Robot 1
        fov_robot2 = Point(robot2_pos).buffer(150).simplify(1.0)  # Circle around Robot 2

        # Find intersection
        intersection = fov_robot1.intersection(fov_robot2)

        # Handle non-polygon intersection types
        if not isinstance(intersection, Polygon):
            intersection = Polygon()

        # Divide intersection into two polygons based on proximity
        divided1, divided2 = divide_intersection(intersection, robot1_pos, robot2_pos)

        # Subtract intersection from original polygons and add divided portions
        fov_robot1 = fov_robot1.difference(intersection).union(divided1)
        fov_robot2 = fov_robot2.difference(intersection).union(divided2)

        # Ensure the results are valid polygons
        fov_robot1 = fov_robot1 if fov_robot1.geom_type in ["Polygon", "MultiPolygon"] else Polygon()
        fov_robot2 = fov_robot2 if fov_robot2.geom_type in ["Polygon", "MultiPolygon"] else Polygon()



        # Draw everything
        screen.fill(BLACK)  # Clear screen

        # Draw obstacles
        for poly in obstacles:
            pygame.draw.polygon(screen, GRAY, poly, 0)

        # Draw boundaries
        pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 1)

        # Draw polygons
        draw_polygon(screen, fov_robot1, GREEN)
        draw_polygon(screen, fov_robot2, BLUE)
        draw_polygon(screen, intersection, RED)  # Optional: Show intersection in red

        # Draw robots
        pygame.draw.circle(screen, WHITE, (int(robot1_pos[0]), int(robot1_pos[1])), ROBOT_RADIUS)
        pygame.draw.circle(screen, WHITE, (int(robot2_pos[0]), int(robot2_pos[1])), ROBOT_RADIUS)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
