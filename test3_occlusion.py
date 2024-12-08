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
    coords = shapely_to_polygon(polygon)
    if len(coords) < 3:
        # Not a valid polygon for drawing
        return
    pygame.draw.polygon(screen, color, coords)




def compute_visibility_polygon(robot_pos, edges):
    """
    Compute the visibility polygon for a robot by ray-casting to all obstacle vertices
    and window edges.
    """
    rays = []
    # Get all unique points (vertices of obstacles and window edges)
    vertices = set()
    for edge in edges:
        vertices.add(edge[0])
        vertices.add(edge[1])

    # Cast rays to each vertex and slightly offset angles
    for vertex in vertices:
        angle = math.atan2(vertex[1] - robot_pos[1], vertex[0] - robot_pos[0])
        rays.extend([angle - 0.0001, angle, angle + 0.0001])

    # Sort rays by angle
    rays = sorted(set(rays))

    # Find intersections for each ray
    points = []
    for ray in rays:
        end_x = robot_pos[0] + RAY_LENGTH * math.cos(ray)
        end_y = robot_pos[1] + RAY_LENGTH * math.sin(ray)
        closest_point = None
        closest_dist = float('inf')

        for edge in edges:
            intersect_pt = line_intersection(robot_pos, (end_x, end_y), edge[0], edge[1])
            if intersect_pt:
                dist = math.dist(robot_pos, intersect_pt)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = intersect_pt

        if closest_point:
            points.append(closest_point)

    # Sort points in counter-clockwise order
    points = sorted(points, key=lambda p: math.atan2(p[1] - robot_pos[1], p[0] - robot_pos[0]))

    # Return as a Shapely Polygon
    return Polygon(points)

# Helper to validate and fix geometry
def validate_geometry(geometry):
    if not geometry.is_valid:
        geometry = geometry.buffer(0)  # Fix invalid geometry
    return geometry

# ---------------------- MAIN SIMULATION LOOP ----------------------
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import split

def split_by_voronoi(intersection, robot1_pos, robot2_pos):
    """
    Split the intersection polygon into two polygons based on the Voronoi boundary line
    (the perpendicular bisector between the two robot positions).
    """
    if intersection.is_empty or intersection.geom_type not in ["Polygon", "MultiPolygon"]:
        return Polygon(), Polygon()

    x1, y1 = robot1_pos
    x2, y2 = robot2_pos
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = (x2 - x1), (y2 - y1)

    # The Voronoi boundary for two points is the perpendicular bisector.
    # Normal vector to the line connecting the robots is (-dy, dx).
    # Create a very long line through midpoint M along (-dy, dx).
    line_length = max(WINDOW_WIDTH, WINDOW_HEIGHT) * 2
    line_coords = [
        (mx - dy * line_length, my + dx * line_length),
        (mx + dy * line_length, my - dx * line_length)
    ]
    dividing_line = LineString(line_coords)

    # Split the intersection with the dividing line
    result = split(intersection, dividing_line)

    # If split is successful, we should get two polygons
    if len(result.geoms) == 2:
        polyA, polyB = result.geoms[0], result.geoms[1]
        cA = polyA.centroid
        cB = polyB.centroid

        distA_r1 = cA.distance(Point(robot1_pos))
        distA_r2 = cA.distance(Point(robot2_pos))
        distB_r1 = cB.distance(Point(robot1_pos))
        distB_r2 = cB.distance(Point(robot2_pos))

        # Assign each polygon to the closest robot
        assignA = 'R1' if distA_r1 < distA_r2 else 'R2'
        assignB = 'R1' if distB_r1 < distB_r2 else 'R2'

        # If both ended up assigned to the same robot, handle gracefully:
        if assignA == assignB:
            # Decide assignment to minimize total distance
            # Try A->R1, B->R2
            cost_1 = distA_r1 + distB_r2
            # Try A->R2, B->R1
            cost_2 = distA_r2 + distB_r1
            if cost_1 < cost_2:
                # A->R1, B->R2
                return polyA, polyB
            else:
                # A->R2, B->R1
                return polyB, polyA
        else:
            # Different assignments; just return accordingly
            if assignA == 'R1' and assignB == 'R2':
                return polyA, polyB
            else:
                return polyB, polyA

    # If we couldn't split into two polygons, return empty
    return Polygon(), Polygon()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Infinite Visibility with Two Robots")
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

        # Compute visibility polygons
        visibility_robot1 = compute_visibility_polygon(robot1_pos, all_edges)
        visibility_robot2 = compute_visibility_polygon(robot2_pos, all_edges)

        # Validate visibility polygons
        visibility_robot1 = validate_geometry(visibility_robot1)
        visibility_robot2 = validate_geometry(visibility_robot2)

        # Compute intersection
        intersection = visibility_robot1.intersection(visibility_robot2)
        intersection = validate_geometry(intersection)

        # If there is intersection (overlap), use the Voronoi line to split it
        if not intersection.is_empty:
            # Split intersection into two polygons
            divided1, divided2 = split_by_voronoi(intersection, robot1_pos, robot2_pos)
            divided1 = validate_geometry(divided1)
            divided2 = validate_geometry(divided2)

            # Update visibility for Robot 1
            temp1 = visibility_robot1.difference(intersection)
            temp1 = validate_geometry(temp1)
            if not divided1.is_empty:
                temp1 = temp1.union(divided1)
                temp1 = validate_geometry(temp1)
            visibility_robot1 = temp1

            # Update visibility for Robot 2
            temp2 = visibility_robot2.difference(intersection)
            temp2 = validate_geometry(temp2)
            if not divided2.is_empty:
                temp2 = temp2.union(divided2)
                temp2 = validate_geometry(temp2)
            visibility_robot2 = temp2

        # Draw everything
        screen.fill(BLACK)

        # Draw obstacles
        for poly in obstacles:
            pygame.draw.polygon(screen, GRAY, poly, 0)

        # Draw boundaries
        pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 1)

        # Draw visibility polygons (only if they have enough points)
        draw_polygon(screen, visibility_robot1, GREEN)
        draw_polygon(screen, visibility_robot2, BLUE)

        # Optionally draw intersection or any debugging info if needed
        # draw_polygon(screen, intersection, RED)

        # Draw robots
        pygame.draw.circle(screen, WHITE, (int(robot1_pos[0]), int(robot1_pos[1])), ROBOT_RADIUS)
        pygame.draw.circle(screen, WHITE, (int(robot2_pos[0]), int(robot2_pos[1])), ROBOT_RADIUS)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()