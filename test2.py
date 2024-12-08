import pygame
from shapely.geometry import Polygon
import math

# ---------------------- CONFIGURATION ----------------------
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
ROBOT_POSITION = (400, 300)  # Robot starting position
ROBOT_RADIUS = 5
NUM_RAYS = 360
RAY_LENGTH = 1000  # Max length of rays

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# ---------------------- ENVIRONMENT -------------------------
# The boundary of the map (top-left: (0,0), bottom-right: (WINDOW_WIDTH, WINDOW_HEIGHT))
# We'll represent obstacles as a list of polygons, where each polygon is a list of vertices.
# For simplicity, define a few obstacles manually.
obstacles = [
    # A rectangle obstacle
    [(200, 200), (300, 200), (300, 250), (200, 250)],
    # A triangle obstacle
    [(500, 100), (550, 200), (450, 200)],
    # Another polygon
    [(600, 400), (650, 450), (600, 500), (550, 450)]
]

# Convert polygon edges into a list of all line segments for easy intersection checks
def polygon_to_edges(polygon):
    edges = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        edges.append((p1, p2))
    return edges

def shapely_to_pygame_coords(polygon):
    return [(int(x), int(y)) for x, y in polygon.exterior.coords]

# Combine all obstacle edges
obstacle_edges = []
for poly in obstacles:
    obstacle_edges.extend(polygon_to_edges(poly))

# Also include the boundary as line segments
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
    Uses vector cross product approach.
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
        # Intersection detected
        ix = p0[0] + (t * s1_x)
        iy = p0[1] + (t * s1_y)
        return (ix, iy)
    return None

def cast_ray(start, angle, edges):
    """
    Casts a ray from start at the given angle and returns the closest intersection point
    with any edge, or None if no intersection within RAY_LENGTH.
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


# ---------------------- MAIN SIMULATION LOOP ----------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("2D Robot Ray Casting")
    clock = pygame.time.Clock()

    running = True
    robot_pos = list(ROBOT_POSITION)

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Optional: Move robot with arrow keys
        keys = pygame.key.get_pressed()
        speed = 2
        if keys[pygame.K_UP]:
            robot_pos[1] -= speed
        if keys[pygame.K_DOWN]:
            robot_pos[1] += speed
        if keys[pygame.K_LEFT]:
            robot_pos[0] -= speed
        if keys[pygame.K_RIGHT]:
            robot_pos[0] += speed

        # Draw background
        screen.fill(BLACK)

        # Draw obstacles
        for poly in obstacles:
            pygame.draw.polygon(screen, GRAY, poly, 0)

        # Draw boundary
        pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 1)

        # Ray casting - 360 degrees
        hits = []
        for i in range(NUM_RAYS):
            angle = math.radians(i)
            hit_point = cast_ray((robot_pos[0], robot_pos[1]), angle, all_edges)
            if hit_point is not None:
                hits.append(hit_point)


        pygame.draw.polygon(screen, GREEN, hits)
        # for hp in hits:
            # pygame.draw.line(screen, GREEN, robot_pos, hp, 1)
            # pygame.draw.circle(screen, RED, (int(hp[0]), int(hp[1])), 2)

        # Draw robot
        r1_polygon = Polygon(hits)
        pygame_polygon_points = shapely_to_pygame_coords(r1_polygon)
        pygame.draw.polygon(screen, RED, pygame_polygon_points)
        pygame.draw.circle(screen, WHITE, (int(r1_polygon.centroid.x), int(r1_polygon.centroid.x)), ROBOT_RADIUS)
        pygame.draw.circle(screen, WHITE, (int(robot_pos[0]), int(robot_pos[1])), ROBOT_RADIUS)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
