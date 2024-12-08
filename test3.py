import math

import pygame

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
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# ---------------------- ENVIRONMENT -------------------------
obstacles = [
    # A rectangle obstacle
    [(200, 200), (300, 200), (300, 250), (200, 250)],
    # A triangle obstacle
    [(500, 100), (550, 200), (450, 200)],
    # Another polygon
    [(600, 400), (650, 450), (600, 500), (550, 450)],
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
    ((0, WINDOW_HEIGHT), (0, 0)),
]

all_edges = obstacle_edges + boundary


# ---------------------- RAY CASTING HELPER FUNCTIONS ----------------------
def line_intersection(p0, p1, p2, p3):
    """Returns the intersection point of line segment p0p1 and p2p3 if exists, otherwise None."""
    s1_x = p1[0] - p0[0]
    s1_y = p1[1] - p0[1]
    s2_x = p3[0] - p2[0]
    s2_y = p3[1] - p2[1]

    denom = -s2_x * s1_y + s1_x * s2_y
    if denom == 0:
        return None  # Parallel or coincident lines

    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / denom
    t = (s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / denom

    if 0 <= s <= 1 and 0 <= t <= 1:
        ix = p0[0] + (t * s1_x)
        iy = p0[1] + (t * s1_y)
        return (ix, iy)
    return None


def cast_ray(start, angle, edges):
    """Casts a ray from start at the given angle and returns the closest intersection point
    with any edge, or None if no intersection.
    """
    end_x = start[0] + RAY_LENGTH * math.cos(angle)
    end_y = start[1] + RAY_LENGTH * math.sin(angle)
    ray_end = (end_x, end_y)

    closest_point = None
    closest_dist = float("inf")

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
    """Given two sets of hit points (hits1 and hits2) from two robots,
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


# ---------------------- MAIN SIMULATION LOOP ----------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("2D Two-Robot Ray Casting")
    clock = pygame.time.Clock()

    running = True
    robot1_pos = list(ROBOT1_POSITION)
    robot2_pos = list(ROBOT2_POSITION)

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move robots with keys:
        # Robot 1: Arrow keys
        keys = pygame.key.get_pressed()
        speed = 2
        if keys[pygame.K_UP]:
            robot1_pos[1] -= speed
        if keys[pygame.K_DOWN]:
            robot1_pos[1] += speed
        if keys[pygame.K_LEFT]:
            robot1_pos[0] -= speed
        if keys[pygame.K_RIGHT]:
            robot1_pos[0] += speed

        # Robot 2: WASD keys
        if keys[pygame.K_w]:
            robot2_pos[1] -= speed
        if keys[pygame.K_s]:
            robot2_pos[1] += speed
        if keys[pygame.K_a]:
            robot2_pos[0] -= speed
        if keys[pygame.K_d]:
            robot2_pos[0] += speed

        # Ray casting
        hits1 = cast_all_rays((robot1_pos[0], robot1_pos[1]), all_edges, NUM_RAYS)
        hits2 = cast_all_rays((robot2_pos[0], robot2_pos[1]), all_edges, NUM_RAYS)

        # Resolve overlaps
        hits1, hits2 = remove_overlapping_points(
            robot1_pos, robot2_pos, hits1, hits2, OVERLAP_THRESHOLD
        )

        # Draw background
        screen.fill(BLACK)

        # Draw obstacles
        for poly in obstacles:
            pygame.draw.polygon(screen, GRAY, poly, 0)

        # Draw boundary
        pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 1)

        # Draw rays for Robot 1
        for hp in hits1:
            if hp is not None:
                pygame.draw.line(screen, GREEN, robot1_pos, hp, 1)
                pygame.draw.circle(screen, RED, (int(hp[0]), int(hp[1])), 2)

        # Draw rays for Robot 2
        for hp in hits2:
            if hp is not None:
                pygame.draw.line(screen, BLUE, robot2_pos, hp, 1)
                pygame.draw.circle(screen, RED, (int(hp[0]), int(hp[1])), 2)

        # Draw robots
        pygame.draw.circle(
            screen, WHITE, (int(robot1_pos[0]), int(robot1_pos[1])), ROBOT_RADIUS
        )
        pygame.draw.circle(
            screen, WHITE, (int(robot2_pos[0]), int(robot2_pos[1])), ROBOT_RADIUS
        )

        # Update display
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
