import pygame
import sys
import os

# Setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Street Loop Navigation")
clock = pygame.time.Clock()

# Load background and sticker
script_dir = os.path.dirname(os.path.abspath(__file__))
background = pygame.image.load(os.path.join(script_dir, "map.png")).convert()
sticker_img = pygame.image.load(os.path.join(script_dir, "car.png")).convert_alpha()
sticker_original = pygame.transform.scale(sticker_img, (sticker_img.get_width() // 7.5, sticker_img.get_height() // 7.5))

# Direction-based rotation
def get_rotated_sticker(direction):
    if direction == 'up':
        return sticker_original
    elif direction == 'left':
        return pygame.transform.rotate(sticker_original, 90)
    elif direction == 'down':
        return pygame.transform.rotate(sticker_original, 180)
    elif direction == 'right':
        return pygame.transform.rotate(sticker_original, -90)
    return sticker_original

# Waypoint format: (x, y, direction)
loop1 = [
    (433, 230, 'left'),    # street1 end → street2
    (114, 230, 'down'),    # street2 end → street3
    (114, 403, 'right'),   # street3 end → street4
    (433, 403, 'up'),      # street4 end → street1 start
]

loop2 = [
    (433, 315, 'left'),    # halfway of street1 to street5
    (114, 315, 'down'),    # street5 to halfway of street3
    (114, 403, 'right'),   # to street4
    (433, 403, 'up'),      # back to start of street1
]

transition = [(433, 315, 'up')]  # smooth move from (433, 403) to (433, 315)

# Initial state
x, y = 433, 403
speed = 2.5
direction = 'up'
current_loop = 0  # 0 for loop1, 1 for loop2
loop_data = [loop1, loop2]
waypoints = loop1
current_wp = 0
in_transition = False

# Main loop
running = True
while running:
    screen.blit(background, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        target_x, target_y, next_dir = waypoints[current_wp]

        # Move in current direction
        if direction == 'up' and y > target_y:
            y -= speed
        elif direction == 'down' and y < target_y:
            y += speed
        elif direction == 'left' and x > target_x:
            x -= speed
        elif direction == 'right' and x < target_x:
            x += speed
        else:
            # Reached waypoint
            direction = next_dir
            current_wp += 1

            if current_wp >= len(waypoints):
                # If coming from loop1, insert transition before loop2
                if not in_transition and current_loop == 0:
                    waypoints = transition
                    current_wp = 0
                    in_transition = True
                elif in_transition:
                    # Now switch to loop2
                    current_loop = 1
                    waypoints = loop2
                    current_wp = 0
                    in_transition = False
                    direction = 'left'
                else:
                    # Completed loop2, back to loop1
                    current_loop = 0
                    waypoints = loop1
                    current_wp = 0
                    direction = 'up'
                    x, y = 433, 403

    # Draw car
    sticker = get_rotated_sticker(direction)
    rect = sticker.get_rect(center=(x, y))
    screen.blit(sticker, rect)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
