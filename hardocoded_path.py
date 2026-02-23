
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()

# AprilTag ID -> (x, y) of the obstacle grid cell’s lower-left corner
# (Multiple tags can share the same (x,y) because they’re mounted on the same obstacle square.)

apriltag_to_grid = {
    30: (3, 8, "L"),
    31: (3, 8, "R"),
    32: (3, 6, "L"),
    33: (3, 6, "R"),
    34: (3, 5, "D"),
    35: (5, 9, "D"),
    36: (7, 9, "D"),
    37: (6, 5, "U"),
    38: (6, 4, "L"),
    39: (6, 4, "R"),
    40: (6, 2, "L"),
    41: (6, 2, "R"),
    42: (9, 8, "L"),
    43: (9, 8, "R"),
    44: (9, 6, "L"),
    45: (9, 6, "R"),
    46: (9, 5, "D"),
}

graph = {}
size_x = 13
size_y = 10
for x in range(0, size_x):
    for y in range(0, size_y):
        graph[(x, y)] = []
        if x > 0:
            graph[(x, y)].append((x - 1, y))
        if x < size_x - 1:
            graph[(x, y)].append((x + 1, y))
        if y > 0:
            graph[(x, y)].append((x, y - 1))
        if y < size_y - 1:
            graph[(x, y)].append((x, y + 1))
        if x > 0 and y > 0:
            graph[(x, y)].append((x - 1, y - 1))
        if x < size_x - 1 and y > 0:
            graph[(x, y)].append((x + 1, y - 1))
        if x > 0 and y < size_y - 1:
            graph[(x, y)].append((x - 1, y + 1))
        if x < size_x - 1 and y < size_y - 1:
            graph[(x, y)].append((x + 1, y + 1))

def add_obstacle(x, y, graph):
    for i in graph[(x, y)]:
        graph[i].remove((x, y))
    graph.pop((x, y))
    return graph

def add_rect(x, y, w=1, h=1, color='gray'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color))

def add_point(x, y):
    ax.scatter(x + 0.5, y + 0.5)

def add_edge(x1, y1, x2, y2):
    ax.plot([x1 + 0.5, x2 + 0.5],
            [y1 + 0.5, y2 + 0.5], color='blue')

for x in range(3, 10):
    add_rect(x, 8)
    add_obstacle(x, 8, graph)

for y in range(4, 8):
    add_rect(3, y)
    add_obstacle(3, y, graph)
for y in range(4, 8):
    add_rect(9, y)
    add_obstacle(9, y, graph)
for y in range(1, 6):
    add_rect(6, y)
    add_obstacle(6, y, graph)

add_rect(1, 5, color='limegreen')
add_rect(11, 5, color='orangered')

ax.set_xlim(0, 13)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_xticks(range(0, 14))
ax.set_yticks(range(0, 11))
ax.grid(True)



plt.title("Map")
plt.show()
