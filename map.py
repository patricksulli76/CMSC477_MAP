
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()

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
    graph.remove((x, y))
    return graph

def add_rect(x, y, w=1, h=1, color='gray'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color))

def add_point(x, y):
    ax.scatter(x + 0.5, y + 0.5)

def add_edge(x1, y1, x2, y2):
    ax.plot([x1 + 0.5, x2 + 0.5],
            [y1 + 0.5, y2 + 0.5])

for x in range(3, 10):
    add_rect(x, 8)
    add_obstacle(x, 8, graph)

for y in range(5, 8):
    add_rect(3, y)
    add_obstacle(3, y, graph)
for y in range(5, 8):
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
