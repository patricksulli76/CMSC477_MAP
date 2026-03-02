import math
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ==========================
# Map Class
# ==========================
class Map:

    def __init__(self, size_x, size_y):
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim(0, size_x)
        self.ax.set_ylim(0, size_y)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(0, size_x+1))
        self.ax.set_yticks(range(0, size_y+1))
        self.ax.grid(True)

        self.graph = {}

        for x in range(size_x):
            for y in range(size_y):
                self.graph[(x, y)] = []

                # 8-connected grid
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue

                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size_x and 0 <= ny < size_y:
                            self.graph[(x, y)].append((nx, ny))

    def add_obstacle(self, x, y):
        if (x, y) not in self.graph:
            return

        for neighbor in self.graph[(x, y)]:
            if neighbor in self.graph:
                if (x, y) in self.graph[neighbor]:
                    self.graph[neighbor].remove((x, y))

        self.graph.pop((x, y))

    def add_rect(self, x, y, w=1, h=1, color='gray'):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=color))

    def add_edge(self, x1, y1, x2, y2):
        self.ax.plot([x1 + 0.5, x2 + 0.5],
                     [y1 + 0.5, y2 + 0.5])

    def show(self):
        plt.title("A* Path Planning")
        plt.show()


# ==========================
# A* Algorithm
# ==========================
def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(graph, start, goal):

    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_score = {node: float('inf') for node in graph}
    f_score = {node: float('inf') for node in graph}

    g_score[start] = 0
    f_score[start] = heuristic(start, goal)

    closed_set = set()

    while open_list:

        _, current = heapq.heappop(open_list)

        if current == goal:
            print("Destination Found!")
            return reconstruct_path(came_from, current)

        closed_set.add(current)

        for neighbor in graph[current]:

            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + 1

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    print("No Path Found")
    return None


# ==========================
# MAIN EXECUTION
# ==========================
def main():

    map_graph = Map(13, 10)

    # Add Obstacles
    for x in range(3, 10):
        map_graph.add_rect(x, 8)
        map_graph.add_obstacle(x, 8)

    for y in range(5, 8):
        map_graph.add_rect(3, y)
        map_graph.add_obstacle(3, y)

    for y in range(5, 8):
        map_graph.add_rect(9, y)
        map_graph.add_obstacle(9, y)

    for y in range(1, 6):
        map_graph.add_rect(6, y)
        map_graph.add_obstacle(6, y)

    start = (1, 5)
    goal = (11, 5)

    # Draw start and goal
    map_graph.add_rect(*start, color='limegreen')
    map_graph.add_rect(*goal, color='orangered')

    # Run A*
    path = a_star(map_graph.graph, start, goal)

    # Draw Path
    if path:
        for i in range(len(path)-1):
            map_graph.add_edge(path[i][0], path[i][1],
                               path[i+1][0], path[i+1][1])

    map_graph.show()


if __name__ == "__main__":
    main()