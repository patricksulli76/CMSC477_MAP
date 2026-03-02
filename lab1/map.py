
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

class Map:

    def __init__(self, size_x, size_y,start, finish):
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim(0, size_x)
        self.ax.set_ylim(0, size_y)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(0, size_x+1))
        self.ax.set_yticks(range(0, size_y+1))
        self.ax.grid(True)
        self.graph = {}
        self.graph_heuristic = {}
        for x in range(0, size_x):
            for y in range(0, size_y):
                distance = math.hypot((finish[0] - x)**2,(finish[1] - y)**2)
                self.graph_heuristic[(x,y)] = distance
                self.graph[(x, y)] = []
                if x > 0:
                    self.graph[(x, y)].append((x - 1, y))
                if x < size_x - 1:
                    self.graph[(x, y)].append((x + 1, y))
                if y > 0:
                    self.graph[(x, y)].append((x, y - 1))
                if y < size_y - 1:
                    self.graph[(x, y)].append((x, y + 1))
                if x > 0 and y > 0:
                    self.graph[(x, y)].append((x - 1, y - 1))
                if x < size_x - 1 and y > 0:
                    self.graph[(x, y)].append((x + 1, y - 1))
                if x > 0 and y < size_y - 1:
                    self.graph[(x, y)].append((x - 1, y + 1))
                if x < size_x - 1 and y < size_y - 1:
                    self.graph[(x, y)].append((x + 1, y + 1))

    def add_obstacle(self, x, y):
        self.graph_heuristic[(x,y)] = math.inf
        for i in self.graph[(x, y)]:
            self.graph[i].remove((x, y))
        self.graph.pop((x, y))
        return self.graph

    def add_rect(self, x, y, w=1, h=1, color='gray'):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=color))

    def add_point(self, x, y):
        self.ax.scatter(x , y )

    def remove_last_point(self):
        # Get all collections (scatter plots) from the axes
        collections = self.ax.collections
        if collections:
            # Remove the last scatter plot collection
            try:
                collections[-1].remove()
            except:
                pass

    def add_edge(self, x1, y1, x2, y2):
        self.ax.plot([x1 + 0.5, x2 + 0.5],
                [y1 + 0.5, y2 + 0.5])

    def show_graph(self):
        plt.title("Map")
        plt.show()


    

    
