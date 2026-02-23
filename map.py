
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
        for x in range(0, size_x):
            for y in range(0, size_y):
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
        for i in self.graph[(x, y)]:
            self.graph[i].remove((x, y))
        self.graph.pop((x, y))
        return self.graph

    def add_rect(self, x, y, w=1, h=1, color='gray'):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=color))

    def add_point(self, x, y):
        self.ax.scatter(x + 0.5, y + 0.5)

    def add_edge(self, x1, y1, x2, y2):
        self.ax.plot([x1 + 0.5, x2 + 0.5],
                [y1 + 0.5, y2 + 0.5])

    def show_graph(self):
        plt.title("Map")
        plt.show()


    

    
