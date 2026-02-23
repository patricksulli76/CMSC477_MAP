import map

map_graph = map.Map(13,11)

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

for x in range(4, 9):
    map_graph.add_rect(x, 9)
    #map_graph.add_obstacle(x, 8)
for y in range(5, 10):
    map_graph.add_rect(3, y)
    #map_graph.add_obstacle(3, y)
for y in range(5, 10):
    map_graph.add_rect(9, y)
    #map_graph.add_obstacle(9, y)
for y in range(1, 6):
    map_graph.add_rect(6, y)
    #map_graph.add_obstacle(6, y)

map_graph.add_rect(1, 5, color='limegreen')
map_graph.add_rect(11, 5, color='orangered')

map_graph.show_graph()

print(map_graph.graph)