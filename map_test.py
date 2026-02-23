import map

map_graph = map.Map(13,10)

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

map_graph.add_rect(1, 5, color='limegreen')
map_graph.add_rect(11, 5, color='orangered')

map_graph.show_graph()

print(map_graph.graph)