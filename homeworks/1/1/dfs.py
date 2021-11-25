from collections import defaultdict
from sys import maxsize

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, n, e, w):
        self.graph[n].append((e, w))

class GraphSearcher:
    def __init__(self, g):
        self.g = g
        self.visited = set()
        self.min_path = []
        self.min_path_weight = maxsize
    
    def dfs(self, start, goal):
        current_path = []
        self.dfs_aux(start, goal, current_path, 0)
        return (self.min_path, self.min_path_weight)

    def dfs_aux(self, current, goal, current_path, current_weight):
        if current in current_path:
            return
        current_path.append(current)
        if current == goal:
            if (current_weight < self.min_path_weight):
                self.min_path_weight = current_weight
                self.min_path = current_path[:]
        for n, w in self.g.graph[current]:
            self.dfs_aux(n, goal, current_path, current_weight+w)
        current_path.pop()

    
g = Graph()
g.add_edge('S', 'A', 1)
g.add_edge('S', 'B', 3)
g.add_edge('A', 'D', 2)
g.add_edge('B', 'D', 8)
g.add_edge('B', 'E', 5)
g.add_edge('C', 'A', 1)
g.add_edge('C', 'G1', 4)
g.add_edge('D', 'C', 5)
g.add_edge('D', 'G1', 14)
g.add_edge('D', 'G2', 6)
g.add_edge('E', 'F', 1)
g.add_edge('E', 'G2', 4)
g.add_edge('F', 'G2', 2)
g.add_edge('G1', 'G2', 0)

gs = GraphSearcher(g)
print(gs.dfs('S', 'G1'))
print(gs.dfs('S', 'G2'))