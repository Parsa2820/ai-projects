student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'

import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

def f_1(x):
    term0 = np.multiply(np.power(x, 2), np.cos(np.divide(x, 10)))
    term1 = np.subtract(term0, x)
    return np.divide(term1, 100)

def f_2(x):
    term0 = np.sqrt(np.sin(np.divide(x, 20)))
    return np.log(term0)

def f_3(x):
    term0 = np.add(np.cos(x), np.divide(45, x))
    return np.log(term0)

def draw(func, x_range):
    y = func(x_range)
    print(np.max(y))
    print(x_range[np.argmax(y)])
    plt.plot(x_range, y)
    plt.show()

def gradiant_descent(func, initial_point: float, learning_rate: float, max_iterations: int):
    h = 10e-10
    x = initial_point
    for i in range(max_iterations):
        y = func(x)
        dy = np.divide(np.subtract(func(np.add(x, h)), y), h)
        if dy == 0:
            return y
        x = np.subtract(x, np.multiply(learning_rate, dy))
    return func(x)

def f(x_1, x_2):
    term0 = np.multiply(2, np.power(x_1, 2))
    term1 = np.multiply(3, np.power(x_2, 2))
    term2 = np.multiply(-4, np.multiply(x_1, x_2))
    term3 = np.multiply(-50, x_1)
    term4 = np.multiply(6, x_2)
    return np.add(term0, np.add(term1, np.add(term2, np.add(term3, term4))))

def gradiant_descent(func, initial_point: Tuple, learning_rate: float, threshold: float, max_iterations: int):
    x_1_sequence = [initial_point[0]]
    x_2_sequence = [initial_point[1]]
    
    for i in range(max_iterations):
        x_1, x_2 = update_points(func, x_1_sequence[-1], x_2_sequence[-1], learning_rate)
        if x_1 > threshold or x_2 > threshold:
            break
        x_1_sequence.append(x_1)
        x_2_sequence.append(x_2)

    return x_1_sequence, x_2_sequence

def update_points(func, x_1, x_2, learning_rate):
    h = 10e-10
    y = func(x_1, x_2)
    dy_x1 = np.divide(np.subtract(func(np.add(x_1, h), x_2), y), h)
    dy_x2 = np.divide(np.subtract(func(x_1, np.add(x_2, h)), y), h)
    x_1 = np.subtract(x_1, np.multiply(learning_rate, dy_x1))
    x_2 = np.subtract(x_2, np.multiply(learning_rate, dy_x2))
    return x_1, x_2

x_1_sequence, x_2_sequence = gradiant_descent(
    f, initial_point, learning_rates[0], threshold, max_iterations)
draw_points_sequence(f, x_1_sequence, x_2_sequence)

x_1_sequence, x_2_sequence = gradiant_descent(
    f, initial_point, learning_rates[1], threshold, max_iterations)
draw_points_sequence(f, x_1_sequence, x_2_sequence)

x_1_sequence, x_2_sequence = gradiant_descent(
    f, initial_point, learning_rates[2], threshold, max_iterations)
draw_points_sequence(f, x_1_sequence, x_2_sequence)

x_1_sequence, x_2_sequence = gradiant_descent(
    f, initial_point, learning_rates[3], threshold, max_iterations)
draw_points_sequence(f, x_1_sequence, x_2_sequence)

import copy


class CspProperties:
    def __init__(self, domains, edges, halls_count) -> None:
        self.domains = domains
        self.edges = edges
        self.halls_count = halls_count
        self.neighbors = self.get_neighbors()

    def parse_csp_properties(halls_count, department_count, prefered_halls, e, next_e_lines):
        domains = {i: [] for i in range(1, halls_count + 1)}
        for index, halls in enumerate(prefered_halls, start=1):
            for hall in halls:
                domains[hall].append(index)
        edges = []
        for edge in next_e_lines:
            edges.append((int(edge[0]), int(edge[1])))
        return CspProperties(domains, edges, halls_count)

    def get_neighbors(self):
        neighbours = {i: [] for i in range(1, self.halls_count + 1)}
        for edge in self.edges:
            neighbours[edge[0]].append(edge[1])
            neighbours[edge[1]].append(edge[0])
        return neighbours

    def constraint_satisfied(self, assignments):
        for e1, e2 in self.edges:
            if assignments[e1] == assignments[e2]:
                return False
        return True

    def select_unassigned_variable(self, assignments):
        for i in range(1, self.halls_count + 1):
            if i not in assignments.keys():
                return i

    def order_domain_values(self, assignments, variable):
        return self.domains[variable]

    def clone(self):
        return copy.deepcopy(self)


def ac_3(csp_properties: CspProperties, assignments):
    queue = csp_properties.edges[:]
    while queue:
        e1, e2 = queue.pop(0)
        if revise(csp_properties, e1, e2):
            if not csp_properties.domains[e1]:
                return False
            for e in csp_properties.neighbors[e1]:
                if e != e2:
                    queue.append((e1, e))
    return True

def revise(csp_properties, e1, e2):
    revised = False
    for i in csp_properties.domains[e1]:
        if len(csp_properties.domains[e2]) == 1 and csp_properties.domains[e2] == i:
            csp_properties.domains[e1].remove(i)
            revised = True
    return revised

def backtrack(csp_properties: CspProperties, assignments):
    if len(assignments) == csp_properties.halls_count:
        if csp_properties.constraint_satisfied(assignments):
            return assignments
        return
    variable = csp_properties.select_unassigned_variable(assignments)
    for value in csp_properties.order_domain_values(assignments, variable):
        assignments[variable] = value
        _csp_properties = csp_properties.clone()
        if ac_3(_csp_properties, assignments):
            result = backtrack(_csp_properties, assignments)
            if result:
                return result
        del assignments[variable]
    return

def backtracking_search(csp_properties):
    assignments = backtrack(csp_properties, {})
    assignments_str = ''
    if not assignments:
        assignments_str = 'NO'
    else:
        assignments_ordered = []
        for i in range(1, csp_properties.halls_count + 1):
            assignments_ordered.append(assignments[i])
        assignments_str = ' '.join(map(str, assignments_ordered))
    return assignments_str



from functools import reduce


class MinimaxPlayer(Player):
    def __init__(self, col, x, y, depth=-1):
        super().__init__(col, x, y)
        self.moveF = [self.moveU, self.moveD, self.moveL, self.moveR,
                      self.moveUR, self.moveUL, self.moveDR, self.moveDL]
        self.depth = depth

    def moveU(self, x, y, board):
        if 0 < x < board.getSize() and 0 < y-1 < board.getSize() and board.__cells[x][y-1].getColor() != 0:
            return IntPair(x, y-1)
        return False

    def moveD(self, x, y, board):
        if 0 < x < board.getSize() and 0 < y+1 < board.getSize() and board.__cells[x][y+1].getColor() != 0:
            return IntPair(x, y+1)
        return False

    def moveR(self, x, y, board):
        if 0 < x+1 < board.getSize() and 0 < y < board.getSize() and board.__cells[x+1][y].getColor() != 0:
            return IntPair(x+1, y)
        return False

    def moveL(self, x, y, board):
        if 0 < x-1 < board.getSize() and 0 < y < board.getSize() and board.__cells[x-1][y].getColor() != 0:
            return IntPair(x-1, y)
        return False

    def moveUR(self, x, y, board):
        if 0 < x+1 < board.getSize() and 0 < y-1 < board.getSize() and board.__cells[x+1][y-1].getColor() != 0:
            return IntPair(x+1, y-1)
        return False

    def moveUL(self, x, y, board):
        if 0 < x-1 < board.getSize() and 0 < y-1 < board.getSize() and board.__cells[x-1][y-1].getColor() != 0:
            return IntPair(x-1, y-1)
        return False

    def moveDR(self, x, y, board):
        if 0 < x+1 < board.getSize() and 0 < y+1 < board.getSize() and board.__cells[x+1][y+1].getColor() != 0:
            return IntPair(x+1, y+1)
        return False

    def moveDL(self, x, y, board):
        if 0 < x-1 < board.getSize() and 0 < y+1 < board.getSize() and board.__cells[x-1][y+1].getColor() != 0:
            return IntPair(x-1, y+1)
        return False

    def canMove(self, x, y, board):
        all_directions = [f(x, y, board) for f in self.moveF]
        return (reduce(lambda a, b: a or b, all_directions, False), all_directions)

    def minValue(self, board, alpha, beta, depth):
        pass

    def maxValue(self, board, alpha, beta, depth):
        pass

    def getMove(self, board: Board, col=self.getCol()):
        alpha = float('-inf')
        beta = float('inf')
        next = IntPair(-20, -20)

        can_move, moves = self.canMove(board.getPlayerX(self.getCol()), board.getPlayerY(self.getCol()), board)

        if not can_move:
            return IntPair(-10, -10)
        else:
            v = float('-inf')
            for move in moves:
                self.maxValue(board )
        
        return next

        x_next = self.getX()
        y_next = self.getY()
        start = time.time()
        while ((x_next == self.getX()) and (y_next == self.getY())):
            rnd = random.randrange(4)
            if (time.time() - start > 2):
                return IntPair(-10, -10)
            if ((rnd == 0) and (self.getX() + 1 < board.getSize()) and (board.getCell(self.getX() + 1, self.getY()).getColor() == 0)):
                x_next += 1
            elif ((rnd == 1) and (self.getX() - 1 >= 0) and (board.getCell(self.getX() - 1, self.getY()).getColor() == 0)):
                x_next -= 1
            elif ((rnd == 2) and (self.getY() + 1 < board.getSize()) and (board.getCell(self.getX(), self.getY() + 1).getColor() == 0)):
                y_next += 1
            elif ((rnd == 3) and (self.getY() - 1 >= 0) and (board.getCell(self.getX(), self.getY() - 1).getColor() == 0)):
                y_next -= 1

        return IntPair(x_next, y_next)


p1 = NaivePlayer(1, 0, 0)
p2 = NaivePlayer(2, 7, 7)
g = Game(p1, p2)
numberOfMatches = 1
score1, score2 = g.start(numberOfMatches)
print(score1/numberOfMatches)









