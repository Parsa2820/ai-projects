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

def board_str(board):
    result = ''
    result += '\n'
    result += str(board.players[0].getX()) + ' '
    result += str(board.players[0].getY()) + '\n'
    result += str(board.players[1].getX()) + ' '
    result += str(board.players[1].getY()) + '\n'
    n = board.getSize()
    for i in range(n):
        for j in range(n):
            result += str(board.getCell(i, j).getColor()) + ' '
        result += '\n'
    return result

from random import shuffle

class MinimaxPlayer(Player):
    def __init__(self, col, x, y, depth=4):
        super().__init__(col, x, y)
        self.moveF = [self.moveU, self.moveD, self.moveL, self.moveR,
                      self.moveUR, self.moveUL, self.moveDR, self.moveDL]
        self.depth = depth

    def moveU(self, x, y, board):
        if 0 <= x < board.getSize() and 0 <= y-1 < board.getSize() and board.getCell(x, y-1).getColor() == 0:
            return IntPair(x, y-1)
        return False

    def moveD(self, x, y, board):
        if 0 <= x < board.getSize() and 0 <= y+1 < board.getSize() and board.getCell(x, y+1).getColor() == 0:
            return IntPair(x, y+1)
        return False

    def moveR(self, x, y, board):
        if 0 <= x+1 < board.getSize() and 0 <= y < board.getSize() and board.getCell(x+1, y).getColor() == 0:
            return IntPair(x+1, y)
        return False

    def moveL(self, x, y, board):
        if 0 <= x-1 < board.getSize() and 0 <= y < board.getSize() and board.getCell(x-1, y).getColor() == 0:
            return IntPair(x-1, y)
        return False

    def moveUR(self, x, y, board):
        if 0 <= x+1 < board.getSize() and 0 <= y-1 < board.getSize() and board.getCell(x+1, y-1).getColor() == 0:
            return IntPair(x+1, y-1)
        return False

    def moveUL(self, x, y, board):
        if 0 <= x-1 < board.getSize() and 0 <= y-1 < board.getSize() and board.getCell(x-1, y-1).getColor() == 0:
            return IntPair(x-1, y-1)
        return False

    def moveDR(self, x, y, board):
        if 0 <= x+1 < board.getSize() and 0 <= y+1 < board.getSize() and board.getCell(x+1, y+1).getColor() == 0:
            return IntPair(x+1, y+1)
        return False

    def moveDL(self, x, y, board):
        if 0 <= x-1 < board.getSize() and 0 <= y+1 < board.getSize() and board.getCell(x-1, y+1).getColor() == 0:
            return IntPair(x-1, y+1)
        return False

    def adverseCol(self):
        if self.getCol() == 1:
            return 2
        return 1

    def is_adverse_place(self, board: Board, place, col):
        adverse = board.players[1 - col]
        return adverse.getX() == place.x and adverse.getY() == place.y

    def canMove(self, x, y, board, col):
        all_directions = []
        shuffle(self.moveF)
        for f in self.moveF:
            can_move_f = f(x, y, board)
            if can_move_f and (not self.is_adverse_place(board, can_move_f, col)):
                all_directions.append(can_move_f)
        return all_directions if len(all_directions) > 0 else False

    def minValue(self, board, alpha, beta, depth):
        if depth == 0:
            return board.getScore(self.getCol())
        col = self.adverseCol()
        can_move = self.canMove(board.getPlayerX(col), board.getPlayerY(col), board, col)
        min_val = float('+inf')
        if not can_move:
            return -1
        else:
            for move in can_move:
                board_clone = Board(board)
                board_clone.move(move, col)
                val = self.maxValue(board_clone, alpha, beta, depth-1)
                if val < min_val:
                    min_val = val
                if min_val <= alpha:
                    return min_val
                if min_val < beta:
                    beta = min_val
        return min_val

    def maxValue(self, board: Board, alpha, beta, depth):
        if depth == 0:
            return board.getScore(self.getCol())
        can_move = self.canMove(board.getPlayerX(self.getCol()), board.getPlayerY(self.getCol()), board, self.getCol())
        max_val = float('-inf')
        if not can_move:
            return -1
        else:
            for move in can_move:
                board_clone = Board(board)
                board_clone.move(move, self.getCol())
                val = self.minValue(board_clone, alpha, beta, depth-1)
                if val > max_val:
                    max_val = val
                if max_val >= beta:
                    return max_val
                if max_val > alpha:
                    alpha = max_val
        return max_val

    def getMove(self, board):
        alpha = float('-inf')
        beta = float('inf')
        next = IntPair(-20, -20)

        can_move = self.canMove(board.getPlayerX(self.getCol()), board.getPlayerY(self.getCol()), board, self.getCol())

        start = time.time()
        if not can_move:
            return IntPair(-10, -10)
        else:
            max_val = float('-inf')
            for move in can_move:
                board_clone = Board(board)
                board_clone.move(move, self.getCol())
                val = self.minValue(board_clone, alpha, beta, self.depth-1)
                if val > max_val:
                    max_val = val
                    next = move

        if (time.time() - start > 2):
            return IntPair(-10, -10)
        with open('log.txt', 'a') as f:
            f.write(board_str(board))
            f.write('harekat dadam ' + str(next.x) + ' ' + str(next.y) + '\n')
        return next

with open("log.txt", "w") as log:
    log.write("")
p1 = MinimaxPlayer(1, 0, 0, 4)
p2 = NaivePlayer(2, 7, 7)
g = Game(p1, p2)
numberOfMatches = 1
score1, score2 = g.start(numberOfMatches)
print(score1/numberOfMatches)

import matplotlib.pyplot as plt

numberOfRepeats = 10
numberOfMatches = 4
x = list(range(numberOfMatches+1))
y = [0 for i in range(numberOfMatches+1)]

for i in range(numberOfRepeats):
    p1 = NaivePlayer(1, 0, 0)
    p2 = MinimaxPlayer(2, 7, 7)
    g = Game(p1, p2)
    score1, score2 = g.start(numberOfMatches)
    y[int(score1)] += 1
    print(f'{i}th iteration done')

plt.plot(x, y)
plt.show()

numberOfRepeats = 10
numberOfMatches = 4
x = list(range(numberOfMatches+1))
y = [0 for i in range(numberOfMatches+1)]

for i in range(numberOfRepeats):
    p1 = MinimaxPlayer(1, 0, 0)
    p2 = MinimaxPlayer(2, 7, 7)
    g = Game(p1, p2)
    score1, score2 = g.start(numberOfMatches)
    y[int(score1)] += 1
    print(f'{i}th iteration done')

plt.plot(x, y)
plt.show()

depth = 5
numberOfMatches = 4
x = list(range(1, depth))
y = [0 for i in range(1, depth)]

for i in range(1, depth):
    p1 = NaivePlayer(1, 0, 0)
    p2 = MinimaxPlayer(2, 7, 7, i)
    g = Game(p1, p2)
    score1, score2 = g.start(numberOfMatches)
    y[i-1] = score1
    print(f'{i}th iteration done')

plt.plot(x, y)
plt.show()

numberOfRepeats = 10
numberOfMatches = 4
x = list(range(numberOfMatches+1))
y = [0 for i in range(numberOfMatches+1)]

for i in range(numberOfRepeats):
    p1 = MinimaxPlayer(1, 0, 0, 4)
    p2 = MinimaxPlayer(2, 7, 7, 2)
    g = Game(p1, p2)
    score1, score2 = g.start(numberOfMatches)
    y[int(score1)] += 1
    print(f'{i}th iteration done')

plt.plot(x, y)
plt.show()

