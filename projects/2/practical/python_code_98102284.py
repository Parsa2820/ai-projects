student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'

import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

def f_1(x):
    pass

def f_2(x):
    pass

def f_3(x):
    pass

def draw(func, x_range):
    pass

def gradiant_descent(func, initial_point: float, learning_rate: float, max_iterations: int):
    pass

def f(x_1, x_2):
    pass

def gradiant_descent(func, initial_point: Tuple, learning_rate: float, threshold: float, max_iterations: int):
    x_1_sequence = [initial_point[0]]
    x_2_sequence = [initial_point[1]]
    
    
    return x_1_sequence, x_2_sequence

def update_points(func, x_1, x_2, learning_rate):
    pass











def ac_3():
    pass

def backtrack():
    pass

def backtracking_search():
    return backtrack()



class MinimaxPlayer(Player):
    
    def __init__(self, col, x, y):
        super().__init__(col, x, y)

    def minValue(self, board, alpha, beta, depth):
        pass
    
    def maxValue(self, board, alpha, beta, depth):
        pass
    
    def getMove(self, board):
        pass

p1 = NaivePlayer(1, 0, 0)
p2 = NaivePlayer(2, 7, 7)
g = Game(p1, p2)
numberOfMatches = 1
score1, score2 = g.start(numberOfMatches)
print(score1/numberOfMatches)









