student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'

from collections import defaultdict
import itertools


MIN_COST = 0


def find_neighbours(roads):
    neighbours = defaultdict(list)
    for road in roads:
        neighbours[road[0]].append(tuple(road[1:]))
        neighbours[road[1]].append((road[0], road[2]))
    return neighbours


def double_backtrack(N, K, NUMS, neighbours, currents, costs, collected_flower_types, visited_firsts, visited_seconds, parents):
    _visited_firsts = visited_firsts[0].copy(), visited_firsts[1].copy()
    _visited_seconds = visited_seconds[0].copy(), visited_seconds[1].copy()
    if currents[0] in _visited_seconds[0] or currents[1] in _visited_seconds[1]:
        return
    if currents[0] in _visited_firsts[0]:
        _visited_seconds[0].add(currents[0])
    if currents[1] in _visited_firsts[1]:
        _visited_seconds[1].add(currents[1])
    _visited_firsts[0].add(currents[0])
    _visited_firsts[1].add(currents[1])
    current_collected_flower_types = collected_flower_types.copy()
    current_collected_flower_types.update(NUMS[currents[0]-1][1:])
    current_collected_flower_types.update(NUMS[currents[1]-1][1:])
    if currents == (N, N) and len(current_collected_flower_types) == K:
        global MIN_COST
        if max(costs[0], costs[1]) < MIN_COST:
            MIN_COST = max(costs[0], costs[1])
        return
    all_combinations = itertools.product(
        neighbours[currents[0]], neighbours[currents[1]])
    for neighbour1, neighbour2 in all_combinations:
        neighbour1_node, cost1 = neighbour1
        neighbour2_node, cost2 = neighbour2
        if neighbour1_node == parents[0] or neighbour1_node == parents[1]:
            continue
        if currents[0] == N:
            double_backtrack(N, K, NUMS, neighbours,
                             (currents[0], neighbour2_node),
                             (costs[0], costs[1]+cost2),
                             current_collected_flower_types,
                             _visited_firsts, _visited_seconds, currents)
        if currents[1] == N:
            double_backtrack(N, K, NUMS, neighbours,
                             (neighbour1_node, currents[1]),
                             (costs[0]+cost1, costs[1]),
                             current_collected_flower_types,
                             _visited_firsts, _visited_seconds, currents)
        double_backtrack(N, K, NUMS, neighbours,
                         (neighbour1_node, neighbour2_node),
                         (costs[0]+cost1, costs[1]+cost2),
                         current_collected_flower_types,
                         _visited_firsts, _visited_seconds, currents)




def solve(N, M, K, NUMS, roads):
    global MIN_COST
    MIN_COST = float('inf')
    neighbours = find_neighbours(roads)
    double_backtrack(N, K, NUMS, neighbours, (1, 1), (0, 0),
                     set(), (set(), set()), (set(), set()), (-1, -1))
    return MIN_COST




def heur_displaced(state):
    heur = 0
    for box_cordinate, box_index in state.boxes.items():
        if box_cordinate not in state.storage:
            heur += 1
        elif not state.restrictions:
            heur += 1
        elif (box_index, state.storage[box_cordinate]) not in state.restrictions:
            heur += 1
    return heur


def heur_manhattan_distance(state):
    heur = 0
    for box_cordinate in state.boxes.keys():
        min = float('inf')
        for storage_cordinate in state.storage.keys():
            distance = abs(box_cordinate[0] - storage_cordinate[0]) + abs(box_cordinate[1] - storage_cordinate[1])
            if distance < min:
                min = distance
        heur += min
    return heur
        

def heur_euclidean_distance(state):  
    heur = 0
    for box_cordinate in state.boxes.keys():
        min = float('inf')
        for storage_cordinate in state.storage.keys():
            distance = math.sqrt((box_cordinate[0] - storage_cordinate[0])**2 + (box_cordinate[1] - storage_cordinate[1])**2)
            if distance < min:
                min = distance
        heur += min
    return heur

import matplotlib.pyplot as plt

heur_displaced_result = [(23, 0.0625, 2555, 6382, 3827, 0), (35, 0.21875, 10980, 25891, 14911, 0), (27, 0.34375, 12620, 31102, 18482, 0),
                         (20, 0.828125, 33932, 77601, 43669, 0), (41, 0.375, 16607, 40003, 23396, 0), (41, 0.359375, 16607, 40003, 23396, 0)]
huer_manhattan_distance_result = [(23, 0.0625, 2209, 5367, 3158, 0), (35, 0.1875, 8289, 19091, 10802, 0), (27, 0.265625, 9472, 22768, 13296, 0),
                                  (20, 0.265625, 8876, 18584, 9708, 0), (41, 0.609375, 14254, 34057, 19803, 0), (41, 0.359375, 14254, 34057, 19803, 0)]
heur_euclidean_distance_result = [(23, 0.0625, 2245, 5540, 3295, 0), (35, 0.40625, 8670, 20266, 11596, 0), (27, 0.296875, 10518, 25910, 15392, 0),
                                  (20, 0.484375, 14021, 29041, 15020, 0), (41, 0.46875, 14513, 34471, 19958, 0), (41, 0.4375, 14513, 34471, 19958, 0)]

result = np.asarray([heur_displaced_result, huer_manhattan_distance_result, heur_euclidean_distance_result])

plt.figure(figsize=(15, 30))

plt.subplot(6, 1, 1)
x = np.arange(0, 6)
y1 = result[0, :, 0]
y2 = result[1, :, 0]
y3 = result[2, :, 0]
plt.plot(x, y1, label='heur_displaced')
plt.plot(x, y2, label='heur_manhattan_distance')
plt.plot(x, y3, label='heur_euclidean_distance')
plt.title('Problem Cost')
plt.legend()

plt.subplot(6, 1, 2)
y1 = result[0, :, 1]
y2 = result[1, :, 1]
y3 = result[2, :, 1]
plt.plot(x, y1, label='heur_displaced')
plt.plot(x, y2, label='heur_manhattan_distance')
plt.plot(x, y3, label='heur_euclidean_distance')
plt.title('Problem Time')
plt.legend()

plt.subplot(6, 1, 3)
y1 = result[0, :, 2]
y2 = result[1, :, 2]
y3 = result[2, :, 2]
plt.plot(x, y1, label='heur_displaced')
plt.plot(x, y2, label='heur_manhattan_distance')
plt.plot(x, y3, label='heur_euclidean_distance')
plt.title('Problem Nodes Expanded')
plt.legend()

plt.subplot(6, 1, 4)
y1 = result[0, :, 3]
y2 = result[1, :, 3]
y3 = result[2, :, 3]
plt.plot(x, y1, label='heur_displaced')
plt.plot(x, y2, label='heur_manhattan_distance')
plt.plot(x, y3, label='heur_euclidean_distance')
plt.title('Problem States Generated')
plt.legend()

plt.subplot(6, 1, 5)
y1 = result[0, :, 4]
y2 = result[1, :, 4]
y3 = result[2, :, 4]
plt.plot(x, y1, label='heur_displaced')
plt.plot(x, y2, label='heur_manhattan_distance')
plt.plot(x, y3, label='heur_euclidean_distance')
plt.title('Problem States Cycle Check Pruned')
plt.legend()

plt.subplot(6, 1, 6)
y1 = result[0, :, 5]
y2 = result[1, :, 5]
y3 = result[2, :, 5]
plt.plot(x, y1, label='heur_displaced')
plt.plot(x, y2, label='heur_manhattan_distance')
plt.plot(x, y3, label='heur_euclidean_distance')
plt.title('Problem States Cost Bound Pruned')
plt.legend()

plt.show()


import time

def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10):
    best_path_cost = float("inf")
    time_remain = 8
    iter = 0

    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    se = SearchEngine('custom', 'full')
    se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)

    while (time_remain > 0) and not se.open.empty():
        iter_start_time = time.time()
        if iter == 0 or not optimal_final:
            final = se.search(timebound)
            if final:
                optimal_final = final
                best_path_cost = final.gval
        else:
            g = optimal_final.gval
            costbound = (g, 0, g + 0)
            final = se.search(timebound, costbound)
            if final and final.gval < best_path_cost:
                optimal_final = final
                best_path_cost = final.gval
        iter += 1
        elapsed_time = time.time() - iter_start_time
        time_remain -= elapsed_time
    try:
        return optimal_final
    except:
        return final

    return False





edge_count = np.count_nonzero(graph_matrix) # Complete This (1 Points)
print(edge_count)

def random_state_generator(n):
    return [random.getrandbits(1) for _ in range(n)]

def neighbour_state_generator(state):
    new_state = state.copy()
    vertex_to_change = random.randint(0, len(new_state) - 1)
    previous_value = new_state[vertex_to_change]
    new_state[vertex_to_change] = int(not previous_value)
    return new_state, previous_value, vertex_to_change

def cost_function(graph_matrix, state, A=1, B=1):
    cost = 0
    cost += A * np.count_nonzero(state)
    tmp = 0
    for i in range(len(state)):
        for j in range(len(state)):
            if state[i] == state[j] == 0:
                tmp += graph_matrix[i][j]
    cost += B * tmp
    return cost

deg = [np.count_nonzero(node_adjacents) / edge_count for node_adjacents in graph_matrix] #Complete This (2 Points)

def prob_accept(si, i, cost_new, cost_old, T): 
    if si:
        return math.exp(-1 * (cost_new - cost_old) * (1 - deg[i]) / T)
    else:
        return math.exp(-1 * (cost_new - cost_old) * (1 + deg[i]) / T)

def accept(current_state , next_state , i, T):
    cost_old = cost_function(graph_matrix, current_state)
    cost_new = cost_function(graph_matrix, next_state)
    if cost_new < cost_old:
        return True
    else:
        return prob_accept(current_state[i], i, cost_new, cost_old, T) > random.random()

def plot_cost(cost_list):
    x = range(len(cost_list))
    plt.figure(figsize=(7, 5))
    plt.plot(x, cost_list)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()

plot_cost(cost_list)



def population_generation(n, k): 
    return [random_state_generator(n) for _ in range(k)]

def cost_function2(graph, state):
    exist_node_cost = 1
    non_exist_edge_cost = 5
    cost = np.count_nonzero(state) * exist_node_cost
    edges = np.nonzero(np.triu(graph))
    for edge in edges:
        if state[edge[0]] == state[edge[1]] == 0:
            cost += non_exist_edge_cost
    return cost

def tournament_selection(graph, population):
    population_size = len(population)
    new_population = []
    step = random.randint(1, population_size - 1)
    for i in range(population_size//step):
        for j in range(step*i, step*(i+1)):
            rival1 = population[j]
            rival2 = population[(j + step) % population_size]
            if cost_function2(graph, rival1) < cost_function2(graph, rival2):
                new_population.append(rival1)
            else:
                new_population.append(rival2)
    return new_population[0:50]

def crossover(graph, parent1, parent2):
    n = len(parent1)
    index = random.randint(0, n - 1)
    child1 = parent1[:index] + parent2[index:]
    child2 = parent1[index:] + parent2[:index]
    return child1, child2

def mutation(graph, chromosme, probability):
    new_chromosme = chromosme.copy()
    if (random.random() > probability):
        n = len(chromosme)
        index = random.randint(0, n - 1)
        new_chromosme[index] = 1 - chromosme[index]
    return new_chromosme

def genetic_algorithm(graph_matrix, mutation_probability=0.1, pop_size=100, max_generation=100):
    best_cost = float("inf")
    best_solution = None
    n = len(graph_matrix)
    population = population_generation(n, pop_size)
    for generation in range(max_generation):
        population = tournament_selection(graph_matrix, population)
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]
            child1, child2 = crossover(graph_matrix, parent1, parent2)
            child1 = mutation(graph_matrix, child1, mutation_probability)
            child2 = mutation(graph_matrix, child2, mutation_probability)
            population.append(child1)
            population.append(child2)
        for i in range(len(population)):
            cost = cost_function2(graph_matrix, population[i])
            if cost < best_cost:
                best_cost = cost
                best_solution = population[i]
    return best_cost, best_solution

