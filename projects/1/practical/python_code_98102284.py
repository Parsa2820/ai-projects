student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'



from collections import defaultdict
import itertools


def solve(N, M, K, NUMS, roads):
    neighbours = find_neighbours(roads)
    print(neighbours)
    double_backtrack(N, K, NUMS, neighbours, (1, 1), (0, 0), set())


def find_neighbours(roads):
    neighbours = defaultdict(list)
    for road in roads:
        neighbours[road[0]].append(tuple(road[1:]))
    return neighbours

MIN_COST = float('inf')

def double_backtrack(N, K, NUMS, neighbours, currents, costs, collected_flower_types):
    current_collected_flower_types = collected_flower_types.copy()
    current_collected_flower_types.add(NUMS[currents[0]][1:])
    current_collected_flower_types.add(NUMS[currents[1]][1:])
    if currents == (N, N) and len(current_collected_flower_types) == K:
        global MIN_COST
        if costs[0] + costs[1] < MIN_COST:
            MIN_COST = costs[0] + costs[1]
    all_combinations = itertools.product(
        neighbours[currents[0]], neighbours[currents[1]])
    for neighbour1, neighbour2 in all_combinations:
        neighbour1_node, cost1 = neighbour1
        neighbour2_node, cost2 = neighbour2
        double_backtrack(N, K, NUMS, neighbours, (neighbour1_node, neighbour2_node),
                         (costs[0]+cost1, costs[1]+cost2), current_collected_flower_types)


def test():
    solve(5, 5, 5, [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]], [
          [1, 2, 10], [1, 3, 10], [2, 4, 10], [3, 5, 10], [4, 5, 10]])


test()




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
    storage_np = np.asarray(list(state.storage.keys()))
    for box_cordinate in state.boxes.keys():
        box_cordinate_np = np.tile(box_cordinate, (storage_np.shape[0], 1))
        heur += np.abs(storage_np - box_cordinate_np).sum(axis=1).min()
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



def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10):
    best_path_cost = float("inf")
    time_remain = 8
    iter = 0

    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    se = SearchEngine('custom', 'full')
    se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)

    while (time_remain > 0) and not se.open.empty():
        pass
    try:
        return optimal_final
    except:
        return final

    return False





edge_count = # Complete This (1 Points)
print(edge_count)

def random_state_generator(n):
    pass

def neighbour_state_generator(state):
    new_state = state.copy()
    previous_value = None
    vertex_to_change = None
    return new_state, previous_value, vertex_to_change

def cost_function(graph_matrix,state , A = 1 , B=1):
    pass

deg = #Complete This (2 Points)

def prob_accept(...): 
    pass

def accept(current_state , next_state , ...):
    pass

def plot_cost(cost_list):
    pass

plot_cost(cost_list)



def population_generation(n, k): 
    pass

def cost_function2(graph,state):
    pass

def tournament_selection(graph, population):
    new_population = None
    return new_population

def crossover(graph, parent1, parent2):
    child1 = None
    child2 = None
    return child1, child2

def mutation(graph,chromosme,probability):
    pass

def genetic_algorithm(graph_matrix,mutation_probability=0.1,pop_size=100,max_generation=100):
    best_cost = None
    best_solution = None
    return best_cost,best_solution

