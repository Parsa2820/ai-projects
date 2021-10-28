student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'



def solve(N, M, K, NUMS, roads): 
    pass



def heur_displaced(state):    
    pass

def heur_manhattan_distance(state):
    pass

def heur_euclidean_distance(state):  
    pass



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

