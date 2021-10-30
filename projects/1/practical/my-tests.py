from collections import defaultdict
import itertools


def solve(N, M, K, NUMS, roads):
    ###################################################################
    # (Point: determined by number of passed test)                    #
    # This function get input N, M, K, NUMS and roads                 #
    # which N is number of floristries,                               #
    # and M is number of roads,                                       #
    # and K is number of flower types,                                #
    # and NUMS are inventory of floristries,                          #
    # and roads are the roads between two floristries.                #
    # This function returns a number                                  #
    # which represents minimum amount of time it'll take for the boys,#
    # to collectively purchase all **k** types of flowers             #
    # and meet up at floristry n                                      #
    ###################################################################
    neighbours = find_neighbours(roads)
    print(neighbours)
    double_backtrack(N, K, NUMS, neighbours, (1, 1), (0, 0),
                     set(), (set(), set()), (set(), set()), (-1, -1))
    return MIN_COST


def find_neighbours(roads):
    neighbours = defaultdict(list)
    for road in roads:
        neighbours[road[0]].append(tuple(road[1:]))
        neighbours[road[1]].append((road[0], road[2]))
    return neighbours


MIN_COST = float('inf')


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
        if costs[0] + costs[1] < MIN_COST:
            MIN_COST = costs[0] + costs[1]
        return
    all_combinations = itertools.product(
        neighbours[currents[0]], neighbours[currents[1]])
    for neighbour1, neighbour2 in all_combinations:
        neighbour1_node, cost1 = neighbour1
        neighbour2_node, cost2 = neighbour2
        if (neighbour1_node, neighbour2_node) == parents:
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


def test():
    return solve(5, 5, 5, [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]], [
        [1, 2, 10], [1, 3, 10], [2, 4, 10], [3, 5, 10], [4, 5, 10]])


print(test())
