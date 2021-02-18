#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os
from search import *  # for search engines
from snowman import SnowmanState, Direction, \
    snowman_goal_state  # for snowball specific classes
from test_problems import PROBLEMS  # 20 test problems
import time

INF = float('inf')


def heur_manhattan_distance(state):
    # IMPLEMENT
    """admissible sokoban puzzle heuristic: manhattan distance"""
    '''INPUT: a snowman state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each snowball that has yet to be stored and the
    # storage point is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.
    dist = 0
    for snowball in state.snowballs:
        man_dist = abs(state.destination[0] - snowball[0]) + abs(state.destination[1] - snowball[1])
        if state.snowballs[snowball] >= 3:
            man_dist *= 2
        dist += man_dist
    return dist


# s = SnowmanState("START", 0, None, 8, 10, (2, 2), {(2, 1): 0, (4, 3): 1, (1, 8): 2},
#              frozenset(((2, 3), (3, 0), (5, 1), (1, 3), (1, 2), (4, 5))), (4, 1),)


# HEURISTICS
def trivial_heuristic(state):
    """trivial admissible snowball heuristic"""
    '''INPUT: a snowball state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state 
    (# of moves required to get) to the goal.'''
    return len(state.snowballs)


def heur_alternate(state):
    # IMPLEMENT
    """a better heuristic"""
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_manhattan_distance has flaws.
    # Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.
    dist = 0
    for snowball in state.snowballs:
        if snowball[0] == state.destination[0] and snowball[1] == state.destination[1]:
            continue

        # check if an obstacle or the border blocks a snowball's movement in a specific direction
        # covers corners and edges
        blocked_right = ((snowball[0] + 1, snowball[1]) in state.obstacles) or (snowball[0] == state.width - 1)
        blocked_left = ((snowball[0] - 1, snowball[1]) in state.obstacles) or (snowball[0] == 0)
        blocked_up = ((snowball[0], snowball[1] - 1) in state.obstacles) or (snowball[1] == 0)
        blocked_down = ((snowball[0], snowball[1] + 1) in state.obstacles) or (snowball[1] == state.height - 1)

        if (blocked_left and blocked_up) or (blocked_left and blocked_down) or\
                (blocked_right and blocked_up) or (blocked_right and blocked_down):
            return 1000

        if snowball[1] == 0 or snowball[1] == state.height - 1:
            if state.destination[1] - snowball[1] != 0:
                return 1000
        elif snowball[0] == 0 or snowball[0] == state.width - 1:
            if state.destination[0] - snowball[0] != 0:
                return 1000

        man_dist = abs(state.destination[0] - snowball[0]) + abs(
            state.destination[1] - snowball[1])
        if state.snowballs[snowball] >= 3:
            man_dist *= 2
        dist += man_dist
    return dist


def heur_zero(state):
    """Zero Heuristic can be used to make A* search perform uniform cost search"""
    return 0


def fval_function(sN, weight):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """

    # Many searches will explore nodes (or states) that are ordered by their f-value.
    # For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue
    # is the hval of the state.
    # You can use this function to create an alternate f-value for states; this must be a function
    # of the state and the weight.
    # The function must return a numeric f-value.
    # The value will determine your state's position on the Frontier list during a 'custom' search.
    # You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    return sN.gval + weight * sN.hval


def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=5):
    # IMPLEMENT
    """Provides an implementation of anytime weighted a-star, as described in the HW1 handout"""
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of weighted astar algorithm'''
    weight = 7
    se = SearchEngine(strategy='custom')
    costbound = None
    temp_state = False
    start = time.time()
    new_timebound = time.time() - start
    while new_timebound < timebound - 0.1:
        wrapped_fval_func = lambda sN: fval_function(sN, weight)
        se.init_search(initial_state, snowman_goal_state, heur_fn, wrapped_fval_func)
        state = se.search(timebound, costbound)
        if state is False:
            return temp_state
        costbound = (state.gval, state.gval, state.gval)
        temp_state = state
        weight -= 1
        if weight < 1:
            weight = 1
        new_timebound = time.time() - start
    return temp_state


def anytime_gbfs(initial_state: SnowmanState, heur_fn, timebound=5):
    # IMPLEMENT
    """Provides an implementation of anytime greedy best-first search, as described in the HW1 handout"""
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of greedy best-first search algorithm'''
    se = SearchEngine(strategy='best_first')
    se.init_search(initial_state, snowman_goal_state, heur_fn)
    temp_state = False
    costbound = None
    start = time.time()
    new_timebound = time.time() - start
    while new_timebound < timebound - 0.1:
        state = se.search(timebound, costbound)
        if state is False:
            return temp_state
        temp_state = state
        costbound = (state.gval, state.gval, state.gval)
        new_timebound = time.time() - start
    return temp_state
