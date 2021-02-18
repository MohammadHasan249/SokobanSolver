import multiprocessing

# import student's functions
from solution import *
from snowman import snowman_goal_state

# Select what to test
test_time_astar = True
test_time_gbfs = True
test_manhattan = False
test_fval_function = False
test_anytime_gbfs = False
test_alternate = False
test_anytime_weighted_astar = False

TIMEOUT = 5  # timeout to impose

if __name__ == '__main__':
    if test_time_astar:

        time_bound = 3
        p = multiprocessing.Process(target=anytime_weighted_astar,
                                    name="Anytime A star", args=(PROBLEMS[19], heur_alternate, 10, time_bound))
        p.start()
        p.join(3.25)  # 1/4 second of fudge
        if p.is_alive():
            print(
                'Process killed. anytime_weighted_astar() not keeping track of time properly.')
            p.terminate()
            p.join()
        else:
            print('anytime_weighted_astar did not exceed timebound')

    if test_time_gbfs:

        time_bound = 3
        q = multiprocessing.Process(target=anytime_gbfs, name="Anytime GBFS",
                                    args=(
                                        PROBLEMS[19], heur_alternate,
                                        time_bound))
        q.start()
        q.join(3.25)  # 1/4 second of fudge
        if q.is_alive():
            print(
                'Process killed. anytime_gbfs() not keeping track of time properly')
            q.terminate()
            q.join()
        else:
            print('anytime_gbfs did not exceed timebound')

    if test_manhattan:
        ##############################################################
        # TEST MANHATTAN DISTANCE
        print('Testing Manhattan Distance')

        # Correct Manhattan distances for the initial states of the provided problem set
        correct_man_dist = [14, 9, 4, 11, 5, 5, 6, 7, 9, 5, 9, 9, 4, 24, 17, 5,
                            16, 25, 14, 14]

        solved = 0;
        unsolved = [];

        for i in range(0, 20):

            s0 = PROBLEMS[i]

            man_dist = heur_manhattan_distance(s0)
            print('calculated man_dist:', str(man_dist))
            # To see state uncomment
            # print(s0.state_string())
            if man_dist == correct_man_dist[i]:
                solved += 1
            else:
                unsolved.append(i)

        print("*************************************")
        print(
            "In the problem set provided, you calculated the correct Manhattan distance for {} states out of 20.".format(
                solved))
        print("States that were incorrect: {}".format(unsolved))
        print("*************************************\n")
        ##############################################################

    if test_alternate:

        ##############################################################
        # TEST ALTERNATE HEURISTIC
        print('Testing alternate heuristic with best_first search')

        solved = 0;
        unsolved = [];
        benchmark = 15;
        timebound = TIMEOUT  # time limit
        lengths = [54, 82, 30, 49, 71, 68, 29, 47, 38, 30, 68, 76, 53, -99, -99,
                   117, -99, -99, 64, -99]
        for i in range(0, len(PROBLEMS)):

            print("*************************************")
            print("PROBLEM {}".format(i))

            s0 = PROBLEMS[i]  # Final problems are hardest
            se = SearchEngine('best_first', 'full')
            se.init_search(s0, goal_fn=snowman_goal_state,
                           heur_fn=heur_alternate)
            final = se.search(timebound)

            if final:
                solved += 1
            else:
                unsolved.append(i)

        print("\n*************************************")
        print(
            "Of {} initial problems, {} were solved in less than {} seconds by this solver.".format(
                len(PROBLEMS), solved, timebound))
        print(
            "Problems that remain unsolved in the set are Problems: {}".format(
                unsolved))
        print(
            "The benchmark implementation solved {} out of {} practice problems given {} seconds.".format(
                benchmark, len(PROBLEMS), timebound))
        print("*************************************\n")
        ##############################################################

    if test_fval_function:

        test_state = SnowmanState("START", 6, None, 8, 10, (2, 2),
                                  {(2, 1): 0, (4, 3): 1, (1, 8): 2}, frozenset(
                ((2, 3), (3, 0), (5, 1), (1, 3), (1, 2), (4, 5))), (4, 1))

        correct_fvals = [6, 11, 16]

        ##############################################################
        # TEST fval_function
        print("*************************************")
        print('Testing fval_function')

        solved = 0
        weights = [0., .5, 1.]
        for i in range(len(weights)):

            test_node = sNode(test_state, hval=10, fval_function=fval_function)

            fval = fval_function(test_node, weights[i])
            print('Test', str(i), 'calculated fval:', str(fval), 'correct:',
                  str(correct_fvals[i]))

            if fval == correct_fvals[i]:
                solved += 1

        print("\n*************************************")
        print(
            "Your fval_function calculated the correct fval for {} out of {} tests.".format(
                solved, len(correct_fvals)))
        print("*************************************\n")
        ##############################################################

    if test_anytime_gbfs:

        len_benchmark = [44, 47, 19, 36, 35, 60, 18, 22, 34, 28, 32, 43, 35,
                         -99, -99, 40, -99, -99, 44, -99]
        ##############################################################
        # TEST ANYTIME GBFS
        print('Testing Anytime GBFS')

        solved = 0;
        unsolved = [];
        benchmark = 0;
        timebound = TIMEOUT  # time limit
        for i in range(0, len(PROBLEMS)):
        # for i in range(5, 6):
            print("*************************************")
            print("PROBLEM {}".format(i))

            s0 = PROBLEMS[i]  # Final problems are hardest
            final = anytime_gbfs(s0, heur_fn=heur_alternate,
                                 timebound=timebound)

            if final:
                if i < len(len_benchmark):
                    index = i
                else:
                    index = 0
                if final.gval <= len_benchmark[index] or len_benchmark[
                    index] == -99:
                    benchmark += 1
                solved += 1
            else:
                unsolved.append(i)

        print("\n*************************************")
        print(
            "Of {} initial problems, {} were solved in less than {} seconds by this solver.".format(
                len(PROBLEMS), solved, timebound))
        print(
            "Of the {} problems that were solved, the cost of {} matched or outperformed the benchmark.".format(
                solved, benchmark))
        print(
            "Problems that remain unsolved in the set are Problems: {}".format(
                unsolved))
        print(
            "The benchmark implementation solved {} out of the 20 practice problems given {} seconds.".format(
                15, timebound))
        print("*************************************\n")

    if test_anytime_weighted_astar:

        len_benchmark = [44, 43, 20, 36, 35, 34, 18, 22, 34, 28, 32, 43, 35,
                         -99, -99, 40, -99, -99, 46, -99]

        ##############################################################
        # TEST ANYTIME WEIGHTED A STAR
        print('Testing Anytime Weighted A Star')

        solved = 0;
        unsolved = [];
        benchmark = 0;
        timebound = TIMEOUT  # time limit
        for i in range(0, len(PROBLEMS)):
            print("*************************************")
            print("PROBLEM {}".format(i))

            s0 = PROBLEMS[i]  # Final problems are hardest
            weight = 100  # we will start with a large weight so you can experiment with rate at which it decrements
            # final = anytime_weighted_astar(s0, heur_fn=heur_alternate,
            #                                weight=weight, timebound=timebound)
            final = anytime_weighted_astar(s0, heur_fn=heur_alternate, timebound=timebound)

            if final:
                if i < len(len_benchmark):
                    index = i
                else:
                    index = 0
                if final.gval <= len_benchmark[index] or len_benchmark[
                    index] == -99:
                    benchmark += 1
                solved += 1
            else:
                unsolved.append(i)

        print("\n*************************************")
        print(
            "Of {} initial problems, {} were solved in less than {} seconds by this solver.".format(
                len(PROBLEMS), solved, timebound))
        print(
            "Of the {} problems that were solved, the cost of {} matched or outperformed the benchmark.".format(
                solved, benchmark))
        print(
            "Problems that remain unsolved in the set are Problems: {}".format(
                unsolved))
        print(
            "The benchmark implementation solved {} out of the 20 practice problems given {} seconds.".format(
                15, timebound))
        print("*************************************\n")
        ##############################################################
