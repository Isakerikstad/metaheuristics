import argparse
import time
import numpy as np

# Import utilities from pdp_utils
from pdp_utils import *

def read_data(file_path):
    return load_problem(file_path)

def check_feasibility(solution, data):
    feasible, _ = feasibility_check(solution, data)
    return feasible

def evaluate_solution(solution, data):
    return cost_function(solution, data)

def generate_random_solution(data, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n_vehicles = int(data['n_vehicles'])
    n_calls = int(data['n_calls'])
    total_vehicles = n_vehicles + 1  # dummy included

    # Prepare cargo data
    cargo = data['Cargo']
    if cargo.ndim > 1 and cargo.shape[1] > 1:
        cargo = cargo[:, 0]
    else:
        cargo = np.ravel(cargo)

    vessel_cargo = data['VesselCargo']
    assignments = np.random.randint(0, total_vehicles, size=n_calls)

    # Reassign calls not allowed by vessel_cargo
    for i in range(n_calls):
        v = assignments[i]
        if v < n_vehicles:
            while v < n_vehicles and vessel_cargo[v, i] != 1:
                new_v = np.random.randint(0, n_vehicles) if np.random.rand() < 0.5 else n_vehicles
                assignments[i] = new_v
                v = new_v

    # Group calls by vehicle
    vehicle_calls = []
    for v in range(total_vehicles):
        call_indices = np.where(assignments == v)[0]
        vehicle_calls.append(call_indices + 1)

    # Capacity check
    vessel_capacity = data['VesselCapacity']
    for v in range(n_vehicles):
        calls_v = vehicle_calls[v].tolist()
        total_cargo = sum(cargo[call - 1] for call in calls_v)
        while calls_v and total_cargo > vessel_capacity[v]:
            idx_remove = np.random.randint(len(calls_v))
            removed_call = calls_v.pop(idx_remove)
            vehicle_calls[n_vehicles] = np.append(vehicle_calls[n_vehicles], removed_call)
            total_cargo = sum(cargo[call - 1] for call in calls_v)
        vehicle_calls[v] = np.array(calls_v, dtype=int)

    # Duplicate & sort or shuffle calls
    loading_time = data.get('LoadingTime', None)
    for v in range(total_vehicles):
        calls_v = vehicle_calls[v]
        if calls_v.size > 0:
            calls_v = np.concatenate((calls_v, calls_v))
            if (loading_time is not None) and (v < n_vehicles):
                lt = loading_time[v, calls_v - 1]
                noise = np.random.rand(calls_v.size) * 1e-3
                sort_order = np.argsort(lt + noise)
                calls_v = calls_v[sort_order]
            else:
                np.random.shuffle(calls_v)
            vehicle_calls[v] = calls_v

    # Concatenate sequences with 0 as separator
    solution_parts = []
    for v in range(total_vehicles):
        if vehicle_calls[v].size > 0:
            solution_parts.append(vehicle_calls[v])
        solution_parts.append(np.array([0], dtype=int))
    solution = np.concatenate(solution_parts)
    if solution.size > 0 and solution[-1] == 0:
        solution = solution[:-1]

    return solution

def all_calls_outsorced(data):
    """
    Assign all calls to the dummy vehicle, with calls duplicated,
    and produce a solution starting each vehicle route with 0.
    """
    import numpy as np
    n_vehicles = data['n_vehicles'] - 1
    n_calls = data['n_calls']
    calls = np.arange(1, n_calls + 1)
    calls_duplicated = np.concatenate((calls, calls))  # duplicate calls

    solution_parts = []
    # For each real vehicle, no calls assigned (route is just [0]).
    for _ in range(n_vehicles):
        solution_parts.append([0])
    # For the dummy vehicle, prepend 0 and then the duplicated calls.
    dummy_route = [0] + calls_duplicated.tolist()
    solution_parts.append(dummy_route)

    solution = np.concatenate(solution_parts)
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default=None)
    parser.add_argument('--file_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None, 
                        help="Optional seed for reproducibility. Using seed=5") # Use seed 5 for testing7
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    if args.function == "blind_random_search_10000" and args.file_path:
        start_time = time.time()    
        data = read_data(args.file_path)

        best_solution = all_calls_outsorced(data)
        initial_score = evaluate_solution(best_solution, data)
        print('All calls outsourced to dummy vehicle gives score', initial_score)
        total_objective = initial_score
        best_objective = initial_score
        feasible_tally = 1
        infeas_counts = {}

        for i in range(10000):
            seed = 20000 * args.seed + i if args.seed is not None else None
            solution = generate_random_solution(data, seed=seed)
            feasible, reason = feasibility_check(solution, data)
            if feasible:
                feasible_tally += 1
                objective_score = evaluate_solution(solution, data)
                total_objective += objective_score
                if objective_score < best_objective:
                    best_solution = solution
                    best_objective = objective_score
            else:
                infeas_counts[reason] = infeas_counts.get(reason, 0) + 1

        improvement = 100.0 * (initial_score - best_objective) / initial_score if initial_score != 0 else 0
        print('The best solution was', best_solution, 'with objective score', best_objective)
        print('The average objective score was', total_objective / feasible_tally)
        print('The improvement was', improvement, '%')
        print('The number of feasible solutions out of 10000 was', feasible_tally)
        print('Infeasibility causes and counts:', infeas_counts)
        print('The time it took to generate this solution was', time.time() - start_time)

        if not args.test:    
            with open(args.file_path.replace(".txt", "_solution.txt"), 'w') as file:
                file.write("Avg. objective: {}\n".format(total_objective / feasible_tally))
                file.write("Best solution (calls): {}\n".format(best_solution))
                file.write("best objective score: {}\n".format(best_objective))
                file.write("Improvement: {}%\n".format(improvement))
                file.write("Runtime: {}\n".format(time.time() - start_time))
        else:
            print('Test so no file written')

if __name__ == '__main__':
    main()