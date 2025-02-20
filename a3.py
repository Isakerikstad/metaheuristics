import argparse
import time
import numpy as np

# Import utilities from pdp_utils
from pdp_utils import *

# import generate_random_solution from a2.py
from a2 import generate_random_solution

file_paths =['data\Call_7_Vehicle_3.txt', 'data\Call_18_Vehicle_5.txt', 'data\Call_35_Vehicle_7.txt', 'data\Call_80_Vehicle_20.txt', 'data\Call_130_Vehicle_40.txt', 'data\Call_300_Vehicle_90.txt']

def read_data(file_path):
    return load_problem(file_path)

def check_feasibility(solution, data):
    feasible, _ = feasibility_check(solution, data)
    return feasible

def evaluate_solution(solution, data):
    return cost_function(solution, data)

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

def local_search(solution, data, iteration_number, seed=None):
    """
    Local search using 1-reinsert operator: removes one call and reinserts it in a new position.
    Handles constraints and ensures meaningful neighborhood moves.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert solution to list for easier manipulation
    solution = solution.tolist()
    
    # Find all calls in the solution (excluding 0s)
    calls = set(x for x in solution if x > 0)
    if not calls:
        return np.array(solution)
    
    # Randomly select a call to remove and reinsert
    call_to_move = np.random.choice(list(calls))
    
    # Find positions of the selected call
    pickup_idx = solution.index(call_to_move)
    delivery_idx = len(solution) - 1 - solution[::-1].index(call_to_move)
    
    # Find vehicle boundaries (positions of 0s)
    vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
    if not vehicle_bounds:
        vehicle_bounds = []
    
    # Determine which vehicle the call is currently in
    current_vehicle = 0
    for i, bound in enumerate(vehicle_bounds):
        if pickup_idx > bound:
            current_vehicle = i + 1
    
    # Remove the call from both positions
    solution.pop(delivery_idx)
    solution.pop(pickup_idx)
    
    n_vehicles = data['n_vehicles']

    # available_vehicles in numpy array
    available_vehicles = np.array(range(n_vehicles))        

    # Remove all vehicles with uncomapatible with our call and capacity
    available_vehicles = np.array([v for v in available_vehicles if data['VesselCargo'][v][call_to_move - 1] == 1])
    
    # Check call size and call outsource cost against mean
    large_call = False
    call_size = data['Cargo'][call_to_move - 1, 2]
    if call_size > data['average_size']:
        large_call = True

    expensive_call = False
    call_cost = data['Cargo'][call_to_move - 1, 3]
    if call_cost > data['average_cost_outsource']:
        expensive_call = True

    # Over available vehicles, give property True if vehicle capacity over mean
    probs = np.ones(len(available_vehicles))
    large_capacity = np.array([data['VesselCapacity'][v] > data['average_capacity'] for v in available_vehicles])

    if large_call:
        # Larger probability for vehicles above average capacity
        probs = np.where(large_capacity, 1.5, 0.8)
        probs[-1] = 2

    else:
        # Larger probability for vehicles below average capacity
        probs = np.where(large_capacity, 0.7, 1)

    if expensive_call and current_vehicle == n_vehicles -1 or iteration_number < n_vehicles // 3:
        # lower probability for dummy
        probs[-1] = 0.1

    # Normalize probabilities
    probs = probs / np.sum(probs)
    #print(available_vehicles, probs)



    # Make sure we still have valid vehicles
    if len(probs) == 0:
        target_vehicle = n_vehicles - 1
    else:
        # Use the computed distribution to choose vehicles
        available_vehicles = np.random.choice(available_vehicles, size=len(available_vehicles), replace=False, p=probs)
        target_vehicle = np.random.choice(available_vehicles)
    
    # Get the range for the target vehicle
    start_idx = 0 if target_vehicle == 0 else vehicle_bounds[target_vehicle - 1] + 1
    end_idx = vehicle_bounds[target_vehicle] if target_vehicle < len(vehicle_bounds) else len(solution)
    
    """# Replace the random position selection with systematic evaluation
    route_length = end_idx - start_idx
    vehicle_route = solution[start_idx:end_idx]
    best_route = []
    best_cost = np.inf
    
    # If route is empty or this is current vehicle
    if route_length == 0:
        best_route = [call_to_move, call_to_move]
    else:
        L = len(vehicle_route)
        # Try all possible positions for pickup and delivery
        for i in range(L + 1):  # +1 to allow insertion at end
            for j in range(i + 1, L + 2):  # ensure delivery comes after pickup
                new_route = vehicle_route.copy()
                # Insert delivery first so insertion at i doesn't shift j
                new_route.insert(j, call_to_move)
                new_route.insert(i, call_to_move)
                
                # Check cost using cache
                candidate_key = tuple(new_route)
                if candidate_key in route_cost_cache:
                    new_cost = route_cost_cache[candidate_key]
                else:
                    new_cost = cost_function_light(new_route, data, target_vehicle)
                    route_cost_cache[candidate_key] = new_cost

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route.copy()
    
    # Insert the best route back into the solution
    solution[start_idx:end_idx] = best_route"""
    

    # Choose random positions for reinsertion within the chosen vehicle's route
    route_length = end_idx - start_idx
    if route_length >= 2:
        # Generate two random positions within the vehicle's route
        pos1 = np.random.randint(start_idx, end_idx + 1)
        pos2 = np.random.randint(start_idx, end_idx + 2)
        
        # Insert the call at the chosen positions
        solution.insert(pos1, call_to_move)
        solution.insert(pos2, call_to_move)
    else:
        # If not enough space, insert at the start of the route
        solution.insert(start_idx, call_to_move)
        solution.insert(start_idx, call_to_move)
    
    return np.array(solution)

def simulated_annealing(solution, data, seed=None):        
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize solutions
    incumbent = solution.copy()
    best_solution = solution.copy()
    delta_w = []  # Store delta E values for initial temperature calculation
    
    # Initial warm-up phase (100 iterations)
    for i in range(100):
        iteration_seed = 100_000 * seed + i if seed is not None else None
        new_solution = local_search(incumbent, data, i, seed=iteration_seed)  # Using existing 1-reinsert operator
        if check_feasibility(new_solution, data):
            delta_e = evaluate_solution(new_solution, data) - evaluate_solution(incumbent, data)
            
            if delta_e < 0:  # If improvement found
                incumbent = new_solution.copy()
                if evaluate_solution(incumbent, data) < evaluate_solution(best_solution, data):
                    best_solution = incumbent.copy()
            elif np.random.random() < 0.8:  # Accept with 0.8 probability during warm-up
                incumbent = new_solution.copy()
            
            delta_w.append(delta_e)
    
    # Calculate initial temperature and cooling rate
    delta_avg = np.mean(delta_w) if delta_w else 1.0
    T0 = max(-delta_avg / np.log(0.8), 1.0)  # Initial temperature with minimum value
    Tf = 0.1  # Final temperature
    alpha = np.power(Tf / T0, 1.0 / 9900)  # Cooling rate
    T = T0
    
    # Main simulated annealing loop
    for i in range(9900):
        iteration_seed = 100_000 * seed + i + 100 if seed is not None else None
        new_solution = local_search(incumbent, data, i + 20, seed=iteration_seed)
        
        if check_feasibility(new_solution, data):
            delta_e = evaluate_solution(new_solution, data) - evaluate_solution(incumbent, data)
            
            if delta_e < 0:  # Accept improvement
                incumbent = new_solution.copy()
                if evaluate_solution(incumbent, data) < evaluate_solution(best_solution, data):
                    best_solution = incumbent.copy()
            else:
                # Accept worse solution with probability e^(-ΔE/T)
                p = np.exp(-delta_e / max(T, 1e-10))  # Prevent division by zero
                if np.random.random() < p:
                    incumbent = new_solution.copy()
        
        T = max(alpha * T, Tf)  # Cool down temperature but don't go below Tf
    
    return best_solution

def a2_random_search(solution, data, i, seed=None):
    return generate_random_solution(data, seed)

def run_experiment(algorithm, data, args):
    """
    Run experiment for specified algorithm and data.
    """
    # Initialize global variables for cost calculation
    init_globals(data)

    initial_solution = all_calls_outsorced(data)
    best_solution = initial_solution.copy()
    initial_score = evaluate_solution(best_solution, data)
    print('All calls outsourced to dummy vehicle gives score', initial_score)
    best_objectives = []  # Store best objective for each run
    best_overall_objective = initial_score
    best_overall_solution = initial_solution.copy()
    feasible_tally = 1
    infeas_counts = {}
    total_time = 0

    # Calculate average weight, average car capacity, and avg cost to outsource
    data['average_size'] = np.mean(data['Cargo'][:, 2])
    data['average_cost_outsource'] = np.mean(data['Cargo'][:, 3])
    data['average_capacity'] = np.mean(data['VesselCapacity'])
    
    for j in range(args.number_runs):
        start_time = time.time()
        best_solution_this_run = initial_solution.copy()
        best_objective_this_run = initial_score
        if args.function == "simulated_annealing":
            best_solution_this_run = simulated_annealing(best_solution_this_run, data, seed=args.seed * (j + 1))
            best_objective_this_run = evaluate_solution(best_solution_this_run, data)
        else:
            for i in range(10_000):
                seed = 100_000 * args.seed * (j + 1) + i if args.seed is not None else None
                solution = algorithm(best_solution_this_run, data, i, seed=seed)
                feasible, reason = feasibility_check(solution, data)
                if feasible:
                    feasible_tally += 1
                    objective_score = evaluate_solution(solution, data)
                    if objective_score < best_objective_this_run:
                        best_solution_this_run = solution.copy()
                        best_objective_this_run = objective_score
                else:
                    infeas_counts[reason] = infeas_counts.get(reason, 0) + 1
            
        best_objectives.append(best_objective_this_run)
            
        # Update overall best if this run found a better solution
        if best_objective_this_run < best_overall_objective:
            best_overall_objective = best_objective_this_run
            best_overall_solution = best_solution_this_run.copy()

        total_time += time.time() - start_time
    

    # Calculate average of best objectives from each run
    average_objective = sum(best_objectives) / len(best_objectives)
    improvement = 100.0 * (initial_score - average_objective) / initial_score if initial_score != 0 else 0
    
    # Print best objective and infeasibility counts
    print(f'improvement: {improvement:0.2f} with bests objective score {best_overall_objective}')
    print('Feasible solutions found:', feasible_tally)
    print('Infeasibility causes and counts:', infeas_counts)

    # Write results to file
    if not args.test:
        algorithm_name = algorithm.__name__
        solution_file = args.file_path.replace(".txt", "_solution.txt")
        results_exist = False
        file_content = ""
        
        # Check if file exists and read its content
        try:
            with open(solution_file, 'r') as file:
                file_content = file.read()
                results_exist = f"{algorithm_name} Results:" in file_content
        except FileNotFoundError:
            pass

        # Prepare new results text
        new_results = f"\n{algorithm_name} Results:\n"
        new_results += "------------------\n"
        new_results += f"Seed: {args.seed}\n"
        new_results += f"Average objective: {average_objective:.2f}\n"
        new_results += f"Best solution objective: {best_overall_objective:.2f}\n"
        new_results += f"Best solution (calls): {list(best_overall_solution)}\n"
        new_results += f"Improvement from initial: {improvement:.2f}%\n"
        new_results += f"Runtime (seconds): {total_time/10:.2f}\n"
        new_results += f"Feasible solutions found: {feasible_tally}\n"
        new_results += f"Infeasibility causes and counts: {infeas_counts}\n"
        if args.number_runs != 10:
            new_results += f"Number of runs: {args.number_runs}\n"
        new_results += f"Time of experiment: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        new_results += "------------------\n"

        if results_exist:
            # Find the section to replace
            start = file_content.find(f"{algorithm_name} Results:")
            end = file_content.find("------------------\n", start) + 19
            
            # Replace only this algorithm's results
            new_content = file_content[:start] + new_results + file_content[end:]
            
            # Write the modified content
            with open(solution_file, 'w') as file:
                file.write(new_content.lstrip())
        else:
            # Append new results if they don't exist
            with open(solution_file, 'a') as file:
                file.write(new_results)
    else:
        print('Test so no file written')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default=None)
    parser.add_argument('--file_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None, 
                    help="Optional seed for reproducibility. Using seed=5")
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--full', type=bool, default=False)
    parser.add_argument('--number_runs', type=int, default=10)
    args = parser.parse_args()

    if args.full:
        start_timer = time.time()
        for file_path in file_paths:
            data = read_data(file_path)
            for function in ["local_search", "simulated_annealing"]:
                # add function as argument
                args.function = function
                args.file_path = file_path
                print(f"Time: {time.time() - start_timer:.2f} seconds \n\nRunning {function} on {file_path}")
                run_experiment(eval(function), data, args)
                args.function = None
                args.file_path = None
    else:
        if args.file_path:
            data = read_data(args.file_path)
            if args.function == "local_search":
                run_experiment(local_search, data, args)
            elif args.function == "simulated_annealing":
                run_experiment(simulated_annealing, data, args)
            elif args.function == "random_search":
                run_experiment(a2_random_search, data, args)
            else:
                print("Invalid function argument. Please use 'local_search' or 'simulated_annealing'")
        else:
            print("Please provide a file path using the --file_path argument")

if __name__ == '__main__':
    main()