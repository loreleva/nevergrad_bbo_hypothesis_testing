import nevergrad as ng
import sfu.evaluation_code.functions as ev
import time, sys, os, math, itertools
from multiprocessing import Process, Queue
from memory_profiler import memory_usage

# execute N run of nevergrad, in each run we have num_workers parallel evaluation of the function, then we take the minim evaluation among num_workers as temp result of nevergrad

num_proc = 0
num_points = 0
function_obj = None
q_inp = None
q_res = None
range_stopping_criteria = None
path_dir_log_file = None
csv_sep = ";"


def minimum_f(dim=None):
	return ev.minimum_value(dim)

def proc_function(q_inp, q_res):
	inp = q_inp.get()
	while (inp != None):
		start_time = time.process_time()
		mem_usage = memory_usage(os.getpid(), interval=.1)
		res = run_nevergrad()
		res = ((time.process_time() - start_time, max(mem_usage)), res)
		q_res.put(res)
		inp = q_inp.get()

def init_processes():
	global num_proc, q_inp, q_res
	q_inp = Queue()
	q_res = Queue()
	processes = []
	for x in range(num_proc):
		processes.append(Process(target=proc_function, args=(q_inp, q_res,)))
		processes[x].start()
	return processes

def kill_processes(processes):
	for x in processes:
		x.kill()


def std_dev(samples):
	N = len(samples)
	sample_mean = sum(samples) / N
	res = math.sqrt(sum([(x - sample_mean)**2 for x in samples]) / N)
	return res

def stopping_criteria(samples):
	if std_dev(samples) < 1e-6:
		return True
	return False



def obtain_queries(optimizer):
	global num_points
	input_points = []
	for i in range(num_points):
		query = optimizer.ask()
		query = list(*query.args)
		input_points.append(query)
	return input_points

def update_optimizer(optimizer, input_points, computed_points):
	global num_points
	for i in range(num_points):
		candidate = optimizer.parametrization.spawn_child(new_value = input_points[i])
		optimizer.tell(candidate, computed_points[i])

def compute_points(input_points):
	global function_obj
	results = []
	for inp in input_points:
		results.append(function_obj.evaluate(inp))
	return results

def run_nevergrad():
	global range_stopping_criteria, log_string, function_obj
	function_obj.input_lb = None
	function_obj.input_ub = None
	if function_obj.input_lb != None and function_obj.input_ub != None:
		param = ng.p.Array(shape=(function_obj.dimension,), lower=function_obj.input_lb, upper=function_obj.input_ub)
	else:
		param = ng.p.Array(shape=(function_obj.dimension,))

	# init nevergrad optimizer
	optimizer = ng.optimizers.NGOpt(parametrization=param, budget=10**6)

	results = []
	for x in range(range_stopping_criteria):
		
		input_points = obtain_queries(optimizer)
		computed_points = compute_points(input_points)
		results.append(min(computed_points))
		#print(f"RESULT : {results}")
		update_optimizer(optimizer, input_points, computed_points)

	while (not stopping_criteria(results)):
		input_points = obtain_queries(optimizer)
		computed_points = compute_points(input_points)
		results.append(min(computed_points))
		results = results[1:]
		#print(f"RESULT : {results}")
		update_optimizer(optimizer, input_points, computed_points)
	recommendation = optimizer.provide_recommendation()
	return (optimizer.num_ask, list(*recommendation.args), function_obj.evaluate(list(*recommendation.args)))


def write_log_file(path, string_res):
	# write temp results
	with open(path, "w") as f:
		f.write(string_res)
		f.close()


def hypothesis_testing(delta, epsilon, tolerance = 1e-6):
	global q_inp, q_res, path_dir_log_file, num_proc, function_obj, csv_sep
	N = math.ceil((math.log(delta)/math.log(1-epsilon)))
	print(f"N: {N}")
	total_process_time = 0
	max_ram_usage = 0
	# init S
	q_inp.put(1)
	res = q_res.get()
	total_process_time += res[0][0]
	if max_ram_usage < res[0][1]:
		max_ram_usage = res[0][1]
	log_runs_string = f"Iteration{csv_sep} Result{csv_sep} S{csv_sep} Number of Asks{csv_sep} Time{csv_sep} Max RAM Megabyte Usage\n"
	print(log_runs_string)
	temp_string = f"0/{N}{csv_sep} {res[1][2]}{csv_sep} {csv_sep} {res[1][0]}{csv_sep} {res[0][0]}{csv_sep} {res[0][1]}\n"
	print(temp_string)
	log_runs_string = log_runs_string + temp_string
	res = res[1]
	S_values = [(0, res[1], res[2])]
	S_prime = res[2]
	num_iterations = 0
	num_iter_internal = 0
	write_string = ""
	while (1):
		S = S_prime
		num_iterations += 1
		
		q_sizes = q_inp.qsize() + q_res.qsize()
		for x in range(N - q_sizes):
			q_inp.put(1)

		counter_samples = 0
		while(counter_samples < N):
			num_iter_internal += 1
			counter_samples += 1
			res = q_res.get()
			total_process_time += res[0][0]
			if max_ram_usage < res[0][1]:
				max_ram_usage = res[0][1]
			temp_string = f"{counter_samples}/{N}{csv_sep} {res[1][2]}{csv_sep} {S}{csv_sep} {res[1][0]}{csv_sep} {res[0][0]}{csv_sep} {res[0][1]}\n"
			print(temp_string)
			log_runs_string = log_runs_string + temp_string
			res = res[1]
			

			# if result smaller than starting S, restart
			if (res[2] + tolerance) < S:
				S_prime = res[2]
				S_values.append((num_iter_internal, res[1], res[2]))
				write_string = ""
				break

		# if after the loop starting S is equal to S_prime, end algorithm
		if S == S_prime:
			break
		
	write_log_file(os.path.join(path_dir_log_file, "log_runs.csv"), log_runs_string)
	
	log_result_string = f"Number external iterations{csv_sep} Number internal iterations{csv_sep} Result{csv_sep} Point Result{csv_sep} Optimum{csv_sep} Point Optimum{csv_sep} Error{csv_sep} Time{csv_sep} Mean time per process{csv_sep} Max RAM Megabyte Usage\n"
	print(log_result_string)
	temp_string = f"{num_iterations}{csv_sep} {num_iter_internal}{csv_sep} {S_values[-1][2]}{csv_sep} {S_values[-1][1]}{csv_sep} {function_obj.minimum_f}{csv_sep} {function_obj.minimum_x}{csv_sep} {abs(function_obj.minimum_f - S_values[-1][2])}{csv_sep} {total_process_time}{csv_sep} {total_process_time/num_proc}{csv_sep} {max_ram_usage}\n"
	print(temp_string)
	log_result_string = log_result_string + temp_string
	write_log_file(os.path.join(path_dir_log_file, "log_results.csv"), log_result_string)

	log_s_values_string = f"Internal iteration when assigned{csv_sep} S value\n"
	for s_value in S_values:
		log_s_values_string = log_s_values_string + f"{s_value[0]}{csv_sep} {s_value[2]}\n"
	print(log_s_values_string)
	write_log_file(os.path.join(path_dir_log_file, "log_s_values.csv"), log_s_values_string)
	return (total_process_time, S_values[-1])


# EGGHOLDER RESULT FUNCTION AND SAVE ALSO POINT OF OPTIMUMS

def main(argv):
	# argv[0] : function name, argv[1] : # points to evaluate, argv[2] : # parallel processes
	global num_proc, num_points, function_obj, range_stopping_criteria, path_dir_log_file
	processes_time = 0

	# path of all the functions results
	path_dir_res = os.path.join(os.path.dirname(__file__),"log_results")
	if not os.path.exists(path_dir_res):
		os.mkdir(path_dir_res)

	# init the module which compute the function and the infos about it
	function_name = argv[0]
	function_json = ev.functions_json(function_name)

	# path of the function results
	path_dir_res = os.path.join(path_dir_res, f"{function_name}")
	if not os.path.exists(path_dir_res):
		os.mkdir(path_dir_res)

	# define the dimensions of the function to test
	if function_json["dimension"] == "d":
		if type(function_json["minimum_f"]) == dict and list(function_json["minimum_f"].keys())[0] == "dimension":
			dimensions = [int(x) for x in function_json["minimum_f"]["dimension"].keys()]
		else:
			dimensions = [2] + [10**(x+1) for x in range(2)]
	else:
		dimensions = [function_json["dimension"]]

	num_points = int(argv[1])
	num_proc = int(argv[2])
	range_stopping_criteria = 20
	delta = 0.001
	epsilon = 0.001
	coeff_parameters = 5

	for dim in dimensions:
		# path of the results for the dimension dim
		path_dir_res_dim = os.path.join(path_dir_res, f"dimension_{dim}")
		if not os.path.exists(path_dir_res_dim):
			os.mkdir(path_dir_res_dim)

		function_obj = ev.objective_function(function_name, dim=dim)

		# if function accepts parameters
		if function_obj.parameters != None:
			# dictionary with all the parameters values, for each parameter
			comb_parameters = {}
			max_len_param_val = 0
			for param_name in function_obj.parameters_names:
				if function_obj.minimum_f_param != None and function_obj.minimum_f_param[0] == param_name:
					comb_parameters[param_name] = function_obj.minimum_f_param[1]
				else:
					param_value = function_obj.parameters_values[param_name]
					if type(param_value) == list:
						if type(param_value[0]) == list:
							temp_matrix = []
							for row in param_value:
								temp_matrix.append([x*coeff_parameters for x in row])
							comb_parameters[param_name] = [param_value, temp_matrix]
						else:
							comb_parameters[param_name] = [param_value, [x*coeff_parameters for x in param_value]]
					else:
						comb_parameters[param_name] = [param_value, param_value*coeff_parameters]
				if len(comb_parameters[param_name]) > max_len_param_val:
						max_len_param_val = len(comb_parameters[param_name])

			# for each possible combination of the parameters values
			for param_values_list in itertools.product(*[comb_parameters[param_name] for param_name in function_obj.parameters_names]):
				i=0
				temp_param_values = {}
				for param_name in function_obj.parameters_names:
					temp_param_values[param_name] = param_values_list[i]
					i+=1
				function_obj.set_parameters(temp_param_values)
				print(f"TESTING FUNCTION DIM: {dim} PARAMETERS: {[ (param_name, str(temp_param_values[param_name])) for param_name in function_obj.parameters_names ]} OPT_POINT: {function_obj.minimum_f}\n")
				path_dir_log_file = os.path.join(path_dir_res_dim, f"function_{function_name}_dimension_{dim}_parameters_{[ (param_name, temp_param_values[param_name]) for param_name in function_obj.parameters_names ]}")
				if not os.path.exists(path_dir_log_file):
					os.mkdir(path_dir_log_file)

				processes = init_processes()
				hypo_testing_res = hypothesis_testing(delta, epsilon)

				kill_processes(processes)

		# if function does not accept parameters
		else:
			print(f"TESTING FUNCTION DIM: {dim} OPT_POINT: {function_obj.minimum_f}\n")
			path_dir_log_file = os.path.join(path_dir_res_dim, f"function_{function_name}_dimension_{dim}")
			if not os.path.exists(path_dir_log_file):
				os.mkdir(path_dir_log_file)

			processes = init_processes()
			hypo_testing_res = hypothesis_testing(delta, epsilon)

			kill_processes(processes)
	




if __name__ == "__main__":
	main(sys.argv[1:])
