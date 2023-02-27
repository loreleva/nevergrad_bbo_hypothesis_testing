import nevergrad as ng
import sfu.evaluation_code.functions as ev
import time, sys, os, math, itertools, random
from datetime import timedelta
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
	global range_stopping_criteria, log_string, function_obj, num_points
	if function_obj.input_lb != None and function_obj.input_ub != None:
		param = ng.p.Array(shape=(function_obj.dimension,), lower=function_obj.input_lb, upper=function_obj.input_ub)
	else:
		param = ng.p.Array(shape=(function_obj.dimension,))

	# init nevergrad optimizer
	optimizer = ng.optimizers.NGOpt10(parametrization=param, num_workers=num_points, budget=10**6)
	if function_obj.input_lb != None and function_obj.input_ub != None:
		for n in range(num_points):
			point = []
			for dim in range(param.dimension):
				point.append(random.uniform(function_obj.input_lb[dim], function_obj.input_ub[dim]))
			optimizer.suggest(point)
	else:
		for n in range(num_points):
			point = []
			for dim in range(param.dimension):
				point.append(random.uniform(-100,100))
			optimizer.suggest(point)

	results = []
	for x in range(range_stopping_criteria):
		
		input_points = obtain_queries(optimizer)
		computed_points = compute_points(input_points)
		results.append(min(computed_points))
		#print(f"RESULT : {results}")
		update_optimizer(optimizer, input_points, computed_points)

	while (not stopping_criteria(results) and optimizer.num_ask <= 100000):
		#print(optimizer.num_ask)
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

def time_to_str(seconds):
	str_time = str(timedelta(seconds=seconds))
	res = ""
	hms = str_time.split()[-1].split(":")
	res = f"{hms[0]}H : {hms[1]}M : {hms[2]}S"
	if "day" in str_time:
		days = str_time.split()[0]
		hms = str_time.split()[-1].split(":")
		res = f"{days}D : " + res
	return res

def eucl_dist(x, y):
	sum_diff = 0
	for idx in range(len(x)):
		sum_diff += (y[idx] - x[idx])**2
	return math.sqrt(sum_diff)

def get_dist(algo_input):
	global function_obj
	if function_obj.minimum_x == None:
		return None
	else:
		if type(function_obj.minimum_x[0]) != list:
			res = eucl_dist(algo_input, function_obj.minimum_x)
			return res
		else:
			results = []
			for inp in function_obj.minimum_x:
				results.append(eucl_dist(algo_input, inp))
			res = min(results)
			return res




def hypothesis_testing(delta, epsilon, tolerance = 1e-6):
	global q_inp, q_res, path_dir_log_file, num_proc, function_obj, csv_sep
	# init N
	N = math.ceil((math.log(delta)/math.log(1-epsilon)))

	# track execution time of algorithm
	start_time = time.time()
	
	print(f"N: {N}")
	
	total_process_time = 0
	max_ram_usage = 0
	
	# request execution of one run of nevergrad and obtain result
	q_inp.put(1)
	res = q_res.get()
	
	processes_runs_time = [res[0][0]]
	number_of_asks = [res[1][0]]
	ram_usage = [res[0][1]]
	
	if max_ram_usage < res[0][1]:
		max_ram_usage = res[0][1]

	idx_csv = 1
	# write data into log csv
	log_runs_string = (
						f"Index{csv_sep} "
						f"Iteration{csv_sep} "
						f"Optimum Found{csv_sep} "
						f"S{csv_sep} "
						f"Number of Asks{csv_sep} "
						f"Time{csv_sep} "
						f"Max RAM Megabyte Usage\n"
					)
	print(log_runs_string)
	temp_string = (
					f"{idx_csv}{csv_sep} "
					f"0/{N}{csv_sep} "
					f"{res[1][2]}{csv_sep} "
					f"{csv_sep} "
					f"{res[1][0]}{csv_sep} "
					f"{res[0][0]}{csv_sep} "
					f"{res[0][1]}\n"
				)
	print(temp_string)
	log_runs_string = log_runs_string + temp_string
	
	# init S and S prime
	res = res[1]
	S_values = [(0, res[1], res[2])]
	S_prime = res[2]

	correct_opts = []
	num_correct_opts = 0
	if abs(function_obj.minimum_f - S_prime) <= tolerance:
		num_correct_opts += 1
	correct_opts.append((idx_csv, num_correct_opts, S_prime, function_obj.minimum_f, abs(function_obj.minimum_f - S_prime))) 
	
	num_iterations = 0
	
	write_string = ""

	while (1):
		S = S_prime
		num_iterations += 1
		
		# if size of the queue is less than N, add requests for nevergrad runs
		q_sizes = q_inp.qsize() + q_res.qsize()
		for x in range(N - q_sizes):
			q_inp.put(1)

		# check results of N nevergrad runs, or exit from the loop if value better than S is found
		counter_samples = 0
		while(counter_samples < N):
			counter_samples += 1
			idx_csv += 1

			res = q_res.get()
			processes_runs_time.append(res[0][0])
			number_of_asks.append(res[1][0])
			ram_usage.append(res[0][1])
			if max_ram_usage < res[0][1]:
				max_ram_usage = res[0][1]

			temp_string = (
							f"{idx_csv}{csv_sep} "
							f"{counter_samples}/{N}{csv_sep} "
							f"{res[1][2]}{csv_sep} "
							f"{S}{csv_sep} "
							f"{res[1][0]}{csv_sep} "
							f"{res[0][0]}{csv_sep} "
							f"{res[0][1]}\n"
						)
			print(temp_string)
			log_runs_string = log_runs_string + temp_string
			res = res[1]

			if abs(function_obj.minimum_f - res[2]) <= tolerance:
				num_correct_opts += 1
			correct_opts.append((idx_csv, num_correct_opts, res[2], function_obj.minimum_f, abs(function_obj.minimum_f - res[2])))

			# if result smaller than starting S, restart
			if (res[2] + tolerance) < S:
				S_prime = res[2]
				S_values.append((counter_samples, res[1], res[2]))
				write_string = ""
				break

		# if after the loop starting S is equal to S_prime, end algorithm
		if S == S_prime:
			break
		
	write_log_file(os.path.join(path_dir_log_file, "log_runs.csv"), log_runs_string)
	total_processes_runs_time = sum(processes_runs_time)
	mean_processes_runs_time = total_processes_runs_time/len(processes_runs_time)
	std_dev_processes_runs_time = std_dev(processes_runs_time)
	mean_number_of_asks = sum(number_of_asks)/len(number_of_asks)
	std_dev_number_of_asks = std_dev(number_of_asks)
	mean_ram_usage = sum(ram_usage)/len(ram_usage)
	std_dev_ram_usage = std_dev(ram_usage)

	log_result_string = (
							f"Runs of nevergrad{csv_sep} "
							f"Number external iterations{csv_sep} "
							f"Number internal iterations{csv_sep} "
							f"Optimum Found{csv_sep} "
							f"Input Optimum Found{csv_sep} "
							f"Function Optimum{csv_sep} "
							f"Function Input Optimum{csv_sep} "
							f"Error{csv_sep} "
							f"Input Error{csv_sep}"
							f"Correctness ratio{csv_sep}"
							f"Total algorithm execution time{csv_sep} "
							f"Sum of processes' system and user CPU time{csv_sep} "
							f"Mean system and user CPU time per process{csv_sep} "
							f"Mean of nevergrad runs' system and user CPU time {csv_sep} "
							f"Standard deviation of nevergrad runs' system and user CPU time{csv_sep} "
							f"Mean of number of asks in nevergrad runs{csv_sep} "
							f"Standard deviation of number of asks in nevergrad runs{csv_sep} "
							f"Max RAM Megabyte usage{csv_sep} "
							f"Mean RAM Megabyte usage{csv_sep} "
							f"Standard deviation of RAM Megabyte usage\n"
						)
	print(log_result_string)
	temp_string = (
					f"{idx_csv}{csv_sep} "
					f"{num_iterations}{csv_sep} "
					f"{idx_csv-1}{csv_sep} "
					f"{S_values[-1][2]}{csv_sep} "
					f"{S_values[-1][1]}{csv_sep} "
					f"{function_obj.minimum_f}{csv_sep} "
					f"{function_obj.minimum_x}{csv_sep} "
					f"{abs(function_obj.minimum_f - S_values[-1][2])}{csv_sep} "
					f"{get_dist(S_values[-1][1])}{csv_sep} "
					f"{correct_opts[-1][1]/idx_csv}{csv_sep} "
					f"{time_to_str(time.time() - start_time)}{csv_sep} "
					f"{time_to_str(total_processes_runs_time)}{csv_sep} "
					f"{time_to_str(total_processes_runs_time/num_proc)}{csv_sep} "
					f"{mean_processes_runs_time}{csv_sep} "
					f"{std_dev_processes_runs_time}{csv_sep} "
					f"{mean_number_of_asks}{csv_sep} "
					f"{std_dev_number_of_asks}{csv_sep} "
					f"{max_ram_usage}{csv_sep} "
					f"{mean_ram_usage}{csv_sep} "
					f"{std_dev_ram_usage}\n"
				)
	print(temp_string)
	log_result_string = log_result_string + temp_string
	write_log_file(os.path.join(path_dir_log_file, "log_results.csv"), log_result_string)

	log_s_values_string = (
							f"Internal iteration when assigned{csv_sep} "
							f"S value\n"
						)
	for s_value in S_values:
		log_s_values_string = log_s_values_string + f"{s_value[0]}/{N}{csv_sep} {s_value[2]}\n"
	print(log_s_values_string)
	write_log_file(os.path.join(path_dir_log_file, "log_s_values.csv"), log_s_values_string)

	log_correctness_ratio_string = (
									f"Index Run{csv_sep} "
									f"Number of correct results{csv_sep} "
									f"Optimum found{csv_sep} "
									f"Global optimum{csv_sep} "
									f"Error\n"
								)
	for val_correct_ratio in correct_opts:
		log_correctness_ratio_string = log_correctness_ratio_string + (
			f"{val_correct_ratio[0]}{csv_sep} "
			f"{val_correct_ratio[1]}{csv_sep} "
			f"{val_correct_ratio[2]}{csv_sep} "
			f"{val_correct_ratio[3]}{csv_sep} "
			f"{val_correct_ratio[4]}\n"
		)
	write_log_file(os.path.join(path_dir_log_file, "log_correctness_ratio.csv"), log_correctness_ratio_string)

	return (total_process_time, S_values[-1])

def get_test_dimensions(dim, min_f):
	"""Returns the list of dimension on which the function must be tested.

	Parameters
	----------
	dim: str
		json value for the field "dimension" of the functions' definition json. (e.g., "d", "2", etc...)
	min_f: dict, float
		json value for the field "minimum_f" of the functions' definition json. 
		It is a dict when the function's global minimum is defined only for specific dimensions (e.g., dict = {"dimension" : {"2" : -1, "5" : 0}}).
		Otherwise it is a float number.

	Returns
	-------
	list
		list of integers representing the dimensions on which the function must be tested
	
	"""
	if dim == "d":
		if type(min_f) == dict and list(min_f)[0] == "dimension":
			dimensions = [int(x) for x in list(min_f["dimension"])]
		else:
			dimensions = [2] + [10**(x+1) for x in range(2)]
	else:
		dimensions = [int(dim)]
	return dimensions

def get_test_parameters()


def main(argv):
	# argv[0] : function name, argv[1] : # points to evaluate, argv[2] : # parallel processes
	global num_proc, num_points, function_obj, range_stopping_criteria, path_dir_log_file

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

	# obtain the list of dimensions on which to test it
	dimensions = get_test_dimensions(function_json["dimension"], function_json["minimum_f"])
	print(dimensions)
	return
	

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

		# create the function class' object
		function_obj = ev.objective_function(function_name, dim=dim)

		# if function accepts parameters create the list of parameters on which to test the function
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