import nevergrad as ng
import sfu.evaluation_code.functions as ev
import time, sys, os, math
from multiprocessing import Process, Queue
from memory_profiler import memory_usage

# execute N run of nevergrad, in each run we have num_workers parallel evaluation of the function, then we take the minim evaluation among num_workers as temp result of nevergrad

num_proc = 0
num_points = 0
dim = 0
parameters = None
q_inp = None
q_res = None
lower_bound = None
upper_bound = None
range_stopping_criteria = None
path_log_file = None
log_string = ""

def init_function(function):
	json_path = os.path.join(os.path.dirname(__file__),"sfu/functions/functions.json")
	ev.load_json(os.path.join(os.path.dirname(__file__),"sfu/functions/functions.json"))
	ev.select_function(function)

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
	global num_proc, q_inp
	for x in processes:
		x.kill()


def entropy(samples):
	sum_samples = sum([math.exp(x) for x in samples])
	prob_samples = [math.exp(x) / sum_samples for x in samples]
	return - sum([x * math.log2(x) for x in prob_samples])

def stopping_criteria(samples):
	if entropy(samples) > math.log2(len(samples)) - 1e-10:
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
	global parameters
	results = []
	for inp in input_points:
		results.append(ev.evaluate(inp, param=parameters))
	return results

def run_nevergrad():
	global lower_bound, upper_bound, range_stopping_criteria, dim, log_string

	if lower_bound != None and upper_bound != None:
		param = ng.p.Array(shape=(dim,), lower=lower_bound, upper=upper_bound)
	else:
		param = ng.p.Array(shape=(dim,))

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
	return (optimizer.num_ask, list(*recommendation.args), ev.evaluate(list(*recommendation.args)))


def write_log_file(path, string_res):
	# write temp results
	with open(path, "w") as f:
		f.write(string_res)
		f.close()


def hypothesis_testing(delta, epsilon, tolerance = 1e-6):
	global q_inp, q_res, dim, parameters, log_string, num_proc
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
	temp_string = f"FIRST RESULT: {res[1][2]}\tNUM OF ASKS: {res[1][0]}\tTIME: {res[0][0]}\tMAX RAM USAGE: {res[0][1]} MB"
	print(temp_string)
	log_string = log_string + '\n' + temp_string
	res = res[1]
	S_values = [res]
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
			temp_string = f"RESULT: {res[1][2]}\tS: {S}\tNUM OF ASKS: {res[1][0]}\tITERATION: {counter_samples}/{N}\tTime: {res[0][0]}\tMAX RAM USAGE: {res[0][1]} MB"
			print(temp_string)
			log_string = log_string + '\n' + temp_string
			res = res[1]
			

			# if result smaller than starting S, restart
			if (res[2] + tolerance) < S:
				S_prime = res[2]
				S_values.append(res)
				write_string = ""
				break

		# if after the loop starting S is equal to S_prime, end algorithm
		if S == S_prime:
			break
		

	temp_string = f"Num external iterations: {num_iterations}\nNum internal iterations: {num_iter_internal}\nRESULT: {S_values[-1]}\nTime: {total_process_time}\nMEAN TIME PER PROCESS: {total_process_time/num_proc}\nMAX RAM USAGE: {max_ram_usage} MB\nS values: {S_values}"
	log_string = log_string + "\n" + temp_string
	print(temp_string)
	return (total_process_time, S_values[-1])

def main(argv):
	# argv[0] : function name, argv[1] : # points to evaluate, argv[2] : # parallel processes
	global num_proc, num_points, dim, parameters, lower_bound, upper_bound, range_stopping_criteria, path_log_file, log_string

	processes_time = 0
	path_dir_res = os.path.join(os.path.dirname(__file__),"log_results")
	if not os.path.exists(path_dir_res):
		os.mkdir(path_dir_res)
	# init the module which compute the function and the infos about it
	function_name = argv[0]
	init_function(function_name)
	param_a = [20, 50]
	param_b = [0.2, 0.8]
	param_c = [2*math.pi, 7*math.pi]
	dimensions = [2] + [10**(x+1) for x in range(6)]
	num_points = int(argv[1])
	num_proc = int(argv[2])
	range_stopping_criteria = 20
	delta = 0.001
	epsilon = 0.001
	for temp_dim in dimensions:
		min_f = minimum_f(temp_dim)
		path_dir_res_dim = os.path.join(path_dir_res, f"dim_{temp_dim}")
		if not os.path.exists(path_dir_res_dim):
			os.mkdir(path_dir_res_dim)
		for par_a in param_a:
			for par_b in param_b:
				for par_c in param_c:
					dim = temp_dim
					parameters = {"a" : par_a, "b" : par_b, "c" : par_c}
					log_string = f"TESTING FUNCTION DIM: {temp_dim} PARAM_A: {par_a} PARAM_B: {par_b} PARAM_C {par_c} OPT_POINT: {min_f}\n"
					path_log_file = os.path.join(path_dir_res_dim, f"log_dimension-{dim}_param_a-{par_a}_param_b-{par_b}_param_c-{str(par_c).replace('.','-')}.txt")
					print(log_string)
					processes = init_processes()
					hypo_testing_res = hypothesis_testing(delta, epsilon)
					temp_string = f"ERROR: {abs(min_f - hypo_testing_res[1][2])}"
					print(temp_string + "\n\n\n")
					log_string = log_string + "\n" + temp_string
					processes_time += hypo_testing_res[0]
					kill_processes(processes)
					write_log_file(path_log_file, log_string)

	print(f"TOTAL TIME: {processes_time}\tMEAN PER PROCESS TIME: {processes_time/num_proc}")
	

if __name__ == "__main__":
	main(sys.argv[1:])
