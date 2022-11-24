import nevergrad as ng
import sfu.evaluation_code.functions as ev
import time, sys, os, math
from multiprocessing import Process, Queue

# execute N run of nevergrad, in each run we have num_workers parallel evaluation of the function, then we take the minim evaluation among num_workers as temp result of nevergrad

num_proc = 0
num_points = 0
dim = 0
q_inp = None
q_res = None
lower_bound = None
upper_bound = None
range_stopping_criteria = None

def init_function(function):
	json_path = os.path.join(os.path.dirname(__file__),"sfu/functions/functions.json")
	ev.load_json(os.path.join(os.path.dirname(__file__),"sfu/functions/functions.json"))
	ev.select_function(function)

def minimum_f(dim=None):
	return ev.minimum_value(dim)

def proc_function(q_inp, q_res):
	# inp = (index input, input)
	inp = q_inp.get()
	while (inp != None):
		res = ev.evaluate(inp[1], multi_proc=[True, inp[0]])
		q_res.put((inp[1], res))
		inp = q_inp.get()

def init_processes():
	global num_proc, q_inp, q_res
	q_inp = Queue()
	q_res = Queue()
	for x in range(num_proc):
		Process(target=proc_function, args=(q_inp, q_res,)).start()

def kill_processes():
	global num_proc, q_inp
	for x in range(num_proc):
		q_inp.put(None)


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
	global num_points, q_inp, q_res
	results = []
	for x in range(num_points):
		q_inp.put((x, input_points[x]))
	for x in range(num_points):
		results.append(q_res.get())
	return results

def run_nevergrad():
	global lower_bound, upper_bound, range_stopping_criteria

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
		input_points, computed_points = list(zip(*computed_points))
		results.append(min(computed_points))
		update_optimizer(optimizer, input_points, computed_points)

	while (not stopping_criteria(results)):
		print(f"RESULTSS: {results}")
		input_points = obtain_queries(optimizer)
		computed_points = compute_points(input_points)
		input_points, computed_points = list(zip(*computed_points))
		results.append(min(computed_points))
		results = results[1:]
		update_optimizer(optimizer, input_points, computed_points)
	recommendation = optimizer.provide_recommendation()
	return (list(*recommendation.args), ev.evaluate(list(*recommendation.args), multi_proc=[False]))



def hypothesis_testing(delta, epsilon, tolerance = 1e-6):

	N = math.ceil((math.log(delta)/math.log(1-epsilon)))

	# init S
	res = run_nevergrad()

	S_values = [res]
	S = res[1]
	num_iterations = 0
	while (1):
		num_iterations += 1
		string_run = ''
		for run in range(N):

			res = run_nevergrad()
			print(f"Run: {run+1}/{N}\tResult: {res[1]}\tS: {S}")
			string_run = string_run + f"Run: {run+1}/{N}\tResult: {res[1]}\tS: {S}\n"

			# if result smaller than starting S, restart
			if (res[1] + tolerance) < S:
				S_prime = res[1]
				S_values.append(res)
				break

		# if after the loop starting S is equal to S_prime, end algorithm
		if S == S_prime:
			break
		# write temp results 
		with open(f"{os.path.dirname(__file__)}/temp_res/temp.txt", "w") as f:
			f.write(string_run)
			f.close()

	print(f"Num iterations: {num_iterations}")
	print(f"S values: {S_values}")
	return S_values[-1]

def main(argv):
	# argv[0] : function name, argv[1] : dimension, argv[2] : # points to evaluate, argv[3] : # parallel processes
	global num_proc, num_points, dim, lower_bound, upper_bound, range_stopping_criteria

	start_time = time.time()
	
	# init the module which compute the function and the infos about it
	function_name = argv[0]
	init_function(function_name)
	dim = int(argv[1])
	min_f = minimum_f(dim)
	num_points = int(argv[2])
	range_f = ev.input_domain(dim)
	bounds = [[],[]]
	for x in range(dim):
		bounds[0].append(range_f[x][0])
		bounds[1].append(range_f[x][1])
	lower_bound = None#bounds[0]
	upper_bound = None#bounds[1]
	range_stopping_criteria = 20

	num_proc = int(argv[3])
	init_processes()

	print(f"minimum_f: {min_f}")
	print(f"Lower bounds: {bounds[0]}\tUpper bounds: {bounds[1]}")
	print(f"RESULT hypothesis testing: {hypothesis_testing(0.001, 0.001)}")
	kill_processes()
	print(f"Time: {time.time() - start_time}")


	

if __name__ == "__main__":
	main(sys.argv[1:])
