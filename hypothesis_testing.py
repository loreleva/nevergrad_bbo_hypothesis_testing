import nevergrad as ng
import sfu.evaluation_code.functions as ev
import time, sys, os, math
from multiprocessing import Process, Array, Value, Barrier

# execute N run of nevergrad, in each run we have num_workers parallel evaluation of the function, then we take the minim evaluation among num_workers as temp result of nevergrad

def init_function(function):
	json_path = os.path.join(os.path.dirname(__file__),"sfu/functions/functions.json")
	ev.load_json(os.path.join(os.path.dirname(__file__),"sfu/functions/functions.json"))
	ev.select_function(function)

def minimum_f(dim=None):
	return ev.minimum_value(dim)

def evaluate_parallel(inp, multi_proc):
	return ev.evaluate(inp, multi_proc=multi_proc)

def entropy(samples):
	sum_samples = sum([math.exp(x) for x in samples])
	prob_samples = [math.exp(x) / sum_samples for x in samples]
	return - sum([x * math.log2(x) for x in prob_samples])

class CustomProcess(Process):
	def __init__(self, arr_inp, var_res, process_idx, barrier_compute, barrier_computed, stop_search, compute):
		super().__init__()
		# array which contain the function input to compute
		self.arr_inp = arr_inp
		# variable in which the process write its result
		self.var_res = var_res
		# idx of the process, used in the computation of the function in parallel
		self.process_idx = process_idx
		# barrier used by the father process in order to fill the arr_inp of the process that must perform the computation
		self.barrier_compute = barrier_compute
		# barrier used by the processes in order to comunicate the result of the computation
		self.barrier_computed = barrier_computed
		# boolean variable representing the stopping condition
		self.stop_search = stop_search
		# 
		self.compute = compute

	def run(self):
		while 1:
			# wait to receive data to compute
			self.barrier_compute.wait()

			# if stopping condition, exit
			if self.stop_search.value:
				break
			if self.compute.value:
				res = evaluate_parallel(self.arr_inp[:], [True, self.process_idx])
				self.var_res.value = res
			self.barrier_computed.wait()

def stop_condition(results, num_values):
	if len(results) < num_values:
		return 0
	if entropy(results[-20:]) > math.log2(20) - 0.1:
		return 1
	return 0

def run_nevergrad_parallel(num_points, num_proc, dim):
	# init nevergrad optimizer
	optimizer = ng.optimizers.NGOpt(parametrization=dim, budget=10**6)

	# init of the multiprocessing
	# arr_inp is a list which contains num_proc n-dimensional arrays, each shared array will contain the function input which the process must compute
	arr_inp = [Array("d", [0 for x in range(dim)]) for i in range(num_proc)]
	
	
	# list of shared values of the evaluation of the function, list of dimension num_points
	arr_res = [Value("d", 0) for x in range(num_points)]
	# shared variable for the stopping condition
	stop_search = Value("i", 0)
	#
	compute = [Value("i", 1) for x in range(num_proc)]
	# barrier where the main process has filled the arr_inp values for the sub-processes
	barrier_compute = Barrier(num_proc+1)
	# barrier where the sub-processes have written their results in arr_res
	barrier_computed = Barrier(num_proc+1)
	# list of parallel processes
	processes = []
	# create processes objects
	for x in range(num_proc):
		processes.append(CustomProcess(arr_inp[x], arr_res[x], x, barrier_compute, barrier_computed, stop_search, compute[x]))

	for proc in processes:
		proc.start()

	results = []
	while (not stop_search.value):
		# list which contains num_points array of function inputs
		computed_points = []
		input_points = []
		for i in range(num_points):
			query = optimizer.ask()
			query = list(*query.args)
			input_points.append(query)

		# execute num_proc evaluations, for num_points // num_proc times
		for idx in range(0, num_points - (num_points % num_proc), num_proc):
			for idx_proc in range(num_proc):
				for idx_input in range(dim):
					arr_inp[idx_proc][idx_input] = input_points[idx+idx_proc][idx_input]
			barrier_compute.wait()
			while(barrier_computed.n_waiting != num_proc):
				pass
			for idx_proc in range(num_proc):
				computed_points.append(arr_res[idx_proc].value)
			barrier_computed.wait()

		# compute rest of the points
		if (num_points % num_proc != 0):
			# execute the remaning num of workers
			for idx in range(num_points - (num_points % num_proc), num_points):
				for idx_input in range(dim):
					arr_inp[num_points-idx-1][idx_input] = input_points[idx][idx_input]
			
			# do not assign computation to the remaining processes
			for idx in range((num_points % num_proc), num_proc):
				compute[idx].value = 0

			# perform computation
			barrier_compute.wait()
			while(barrier_computed.n_waiting != num_proc):
				pass
			for idx in range(num_points - (num_points % num_proc), num_points):
				computed_points.append(arr_res[num_points-idx-1].value)
			barrier_computed.wait()

			for x in range((num_points % num_proc), num_proc):
				compute[x].value = 1

		for i in range(num_points):
			candidate = optimizer.parametrization.spawn_child(new_value = input_points[i])
			optimizer.tell(candidate, computed_points[i])

		# append to the results list the minimum point found
		results.append(min(computed_points))

		if stop_condition(results, 20):
			stop_search.value = 1
			barrier_compute.wait()
		if len(results) >= 20:
			print(results[-20:])
	
	for proc in processes:
		proc.join()
	recommendation = optimizer.provide_recommendation()
	return (list(*recommendation.args), evaluate_parallel(list(*recommendation.args), [False]))

def hypothesis_testing(delta, epsilon, num_points, num_proc, dim):
	N = math.ceil((math.log(delta)/math.log(1-epsilon)))
	print(f"N: {N}")
	S = run_nevergrad_parallel(num_points, num_proc, dim)[1]
	while (1):
		results = []
		for x in range(N):
			results.append(run_nevergrad_parallel(num_points, num_proc, dim)[1])
		S_prime = min(results)
		if S_prime < S:
			S = S_prime
		else:
			break
		print(f"S_prime: {S_prime}\tS: {S}")
	return S

def main(argv):
	# argv[0] : function name, argv[1] : dimension, argv[2] : # points to evaluate, argv[3] : # parallel processes

	# init of the function to evaluate
	function_name = argv[0]
	# init the module which compute the function and the infos about it
	init_function(function_name)
	dim = int(argv[1])
	min_f = minimum_f(dim)
	print(f"minimum_f: {min_f}")
	print(f"RESULT: {hypothesis_testing(0.001, 0.001, int(argv[2]), int(argv[3]), dim)}")


	

if __name__ == "__main__":
	main(sys.argv[1:])
