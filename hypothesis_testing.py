import nevergrad as ng
import sfu.evaluation_code.functions as ev
import time, sys, os
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
	def __init__(self, arr_inp, arr_res, worker_idx, input_output_index, barrier_compute, barrier_computed, stop_search, compute):
		super().__init__()
		self.arr_inp = arr_inp
		self.arr_res = arr_res
		self.worker_idx = worker_idx
		self.barrier_compute = barrier_compute
		self.barrier_computed = barrier_computed
		self.stop_search = stop_search
		self.input_output_index = input_output_index
		self.compute = compute

	def run(self):
		while 1:
			self.barrier_compute.wait()
			if self.stop_search.value:
				break
			if self.compute[self.worker_idx]:
				print(f"Ecco: {self.input_output_index[self.worker_idx]}")
				#res = evaluate_parallel(self.arr_inp[self.input_output_index[self.worker_idx]][:], [True, self.worker_idx])
				#self.arr_res[self.input_output_index[self.worker_idx]] = res
			self.barrier_computed.wait()

def stop_condition():
	return 1

def main(argv):
	# argv[0] : function name, argv[1] : dimension, argv[2] : # workers, argv[3] : # parallel processes
	function_name = argv[0]
	init_function(function_name)
	dim = int(argv[1])
	min_f = minimum_f(dim)
	print(f"minimum_f: {min_f}")
	optimizer = ng.optimizers.NGOpt(parametrization=dim, budget=10**6)
	num_workers = int(argv[2])
	num_proc = int(argv[3])

	# init shared array 
	arr_inp = [Array("d", [0 for x in range(dim)]) for i in range(num_workers)]
	for i in range(num_workers):
		query = optimizer.ask()
		query = list(*query.args)
		for i_arr in range(dim):
			arr_inp[i][i_arr] = query[i_arr]
	arr_res = Array("d", [0 for x in range(num_workers)])
	input_output_indexes = Array("i", [0 for x in range(num_proc)])
	stop_search = Value("i", 0)
	compute = Array("i", [0 for x in range(num_proc)])
	barrier_compute = Barrier(num_proc+1)
	barrier_computed = Barrier(num_proc+1)
	processes = []
	for x in range(num_proc):
		processes.append(CustomProcess(arr_inp, arr_res, x, input_output_indexes, barrier_compute, barrier_computed, stop_search, compute))

	for proc in processes:
		proc.start()

	# execute num_proc evaluations, for num_workers // num_proc times
	for x in range(num_proc):
		compute[x] = 1
	for idx in range(0, num_workers - (num_workers % num_proc), num_proc):
		for idx_proc in range(num_proc):
			input_output_indexes[idx_proc] = idx + idx_proc
			print(f"assegno: {idx+idx_proc}")
		barrier_compute.wait()
		while(barrier_computed.n_waiting != num_proc):
			pass
		barrier_computed.wait()

	if (num_workers % num_proc != 0):
		# execute the remaning num of workers
		for idx in range(0, num_workers % num_proc):
			input_output_indexes[idx] = (num_workers // num_proc) * num_proc + idx
			print(f"assegno: {(num_workers // num_proc) * num_proc + idx} a index: {idx}")
		print(f"eccoli stop: {num_proc - (num_workers % num_proc)}")
	
		# the remaning processes do not compute
		for idx in range(num_proc - (num_workers % num_proc)):
			print(f"Assegno None a processo: {num_proc - (num_proc - (num_workers % num_proc))+idx}")
			compute[num_proc - (num_proc - (num_workers % num_proc))+idx] = 0
		barrier_compute.wait()
		while(barrier_computed.n_waiting != num_proc):
			pass
		barrier_computed.wait()

		#FUNZIONA!


	return
	while(1):
		if (barrier.n_waiting == num_workers):
			opt_value = min(arr_res)
			for i in range(num_workers):
				candidate = optimizer.parametrization.spawn_child(new_value=arr_inp[i][:])
				optimizer.tell(candidate, arr_res[i])
			for i in range(num_workers):
				query = optimizer.ask()
				query = list(*query.args)
				for i_arr in range(dim):
					arr_inp[i][i_arr] = query[i_arr]
			recommendation = optimizer.provide_recommendation()
			print(f"Call Minimum: {opt_value}\tTemp Minimum: {evaluate_parallel(list(*recommendation.args), [False])}")
			barrier.wait()

		
	for proc in processes:
		proc.join()
	print(arr[:])

if __name__ == "__main__":
	main(sys.argv[1:])
