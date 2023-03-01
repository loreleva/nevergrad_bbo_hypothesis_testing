from multiprocessing import Process, Queue

def init_processes(num_proc, multiproc_function, function_obj, num_points):
	q_inp = Queue()
	q_res = Queue()
	processes = []
	for x in range(num_proc):
		processes.append(Process(target=multiproc_function, args=(q_inp, q_res, function_obj, num_points)))
		processes[x].start()
	return q_inp, q_res, processes

def kill_processes(processes):
	for x in processes:
		x.kill()