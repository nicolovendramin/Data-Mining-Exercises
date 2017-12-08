import numpy as np
from progressbar import ProgressBar
import time
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default="base")
parser.add_argument('--bound', type=int, default=10000)
parser.add_argument('--input_file', type=str, default="data.txt")
args = parser.parse_args()

DELETION_PROPORTION = 30

class Triest:

	def __init__(self, M, input_file, dynamic=False):
		self.M = M
		self.graph = self.import_graph(input_file, dynamic)
		self.t_counter = 0
		self.t_dictionary = {}
		self.s = []
		self.di = 0
		self.do = 0
		self.avg_term = 0


	def import_graph(self, filename, dynamic=False):
		file = open(filename, "rU")
		line = file.readline()
		graph = []
		while(len(line)>0): 
			edge = line.split(" ")
			graph.append((edge[0], edge[1]))
			line = file.readline()
		file.close()
		number_of_edges = len(graph)
		print(number_of_edges)
		if dynamic:
			return self.make_graph_dynamic(graph)
		return graph


	def make_graph_dynamic(self, graph):
		graph = [(1, edge) for edge in graph]
		return graph
		graph_ = list(graph)

		for i in range(0, len(graph)):
			if self.flip_biased_coin(DELETION_PROPORTION, 100):
				new_ = (-1, graph[i][1])
				pos = np.random.randint(i, len(graph)+1)
				graph_.insert(pos, new_)

		return graph_


	def sample_edges(self, edge, time_step):
		if time_step <= self.M:
			return True
		elif self.flip_biased_coin(self.M, time_step):
			pos = np.random.randint(len(self.s))
			random_edge = self.s.pop(pos)
			self.update_counters(random_edge, operation=-1)
			return True
		return False


	def sample_edges_improved(self, edge, time_step):
		if time_step <= self.M:
			return True
		elif self.flip_biased_coin(self.M, time_step):
			pos = np.random.randint(len(self.s))
			random_edge = self.s.pop(pos)
			return True
		return False


	def sample_edges_fully_dynamic(self, edge, time_step):
		if self.do + self.di == 0:
			if len(self.s) < self.M:
				self.s.append(edge)
				return True
			elif self.flip_biased_coin(self.M, time_step):
				pos = np.random.randint(len(self.s))
				random_edge = self.s.pop(pos)
				self.update_counters(random_edge, operation=-1)
				self.s.append(edge)
				return True
		elif self.flip_biased_coin(self.di, self.di + self.do):
			self.s.append(edge)
			self.di -= 1
			return True
		else:
			self.do -= 1
			return False


	def flip_biased_coin(self, favorable, possible):
		return np.random.randint(0, possible) < favorable


	def update_counters(self, edge, operation=1):
		neighbors = self.get_neighbors(edge)

		if edge[0] not in self.t_dictionary.keys():
			self.t_dictionary[edge[0]] = 0
		if edge[1] not in self.t_dictionary.keys():
			self.t_dictionary[edge[1]] = 0

		for neighbor in neighbors:
			self.t_counter += 1 * operation
			self.t_dictionary[edge[0]] += 1 * operation
			self.t_dictionary[edge[1]] += 1 * operation

			if neighbor in self.t_dictionary.keys():
				self.t_dictionary[neighbor] += 1 * operation
			else:
				self.t_dictionary[neighbor] = 1 * operation


	def update_counters_improved(self, edge, time_step, operation=1):
		neighbors = self.get_neighbors(edge)

		if edge[0] not in self.t_dictionary.keys():
			self.t_dictionary[edge[0]] = 0
		if edge[1] not in self.t_dictionary.keys():
			self.t_dictionary[edge[1]] = 0

		n = max(1, ((time_step - 1)*(time_step -2)/(self.M*(self.M-1))))

		for neighbor in neighbors:
			self.t_counter += n * operation
			self.t_dictionary[edge[0]] += n * operation
			self.t_dictionary[edge[1]] += n * operation

			if neighbor in self.t_dictionary.keys():
				self.t_dictionary[neighbor] += n * operation
			else:
				self.t_dictionary[neighbor] = n * operation


	def get_neighbors(self, edge):
		n1 = []
		n2 = []
		for edge_ in self.s:
			if edge_[0] == edge[0]:
				n1.append(edge_[1])
			if edge_[1] == edge[0]:
				n1.append(edge_[0])
			if edge_[0] == edge[1]:
				n2.append(edge_[1])
			if edge_[1] == edge[1]:
				n2.append(edge_[0])

		return set(n1) & set(n2)


	def triest_base(self):
		t = 0	
		pbar = ProgressBar()
		for edge in pbar(self.graph):
			t += 1
			if self.sample_edges(edge, t):
				self.s.append(edge)
				self.update_counters(edge)
				epsilon = max(1, (t * (t - 1) * (t - 2)) / (self.M * (self.M - 1) * (self.M -2)))
				self.avg_term = (float((t - 1) * self.avg_term + 1 * self.t_counter * epsilon)) / t

		return self.t_counter


	def triest_improved(self):
		t = 0
		pbar = ProgressBar()
		for edge in pbar(self.graph):
			t += 1
			self.update_counters_improved(edge, t)
			if self.sample_edges_improved(edge, t):
				self.s.append(edge)

		return self.t_counter


	def triest_fully_dynamic(self):
		t = 0	
		s = 0
		pbar = ProgressBar()
		for opt, edge in pbar(self.graph):
			t += 1
			s += 1 * opt
			if opt == +1:
				if self.sample_edges_fully_dynamic(edge, t):
					self.update_counters(edge)
			elif edge in self.s:
				self.update_counters(edge, -1)
				self.s.remove(edge)
				self.di += 1
			else:
				self.do += 1

		return self.t_counter


if args.version == "base":
	print("Running the base version of the algorithm")
	triangle_estimator = Triest(args.bound, args.input_file)
	print(triangle_estimator.triest_base())
elif args.version == "improved":
	print("Running the improved version of the algorithm")
	triangle_estimator = Triest(args.bound, args.input_file)
	print(triangle_estimator.triest_improved())
elif args.version == "fd":
	print("Running the fully dynamic version of the algorithm")
	triangle_estimator = Triest(args.bound, args.input_file, dynamic=True)
	print(triangle_estimator.triest_fully_dynamic())
else:
	print("Version not supported. Try [\"base\", \"imroved\", \"fd\"")
