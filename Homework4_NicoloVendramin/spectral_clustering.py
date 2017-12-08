import numpy as np
from progressbar import ProgressBar
from scipy.sparse import csr_matrix
from scipy.linalg import fractional_matrix_power
import time
import math as m
import argparse 
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--clusters', type=int, default=4)
parser.add_argument('--input_file', type=str, default="real_graph.txt")
parser.add_argument('--plot', type=str, default="all")

args = parser.parse_args()


def import_graph(input_file):
	file = open(input_file, "rU")
	line = file.readline()
	first_edges = []
	second_edges = []
	affinity_scores = []
	edge_list = []
	affinity_dictionary = {}

	while(len(line)>0): 
		infos = line.split(",")
		edge_list.append((int(infos[0]),int(infos[1])))
		
		first_edges.append(int(infos[0])-1)
		second_edges.append(int(infos[1])-1)
		if len(infos) > 2:
			affinity_scores.append(int(infos[2]))
		else:
			affinity_scores.append(1)

		line = file.readline()

	affinity_matrix = csr_matrix((affinity_scores, (first_edges, second_edges)))

	return edge_list, affinity_matrix


def spectral_clustering(affinity_matrix, k=5):
	d_elements = np.ravel(affinity_matrix.sum(axis=1))
	d_matrix = csr_matrix((d_elements, (np.arange(0, d_elements.shape[0]), (np.arange(0, d_elements.shape[0])))))
	
	d_neg_sqrt = fractional_matrix_power(d_matrix.todense(), -0.5)
	L = d_neg_sqrt.dot(affinity_matrix.todense()).dot(d_neg_sqrt)

	w, v = np.linalg.eig(L)
	top_eigenvalues_indices = np.argsort(-w)[:k]

	x = w[top_eigenvalues_indices[0]]*v[:,top_eigenvalues_indices[0]]

	for index in top_eigenvalues_indices[1:]:
		x = np.append(x, w[index]*v[:,index], 1)

	factors = []
	for row in x:
		factor = np.power(row, 2).sum()
		factors.append(1/m.sqrt(factor))

	y = np.diag(np.asarray(factors)) * x
	
	kmeans = KMeans(n_clusters=k, random_state=0).fit(y)
	
	return kmeans.labels_


def graph_plot(graph, (clusters, k)=(None, 1)):
	G = nx.from_edgelist(graph)
	plt.figure(1)
	values = []
	if clusters != None:
		for node in G.nodes():
			values.append(clusters.get(int(node), k+1))
		nx.draw(G, cmap=plt.get_cmap('jet'), node_color=values, node_size=80, with_labels=False)
	else:
		nx.draw_networkx(G,node_color='b', node_size=80, with_labels=False)
	plt.show()


def main():
	edge_list, affinity_matrix = import_graph(args.input_file)
	clusters = spectral_clustering(affinity_matrix, args.clusters)
	print(clusters)
	cl = {i+1: clusters[i] for i in range(0, clusters.shape[0])}
	print(cl)
	if args.plot == "normal":
		graph_plot(edge_list)
	elif args.plot == "clustered":
		graph_plot(edge_list, cl, args.clusters)
	elif args.plot == "all":
		graph_plot(edge_list)
		graph_plot(edge_list, (cl, args.clusters))
	else:
		print("Plot option not supported. Try [\"normal\", \"clustered\", \"all\"].")


main()



