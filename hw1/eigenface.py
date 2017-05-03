import scipy.linalg as scl
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg.eigen.arpack import eigsh


def load_data(N_users, n_rows):
	users = []

	for i in range(N_users):
		data = pd.read_csv("User" + str(i+1), sep='\n', header=None, nrows=n_rows)
		users.append(data)

	return users

def construct_cooccurence_matrix(seq, L, w, command_dict):
	k = len(command_dict)
	m = np.zeros((k,k))

	for i in range(len(seq)):
		for j in range(max(0, i - w), i):
			if seq[j] not in command_dict or seq[i] not in command_dict:
				continue
			x = command_dict[seq[j]]
			y = command_dict[seq[i]]
			m[x, y] += 1


	return m

def calculate_mean(matrices):
	mean = matrices[0]
	for i in range(1, len(matrices)):
		mean = np.add(mean, matrices[i])
	mean = np.divide(mean, len(matrices))
	return mean

def convert_to_vector(m):
	l = len(m)
	v = np.reshape(m, l**2)
	return v

def construct_covariance_matrix(matrices):
	vec_mat = np.matrix(matrices)
	return np.dot(vec_mat, vec_mat.T)

def get_most_common(data):
	data = pd.DataFrame(data)
	most_freq = data[0].value_counts()[:200]
	most_freq = pd.DataFrame(most_freq.index) # Outch that's ugly :(
	most_freq = most_freq.to_dict()[0]
	most_freq = {v: k for k, v in most_freq.items()} # :(((
	return most_freq


if __name__ == '__main__':

	N_users = 50
	n_rows = 15000
	L = 100

	print("[+] Loading data")
	users = load_data(N_users, n_rows)
	print("[+] Done loading data")
	comatrices = []


	print("[+] Constructing sequences")
	seqs = []
	tot = []
	for i in range(len(users)):
		offset = 0
		tot.extend(users[i][0])
		for j in range(len(users[i]) // L):
			seqs.append((users[i][offset*L:(offset + 1)*L][0].tolist()))
			offset += 1
	print("[+] Done constructing sequences")

	most_freq = get_most_common(tot)

	print("[+] Constructing co-occurence matrices")
	t1 = time.time()
	for i in range(len(seqs)):
		m = construct_cooccurence_matrix(seqs[i], 100, 7, most_freq)
		comatrices.append(m)

	t2 = time.time()
	print("[+] Done constructing co-occurence matrices in " + str(t2 - t1) + " seconds")

	print("[+] Normalizing matrices")
	t1 = time.time()
	mean = calculate_mean(comatrices)
	for i in range(len(comatrices)):
		comatrices[i] = np.subtract(comatrices[i], mean)
	t2 = time.time()
	print("[+] Done normalizing matrices in " + str(t2 - t1) + " seconds")

	print("[+] Unfolding matrices")
	A = []
	t1 = time.time()
	for i in range(len(comatrices)):
			A.append(convert_to_vector(comatrices[i]))
	t2 = time.time()
	print("[+] Done unfolding matrices in " + str(t2 - t1) + " seconds")

	print("[+] Constructing covariance matrix")
	t1 = time.time()
	cov = construct_covariance_matrix(A)
	t2 = time.time()
	print("[+] Done constructing covariance matrix in " + str(t2 - t1) + " seconds")

	if not np.allclose(cov, cov.T):
		print("[-] Covariance matrix is not symetric ! :(")
		exit(1)


	print("[+] Computing and sorting eingenvalues and eigenvectors")
	t1 = time.time()
	eigenValues, eigenVectors = eigsh(cov, 200, which="LM")

	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	t2 = time.time()
	print("[+] Done computing and sorting eingenvalues and eigenvectors in " + str(t2 - t1) + " seconds")

	print("[+] Plotting contribution rate")
	t1 = time.time()
	sum_eigen = np.sum(eigenValues)
	rates = []
	for i in range(len(eigenValues)):
		rate = 0
		for j in range(i):
			rate += eigenValues[j]
		rate /= sum_eigen
		rates.append(rate)

	plt.plot(rates)
	t2 = time.time()
	print("[+] Done plotting contribution rate in " + str(t2 - t1) + " seconds")
	plt.show()

	
	