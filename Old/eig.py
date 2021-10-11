import numpy as np
import pandas as pd
import csv
from util import circ

# The Pauli operators:
px = np.array([[0, 1], [1, 0]])
py = np.array([[0, -1j], [1j, 0]])
pz = np.array([[1, 0], [0, -1]])
# The id operator (dim=2)
id2 = np.eye(2)


class tfd_generator(object):

    def __init__(self):
        calc = self.calc_evals_evecs()
        self.evals0 = calc[0] - min(calc[0])
        self.evecs = calc[1]

        #print('evals0: ', self.evals0, '\n')
        #print('evecs: ', self.evecs, '\n')

        #self.beta = 0
        #self.bad_counts = 0
        #self.prob_for_0000000 = 0

    def pzi(self, i):
        """
        Returns the kronecker product of which the pz operator operates on the i'th qubit
        (notice that the right-most qubit is q0!
        """
        if i == 0:
            return np.kron(id2, np.kron(id2, pz))
        elif i == 1:
            return np.kron(id2, np.kron(pz, id2))
        else:
            return np.kron(pz, np.kron(id2, id2))

    def pxi(self, i):
        """
        Returns the kronecker product of which the px operator operates on the i'th qubit
        (notice that the right-most qubit is q0!
        """
        if i == 0:
            return np.kron(id2, np.kron(id2, px))
        elif i == 1:
            return np.kron(id2, np.kron(px, id2))
        else:
            return np.kron(px, np.kron(id2, id2))


    def outpzipzip1(self, i):
        """
        Returns the kronecker product of which the pz operator operates on the i'th qubit and the i+1 qubit
        (notice that the right-most qubit is q0!
        Also, this is the periodic version (3 -> 1 also exists).
        """
        if i == 0:
            return np.kron(id2, np.kron(pz, pz))
        elif i == 1:
            return np.kron(pz, np.kron(pz, id2))
        else:
            return np.kron(pz, np.kron(id2, pz))


    def ising_ham(self, g, h):
        """ Calculates the Ising Hamiltonian of 3 qubits, given g and h"""
        s = 0
        for i in range(3):
            s -= self.outpzipzip1(i) + g * self.pxi(i) + h * self.pzi(i)
        return s

    def calc_evals_evecs(self):
        #print('ising ham: ', self.ising_ham(-1.05, 0.5), '\n')
        #print(self.ising_ham(-1.05, 0.5) - np.transpose(self.ising_ham(-1.05, 0.5)))
        return np.linalg.eig(self.ising_ham(-1.05, 0.5))

    def chop(self, expr, max=1*pow(10, -10)):
        return [i if i > max else 0 for i in expr]

    def generate_tfd(self, beta):
        self.beta = beta
        acc = [0] * pow(2, 6)
        for i in range(len(self.evecs)):
            acc += np.exp(-beta * self.evals0[i]/2) * np.ndarray.flatten(np.kron(self.evecs[i], self.evecs[i]))

        #print('acc: ', acc, '\n')
        #chopped = self.chop(acc)
        inner_prod = 0
        for i in range(pow(2, 3)):
            inner_prod += np.exp(-beta * self.evals0[i])
        tfd = acc / np.sqrt(inner_prod)

        #normalized_tfd = tfd / np.linalg.norm(tfd)
        return tfd


    def write_to_csv(self, file_name, tfd):
        with open(file_name + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # write the header
            header = ['beta', 'bad_counts', 'probability_of_0000000']
            writer.writerow(header)
            for beta in np.arange(0, 2, 0.25):
                data = circ(tfd, beta)
                print(data)
                writer.writerow(data)


    def generate_graphs(self, file_name):
        """ Generates the graphs of the data in 'file_name'"""
        #TBD

gen = tfd_generator()

"""
# Showing that the eigenstates are not orthogonal. This is confusing.
print(gen.evals0)
for i in range(len(gen.evecs)):
    for j in range(i+1, len(gen.evecs)):
        print('i: ', i, 'j: ', j, ', evecs[i]=evecs[j]: ', [i, j] in [[4, 6], [5, 7]])
        print(np.inner(gen.evecs[i], gen.evecs[j]))
"""
































