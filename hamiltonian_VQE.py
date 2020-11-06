from pyquil.paulis import *
import qulacs as q
from openfermion.utils import slater_determinant_preparation_circuit
from qulacs.observable import create_observable_from_openfermion_text
from qulacs import Observable, QuantumStateGpu, QuantumCircuit, QuantumState, PauliOperator, GeneralQuantumOperator
import numpy as np
from scipy.optimize import minimize
import math
import random
from hf import prepare_Q_matrix

class Hamiltonian_VQE():

    def __init__(self, h1e, v2e, gpu=False):
        self.N = len(h1e)
        self.model = []
        hoppings = np.around(h1e, 10) * (-np.ones(4) + 2 * np.eye(4))
        for i in range(self.N):
            self.model.append({
                "U": float(v2e[i,i,i,i]), "neighbors": hoppings[i].nonzero()[0].tolist(), "hoppings": hoppings[i].tolist()
            })
        self.hamiltonian = self.__build_hamiltonian()
        self.h1e = h1e
        self.v2e = v2e
        self.Q = prepare_Q_matrix(self.h1e, self.v2e, self.N, self.N)
        self.gpu = gpu

    def run_VQE(self, S, param_init=None):
        # convert to hopping direction matrix
        simplified_h_t = self.h1e
        for i in range(self.N):
            simplified_h_t[i][self.N-1-i] *= -1

        hops = []
        for j in range(self.N):
            n = self.model[j]["neighbors"]
            for k in range(len(n)):
                # if a hopping is not already in the array i.e. don't add (3,0) if (0,3) is there
                if (j, n[k]) in hops or (n[k], j) in hops or j == n[k]:
                    pass
                else:
                    hops.append((j, n[k]))
        self.h_hops = []
        self.v_hops = []
        for hop in hops:
            if (simplified_h_t[hop[0]][hop[1]] < 0):
                self.h_hops.append(hop)
            else:
                self.v_hops.append(hop)
        self.S = S
        self.res = self.__global_variational(param_init)
        return self.res

    def __build_hamiltonian(self):
        def U():
            """
            Builds the interaction term: a_p^dagger a_q^dagger a_q a_p
            """    
            hamiltonian = ZERO()
            for i in range(self.N):
                hamiltonian += (self.model[i]["U"] / 4) * (sI() - sZ(i) - sZ(i+self.N) + sZ(i)*sZ(i+self.N))
                
            return hamiltonian

        def t():
            """
            Builds the hopping term: a_p^dagger a_q
            """
            def op(s1, s2):
                """
                returns a PauliSum representing a_s1^daggar a_s2
                """    
                # for all sites in between site one (s1) and site two (s2), multiply by sigma z
                z = sI()
                for i in range(s1+1, s2):
                    z *= PauliTerm('Z', i)

                return sX(s1) * z * sX(s2) + sY(s1) * z * sY(s2)
                
            hops = []
            for i in range(self.N):
                n = self.model[i]["neighbors"]
                for j in range(len(n)):
                    # if a hopping is not already in the array i.e. don't add (3,0) if (0,3) is there
                    if (i, n[j]) in hops or (n[j], i) in hops or i == n[j]:
                        pass
                    else:
                        hops.append((i, n[j]))
                        
            hamiltonian = ZERO()
            for hop in hops:
                # add hoppings for up and down spins
                hamiltonian += (-self.model[hop[0]]["hoppings"][hop[1]]) * op(hop[0], hop[1])
            for hop in hops:
                hamiltonian += (-self.model[hop[0]]["hoppings"][hop[1]]) * op(hop[0]+self.N, hop[1]+self.N)
            return hamiltonian * (1/2)

        def e():
            """
            Builds the number term: a_p^dagger a_p
            """
            hamiltonian = ZERO()
            for i in range(N):
                hamiltonian += (self.model[i]['hoppings'][i] / 2) * (2*sI() - sZ(i) - sZ(i+self.N))

            return hamiltonian

        hamU = U()
        hamt = t()
        hame = e()
        hamiltonian = hamU + hamt + hame
        ham_str = self.__pauli_to_qulacs_string(hamiltonian)
        hamiltonian = create_observable_from_openfermion_text(ham_str)
        return hamiltonian

    def __prepare_slater_determinant(self, c): 
        """
        Prepares the Slater determinant as described in https://arxiv.org/pdf/1711.05395.pdf
        
        Args:
            Q: The (N_f x N) matrix Q with orthonormal rows which describes the Slater determinant to be prepared.
            c: Program to append to
        Returns:
            c: A program that applies the sequence of Givens rotations returned 
                from slater_determinant_preparation_circuit
        """
        # Defining a controlled-RY gate    
        def CRY(control, target, angle):
            ry = q.gate.RY(target, angle*2)
            cry = q.gate.to_matrix_gate(ry)
            cry.add_control_qubit(control, 1) # add_control_qubit(control_index, control_with value) 
            return cry
        
        # Q is a (N_f x N) matrix
        # N = Q[0].size
        N_f = len(self.Q)
        
        givens = slater_determinant_preparation_circuit(self.Q)

        def givens_rotation(tups, spin, c):
            """
            Performs Givens rotations
            
            Args:
                tups: tuple containing Givens rotations to be performed together
                spin: 0 represents up spin, and 1 represents down spin
            Returns:
                p: A program that applies the Givens rotations in tups
            """
            for tup in tups:
                # where tup is (j, k, theta, phi)
                c.add_CNOT_gate(tup[1]+self.N*spin, tup[0]+self.N*spin)

                # controlled-RY
                c.add_gate(CRY(tup[0]+self.N*spin, tup[1]+self.N*spin, tup[2]))
                c.add_CNOT_gate(tup[1]+self.N*spin, tup[0]+self.N*spin)
                c.add_RZ_gate(tup[1]+self.N*spin, tup[3]) # all psis are zero, anyway       
            return c
    
        # Fill first N_f orbitals for each spin
        for i in range(N_f):
            c.add_X_gate(i)
            c.add_X_gate(i+self.N)

        # Perform Givens rotations for up and down spins
        for rot in givens:
            givens_rotation(rot, 0, c)
            givens_rotation(rot, 1, c)
        return c
    
    def __var_ansatz(self, params):
        """
        Prepares $\sum_S [U_U(\theta_U / 2) U_h(\theta_h) U_v(\theta_v) U_U(\theta_U / 2)] | \Psi_I \rangle$
        Where $| \Psi_I \rangle$ is the Slater determinant
        """

        def U_ansatz(param, c):
            """
            Creates a circuit for e^{i \theta_U H_U}
            """
            for i in range(self.N):
                c.add_RZ_gate(i, self.model[i]["U"]*param/4)
                c.add_RZ_gate(i+self.N,  self.model[i]["U"]*param/4)
                c.add_CNOT_gate(i, i+self.N)
                c.add_RZ_gate(i+self.N, -self.model[i]["U"]*param/4)
                c.add_CNOT_gate(i, i+self.N)    
            return c

        def t_ansatz(s1, s2, param, c):
            """
            Creates a circuit for e^{i \theta_t H_t} for horizontal and vertical hoppings
            """    
            for i in range(s1+1, s2):
                if (i == s2-1):
                    c.add_CZ_gate(i, i+1)
                else:
                    c.add_CNOT_gate(i, i+1)
            
            c.add_H_gate(s1)
            c.add_H_gate(s2)
            c.add_CNOT_gate(s1, s2)
            c.add_RZ_gate(s2, param)
            c.add_CNOT_gate(s1, s2)
            c.add_H_gate(s1)
            c.add_H_gate(s2)
            c.add_RX_gate(s1, -np.pi/2)
            c.add_RX_gate(s2, -np.pi/2)
            c.add_CNOT_gate(s1, s2)
            c.add_RZ_gate(s2, param)
            c.add_CNOT_gate(s1, s2)
            c.add_RX_gate(s1, -np.pi/2)
            c.add_RX_gate(s2, -np.pi/2)
            
            for i in reversed(range(s1+1, s2)):
                if (i == s2-1):
                    c.add_CZ_gate(i, i+1)
                else:
                    c.add_CNOT_gate(i, i+1)
            return c

        c = QuantumCircuit(self.N*2)
        c = self.__prepare_slater_determinant(c)
        
        for i in range(self.S):
            # building U_u
            c = U_ansatz(params[3*i], c)

            # building U_t, horizontal and vertical
            # up spin horizontal hops
            for hop in self.h_hops:
                c = t_ansatz(hop[0], hop[1], params[3*i+1], c)
            # down spin horizontal hops
            for hop in self.h_hops:
                c = t_ansatz(hop[0]+self.N, hop[1]+self.N, params[3*i+1], c)

            # up spin vertical hops
            for hop in self.v_hops:
                c = t_ansatz(hop[0], hop[1], params[3*i+2], c)
            # down spin vertical hops
            for hop in self.v_hops:
                c = t_ansatz(hop[0]+self.N, hop[1]+self.N, params[3*i+2], c)  
        
            # building U_u
            c = U_ansatz(params[3*i], c)
            
        return c

    def __pauli_to_qulacs_string(self, ham_str):
        # converting pyquil pauliSum to the form qulacs needs

        ham_str = str(ham_str)
        ham_str = ham_str.replace("*I", "[I]")
        for p in ["X", "Y", "Z"]:
            for i in reversed(range(self.N*2)):            
                ham_str = ham_str.replace("*{}{}".format(p, i), "[{}{}]".format(p, i))
        ham_str = ham_str.replace("][", " ")
        ham_str = ham_str.replace(" + ", " +\n")
        return ham_str

    def __expectation(self, x):
        """
        Expectation value for parameters, x
        """
        if (self.gpu):
            state = QuantumStateGpu(self.N*2)
        else:
            state = QuantumState(self.N*2)
        c = self.__var_ansatz(x)
        c.update_quantum_state(state)
        return self.hamiltonian.get_expectation_value(state)

    def __global_variational(self, initial_params=None):
        """
        Complete a round of greedy and Powell's optimization for six randomly chosen points
        Further optimizes the best point
        """

        def greedy_noisy_search(initial_state):
            """
            Slightly perturb the values of the points, accepting whenever this results in a lower energy
            Total of 150 evaulations of the energy
            Change step size after 30 evaluations based on number of acceptances in previous trial group
            """
            params = initial_state
            min_energy = self.__expectation(initial_state)

            def random_point(dim, step):
                """
                Generates a random point on the perimeter of a circle with radius, step
                """
                coords = [random.gauss(0, 1) for i in range(dim)]
                norm = math.sqrt(sum([i**2 for i in coords]))
                coords = [(i / norm) * step for i in coords]
                return coords

            step = 0.2
            acceptances = 0

            # Five groups of 30 trials
            for i in range(5):
                acceptances = 0
                for j in range(30):
                    
                    # Slightly perturb the values of the points
                    coords = random_point(len(initial_state), step)
                    temp_params = [sum(x) for x in zip(params, coords)]
                    
                    # Calculate expectation value with new parameters
                    temp_energy = self.__expectation(temp_params)
                    
                    # Greedily accept parameters that result in lower energy
                    if (temp_energy < min_energy):
                        min_energy = temp_energy
                        params = temp_params
                        acceptances += 1
                # Update step size
                step *= (acceptances / 15)

            return {'x': params, 'energy': min_energy, 'step': step}
        
        def scipy_minimize(x):
            """
            Minimizes expectation value using parameters found in greedy search as starting point
            Powell's conjugate direction method
            """
            return minimize(self.__expectation, x, method='Powell')

        if (initial_params == None):
            points = [np.random.uniform(-1, 1, self.S*3) for i in range(6)]
        else:
            points = [initial_params]

        greedy_results = [greedy_noisy_search(x) for x in points]
        # print("Done with greedy search")
        # change it to only optimize the lowest three energies
        results = [scipy_minimize(i['x']) for i in sorted(greedy_results, key=lambda x:x['x'])]
        # results = [scipy_minimize(i['x']) for i in greedy_results]
        # print("Done with Powell's")
        res = min(results, key=lambda x:x.fun)
        # For the chosen point, we continue optimizing until we cannot find improvement
        res_copy = res
        energy = res_copy.fun
        while (True):
            result = greedy_noisy_search(res_copy.x)
            temp_res = scipy_minimize(result['x'])
            temp_energy = temp_res.fun

            # Optimizing will usually find an energy ~1e-6 lower
            # Stop it eventually
            tolerance = 0.01
            if (abs(temp_energy - energy) > tolerance):
                res_copy = temp_res
                energy = temp_energy
            else:
                break
        return res_copy

    def calculate_spdm(self):
        def op(s1, s2):
            # returns a GeneralQuantumOperator representing a_s1^daggar a_s2
            def sigma_plus(s):
                # returns sigma^+ operator on side s
                # sigma^+ = 1/2(X + iY)
                # return (0.5 * sX(s)) + (0.0+0.5j * sY(s))
                return PauliTerm('X', s, 0.5) + PauliTerm('Y', s, complex(0.0+0.5j))
            
            def sigma_minus(s):
                # returns sigma^- operator on side s
                # sigma^- = 1/2(X - iY)
                # return (0.5 * sX(s)) - (0.0+0.5j * sY(s))
                return PauliTerm('X', s, 0.5) - PauliTerm('Y', s, complex(0.0+0.5j))
            
            # for all sites in between site one (s1) and site two (s2), multiply by sigma z
            z = sI()
            # for i in range(s1+1, s2):
            #     z *= PauliTerm('Z', i)
            pauli_string = sigma_minus(s1) * z * sigma_plus(s2)    
            return pauli_string

        if (self.gpu):
            state = QuantumStateGpu(self.N*2)
        else:
            state = QuantumState(self.N*2)
        c = self.__var_ansatz(self.res.x)   
        c.update_quantum_state(state)

        spdm = np.zeros((self.N*2, self.N*2))
        for i in range(self.N*2):
            for k in range(self.N*2):
                # building an operator and vqe for each i, j        
                # element (j, i) in the spdm is a_i^dagger a_j

                # pretty hacky, but it seems as though this general operator works
                operator = GeneralQuantumOperator(self.N*2)
                pauli_string = op(i, k)
                for term in pauli_string:
                    coeff = term.coefficient
                    new_str = str(term)
                    new_str = new_str[new_str.index('*')+1:].replace('*', ' ').replace('X', 'X ').replace('Y', 'Y ').replace('Z', 'Z ')
                    operator.add_operator(coeff, new_str)
                
                spdm[i, k] = operator.get_expectation_value(state).real
        return spdm


if __name__ == "__main__":
    N = 4
    h1e = np.array(
        [[-2.500000000000e-01,  4.000000000000e-01, -1.153709150317e-16,   8.230615837392e-17],
        [ 4.000000000000e-01 , 0.000000000000e+00,  4.495730075021e-02 , -4.811263437330e-01],
        [-1.153709150317e-16 , 4.495730075021e-02,  8.212469092606e-02 , -4.356017744890e-01],
        [ 8.230615837392e-17 ,-4.811263437330e-01, -4.356017744890e-01 , -8.212245620464e-02]]
    )
    v2e = np.zeros((N, N, N, N))
    for i in range(N):
        if (i==0):
            v2e[i, i, i, i] = 0.5
        else:
            v2e[i, i, i, i] = 0.0

    vqe = Hamiltonian_VQE(h1e, v2e, gpu=False)
    print("Starting VQE")
    res = vqe.run_VQE(4) # put in number of Trotter steps, S
    print(res)
    spdm = vqe.calculate_spdm()
    print(np.around(spdm, decimals=3))