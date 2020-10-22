import numpy
from pyscf import gto, scf, ao2mo


def get_mo(
        h1e,  # array[n], one-body part of the hamiltonian, not including spin.
        v2e,  # array[n,n,n,n], two-boidy part
        ne,   # number of electrons
        max_cycle=100,  # maximal steps for the scf calculation
        init_guess='2e',  # initial guess for the hf scf calculation.
        ):
    '''get spin-restricted Hartree-Fock molecular orbitals.
    '''
    # set up a pseudo molecule
    mol = gto.M()
    mol.nelectron = ne
    # instantiate a spin-restricted Hartree-Fock calculation class.
    mf = scf.RHF(mol)
    # set up interface for one-body part
    mf.get_hcore = lambda *args: h1e
    # basis overlap
    mf.get_ovlp = lambda *args: numpy.eye(h1e.shape[0])
    # two-body part
    mf._eri = ao2mo.restore(8, v2e, h1e.shape[0])
    mf.init_guess = init_guess
    mf.max_cycle = max_cycle
    mf.kernel()
    # get occupied molecular orbitals
    mo_coeff_occ = mf.mo_coeff[ne//2:]
    return mo_coeff_occ



if __name__ == "__main__":
    # 1D Hubbard model at half filling
    n = 4
    h1 = numpy.zeros((n, n))
    for i in range(n-1):
        h1[i, i+1] = h1[i+1, i] = -1.0
    # h1 = numpy.array([  [ 0, -1,  0, -0.99],
    #                     [-1,  0, -0.99,  0],
    #                     [ 0, -0.99,  0, -1],
    #                     [-0.99,  0, -1,  0]])
    print(h1)
    eri = numpy.zeros((n, n, n, n))
    for i in range(n):
        if (i==0):
            eri[i, i, i, i] = 1.0
        else:
            eri[i, i, i, i] = 1.0
    # print(eri)
    mo = get_mo(h1, eri, n)
    print(repr(mo))
