import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

import diatomic
from diatomic.systems import SingletSigmaMolecule
import diatomic.operators as operators
import diatomic.calculate as calculate

#%%
GAUSS = 1e-4  # T
MHz = scipy.constants.h * 1e6
muN = scipy.constants.physical_constants["nuclear magneton"][0]

# Define helpful constants
pi = scipy.constants.pi
bohr = scipy.constants.physical_constants["Bohr radius"][0]
eps0 = scipy.constants.epsilon_0
to_cgs = 4 * pi * eps0 * bohr**3

kWpercm2 = 1e7

# Set logging
diatomic.configure_logging()

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 4

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)
Hac1065 = operators.ac_ham(mol, mol.a02[1065], beta=0)

INTEN1065 = np.linspace(0, 17.1 * kWpercm2, 20)

B=181.699*GAUSS

# Overall Hamiltonian
Htot = H0 + Hz * B + Hac1065 * INTEN1065
#%%
# Solve (diagonalise) Hamiltonians
eigenenergies, eigenstates = calculate.solve_system(Htot)

# Apply labels (in some way arbitrary) warn if duplicate
eigenlabels = calculate.label_states(mol, eigenstates[0], ["N", "MF"])

magnetic_moments = calculate.magnetic_moment(mol, eigenstates)

#%%
def label_to_indices(labels, N, MF=None):
    labels = np.asarray(labels)
    if MF is not None:
        indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
    else:
        indices = np.where((labels[:, 0] == N))[0]
    return indices


#%%
#0,5
#1,6
#2,7
#3,8

N_INITIAL = 3
MF_INITIAL = 8
state = int(label_to_indices(eigenlabels,N_INITIAL,MF_INITIAL)[0])

transition_sigma_plus = calculate.transition_electric_moments(
    mol, eigenstates[:, :, :], h=1, from_states=state
)
transition_pi = calculate.transition_electric_moments(
    mol, eigenstates[:, :, :], h=0, from_states=state
)
transition_sigma_minus = calculate.transition_electric_moments(
    mol, eigenstates[:, :, :], h=-1, from_states=state
)

idx=-1
minimum_dipole_moment = mol.d0 * 0.001
state_energy = eigenenergies[idx, state] / MHz

fig,ax = plt.subplots(figsize=(6,3))

print("At B = %.3f G" % (B / GAUSS))
print("All transitions from ground state with dipole moment > %.3f d_mol" % (minimum_dipole_moment / mol.d0))
print("State\tΔE (MHz)\t<mu>\tN\tMF\td (d_mol)")
for i in range(eigenstates.shape[1]):
    #only print transitions with some strength
    if (
        transition_sigma_plus[idx, 0, i] < minimum_dipole_moment
        and transition_pi[idx, 0, i] < minimum_dipole_moment
        and transition_sigma_minus[idx, 0, i] < minimum_dipole_moment
    ):
        continue
    
    if eigenlabels[i][0] < N_INITIAL: continue

    print(
        "%d\t%.6f\t%.3f\t%s\t%.1f\t(%.3f, %.3f, %.3f)"
        % (
            i,
            eigenenergies[idx, i] / MHz-state_energy,
            magnetic_moments[idx, i] / muN,
            eigenlabels[i][0],
            eigenlabels[i][1],
            transition_sigma_plus[idx, 0, i] / mol.d0,
            transition_pi[idx, 0, i] / mol.d0,
            transition_sigma_minus[idx, 0, i] / mol.d0,
        )
    )
    if transition_sigma_minus[idx, 0, i] > minimum_dipole_moment:
        x_pos = eigenenergies[idx, i] / MHz-state_energy
        y_pos = transition_sigma_minus[idx, 0, i] / mol.d0
        ax.scatter(x_pos, y_pos, color='green', alpha=0.5)
        ax.plot([x_pos, x_pos], [0, y_pos], color='green', alpha=0.5, linewidth=1)
    if transition_pi[idx, 0, i] > minimum_dipole_moment:
        x_pos = eigenenergies[idx, i] / MHz-state_energy
        y_pos = transition_pi[idx, 0, i] / mol.d0
        ax.scatter(x_pos, y_pos, color='blue', alpha=0.5)
        ax.plot([x_pos, x_pos], [0, y_pos], color='blue', alpha=0.5, linewidth=1)
    if transition_sigma_plus[idx, 0, i] > minimum_dipole_moment:
        x_pos = eigenenergies[idx, i] / MHz-state_energy
        y_pos = transition_sigma_plus[idx, 0, i] / mol.d0
        ax.scatter(x_pos, y_pos, color='red', alpha=0.5)
        ax.plot([x_pos, x_pos], [0, y_pos], color='red', alpha=0.5, linewidth=1)
plt.ylim(bottom=0)
ax.set_xlabel('Energy / h (MHz)')
ax.set_ylabel('Transition Dipole Moment / $d_{mol}$')
plt.show()
# %%
