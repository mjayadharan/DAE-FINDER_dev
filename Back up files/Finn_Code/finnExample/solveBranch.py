import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def branchRHS(y, t, k_rates):
    # Unpack states, params
    A, E1, AE1, B, E2, BE2, C, E3, BE3, D = y
    k_E1, kr_E1, kcat_E1, k_E2, kr_E2, kcat_E2, k_E3, kr_E3, kcat_E3 = [k_rates[x] for x in
                                                                        ['k_E1', 'kr_E1', 'kcat_E1', 'k_E2', 'kr_E2',
                                                                         'kcat_E2', 'k_E3', 'kr_E3', 'kcat_E3']]

    dydt = [kr_E1 * AE1 - k_E1 * E1 * A,  # A
            (kr_E1 + kcat_E1) * AE1 - k_E1 * E1 * A,  # E1
            k_E1 * E1 * A - (kr_E1 + kcat_E1) * AE1,  # AE1
            kcat_E1 * AE1 + kr_E2 * BE2 - k_E2 * E2 * B - k_E3 * B * E3 + kr_E3 * BE3,  # B
            (kr_E2 + kcat_E2) * BE2 - k_E2 * B * E2,  # E2
            k_E2 * E2 * B - (kr_E2 + kcat_E2) * BE2,  # BE2
            kcat_E2 * BE2,  # C
            (kr_E3 + kcat_E3) * BE3 - k_E3 * B * E3,  # E3
            k_E3 * E3 * B - (kr_E3 + kcat_E3) * BE3,  # BE3
            kcat_E3 * BE3]  # D
    return dydt


def branchMMRHS(y, t, k_rates, IC):
    # Unpack states, params
    A, B, C, D = y
    k_E1, kr_E1, kcat_E1, k_E2, kr_E2, kcat_E2, k_E3, kr_E3, kcat_E3 = [k_rates[x] for x in
                                                                        ['k_E1', 'kr_E1', 'kcat_E1', 'k_E2', 'kr_E2',
                                                                         'kcat_E2', 'k_E3', 'kr_E3', 'kcat_E3']]
    E1_0 = IC["E1"]
    E2_0 = IC["E2"]
    E3_0 = IC["E3"]

    dydt = [-(k_E1 * kcat_E1 * E1_0 * A) / (kr_E1 + kcat_E1 + k_E1 * A),  # A
            (k_E1 * kcat_E1 * E1_0 * A) / (kr_E1 + kcat_E1 + k_E1 * A) - (k_E2 * kcat_E2 * E2_0 * B) / (
                        kr_E2 + kcat_E2 + k_E2 * B)
            - (k_E3 * kcat_E3 * E3_0 * B) / (kr_E3 + kcat_E3 + k_E3 * B),  # B
            (k_E2 * kcat_E2 * E2_0 * B) / (kr_E2 + kcat_E2 + k_E2 * B),  # C
            (k_E3 * kcat_E3 * E3_0 * B) / (kr_E3 + kcat_E3 + k_E3 * B)]  # D
    return dydt
def solveBranch(init_cond, k_rates, solvedT, tsID, print_to_scr = False, print_to_file=False):
    if print_to_scr:
     print("Solving for Initial Conditions: {} \n and k_rates: {}".format(init_cond, k_rates))
    y0 = [init_cond["A"], init_cond["E1"], init_cond["AE1"], init_cond["B"], init_cond["E2"], init_cond["BE2"],
          init_cond["C"], init_cond['E3'], init_cond['BE3'], init_cond['D']]

    sol = odeint(lambda y, t: branchRHS(y, t, k_rates), y0, solvedT)

    final_sol = np.column_stack((sol[:, 0], sol[:, 1], sol[:,2], sol[:,3], sol[:,4], sol[:,5], sol[:,6], sol[:,7], sol[:,8],
                                 sol[:,9]))

    # paramID = "".join(str(k_rates.values).strip("()").split())
    # print(paramID)
    if print_to_file:
        np.savetxt('data/MM_Data_' + 'k_' + str(k_rates.values) + '__' + str(init_cond.values) + '_' + tsID + '.txt',
                   final_sol)
    return final_sol

def solveBranchMM(init_cond, k_rates, solvedT, tsID, print_to_scr = False, print_to_file=False):
    if print_to_scr:
     print("Solving for Initial Conditions: {} \n and k_rates: {}".format(init_cond, k_rates))
    y0 = [init_cond["A"], init_cond["B"], init_cond["C"], init_cond['D']]
    E_10 = init_cond["E1"]
    E_20 = init_cond["E2"]
    E_30 = init_cond['E3']
    k_E1, kr_E1, kcat_E1, k_E2, kr_E2, kcat_E2, k_E3, kr_E3, kcat_E3 = (k_rates['k_E1'], k_rates['kr_E1'], k_rates['kcat_E1'], k_rates['k_E2'],
                                                  k_rates['kr_E2'], k_rates['kcat_E2'], k_rates['k_E3'], k_rates['kr_E3'],
                                                                        k_rates['kcat_E3'])
    sol = odeint(lambda y, t: branchMMRHS(y, t, k_rates, init_cond), y0, solvedT)

    AE1_sol = (k_E1 * E_10 * sol[:, 0] / (kr_E1 + kcat_E1 + k_E1*sol[:,0]))
    E1_sol = E_10 - AE1_sol
    BE2_sol = (k_E2*sol[:,1]*E_20)/(kr_E2 + kcat_E2 + k_E2 * sol[:,1])
    E2_sol = E_20 - BE2_sol
    BE3_sol = (sol[:,1]*k_E3*E_30)/(kr_E3 + kcat_E3 + sol[:,1]*k_E3)
    E3_sol = E_30 - BE3_sol


    final_sol = np.column_stack((sol[:, 0], E1_sol, AE1_sol, sol[:, 1], E2_sol, BE2_sol, sol[:, 2], E3_sol, BE3_sol, sol[:,3]))

    # paramID = "".join(str(k_rates.values).strip("()").split())
    # print(paramID)
    if print_to_file:
        np.savetxt('data/MM_Data_' + 'k_' + str(k_rates.values) + '__' + str(init_cond.values) + '_' + tsID + '.txt',
                   final_sol)
    return final_sol


def plotToyEnz(solT, sol, title = ""):
    plt.plot(solT, sol[:, 0], '-b', label='A(t)', ms=3)
    plt.plot(solT, sol[:, 1], '-g', label='E1(t)', ms=3)
    plt.plot(solT, sol[:, 2], '-r', label='AE1(t)', ms=3)
    plt.plot(solT, sol[:, 3], '-k', label='B(t)', ms=3)
    plt.plot(solT, sol[:, 4], '-c', label='E2(t)', ms=3)
    plt.plot(solT, sol[:, 5], '-y', label='BE2(t)', ms=3)
    plt.plot(solT, sol[:, 6], '-m', label='C(t)', ms=3)
    plt.plot(solT, sol[:, 7], '--b', label='E3(t)', ms=3)
    plt.plot(solT, sol[:, 8], '--g', label='BE3(t)', ms=3)
    plt.plot(solT, sol[:, 9], '--r', label='D(t)', ms=3)
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.title(title)
    plt.show()
    return