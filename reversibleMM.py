""" Mass-action kinetics for reversible reaction network reducible by MM assumption:

    A + E <-> C <-> B + E

    Substrate A is catalyzed by E resulting in intermediate B. B is catalyzed
    by E, resulting in A. This is a reversible reaction. Un/binding complex
    is also reversible. 

            A
            | E1
            B
"""

def reversibleRHS(y, t, k_rates):
    # Unpack states, params
    A, E, C, B = y
    k, kr, kcatf, kcatr = [k_rates[x] for x in
                        ['k', 'kr', 'kcatf', 'kcatr']]

    dydt = [kr*C - k*E*A, # A
            (kr + kcatf)*C - k*E*A - kcatr*B*E, # E
            k*E*A - (kr + kcatf)*C + kcatr*B*E, # C
            kcatf*C - kcatr*B*E] # B
    return dydt

def reversibleMMRHS(y, t, k_rates, IC):
    # Unpack states, params
    A, B = y
    E_0 = IC["E"]
    k, kr, kcatf, kcatr = [k_rates[x] for x in
                        ['k', 'kr', 'kcatf', 'kcatr']]

    dydt = [-E_0*(k*kcatf*A - kr*kcatr*B)/(kr + kcatf + k*A + kcatr*B), # A
            E_0*(k*kcatf*A - kr*kcatr*B)/(kr + kcatf + k*A + kcatr*B)] # B
    return dydt