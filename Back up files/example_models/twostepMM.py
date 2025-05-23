""" Mass-action kinetics for two-step reaction network reducible by MM assumption:

    A + E1 <-> AE1 -> B + E1
    B + E2 <-> BE2 -> C + E2

    Substrate A is catalyzed by E1 resulting in intermediate B. B is catalyzed
    by E2, resulting in product C. All reactions are irreversible,
    though un/binding complex is reversible. 

            A
            | E1
            B
         E2 |
            C   
"""

def twostepRHS(y, t, k_rates):
    # Unpack states, params
    A, E1, AE1, B, E2, BE2, C = y
    k_E1, kr_E1, kcat_E1, k_E2, kr_E2, kcat_E2 = [k_rates[x] for x in
                        ['k_E1', 'kr_E1', 'kcat_E1', 'k_E2', 'kr_E2', 'kcat_E2']]

    dydt = [kr_E1*AE1 - k_E1*E1*A, # A
            (kr_E1 + kcat_E1)*AE1 - k_E1*E1*A, # E1
            k_E1*E1*A - (kr_E1 + kcat_E1)*AE1, # AE1
            kcat_E1*AE1 + kr_E2*BE2 - k_E2*E2*B, # B
            (kr_E2 + kcat_E2)*BE2 - k_E2*B*E2, # E2
            k_E2*E2*B - (kr_E2 + kcat_E2)*BE2, # BE2
            kcat_E2*BE2] # C
    return dydt

def twostepMMRHS(y, t, k_rates, IC):
    # Unpack states, params
    A, B, C = y
    E1_0 = IC["E1"]
    E2_0 = IC["E2"]
    k_E1, kr_E1, kcat_E1, k_E2, kr_E2, kcat_E2 = [k_rates[x] for x in
                      ['k_E1', 'kr_E1', 'kcat_E1', 'k_E2', 'kr_E2', 'kcat_E2']]
  
    dydt = [-(k_E1*kcat_E1*E1_0*A)/(kr_E1+kcat_E1+k_E1*A), # A
            (k_E1*kcat_E1*E1_0*A)/(kr_E1+kcat_E1+k_E1*A) - (k_E2*kcat_E2*E2_0*B)/(kr_E2+kcat_E2+k_E2*B), # B
            (k_E2*kcat_E2*E2_0*B)/(kr_E2+kcat_E2+k_E2*B)] # C
    return dydt