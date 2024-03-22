""" Mass-action kinetics for branched reaction network reducible by MM assumption:

    A + E1 <-> AE1 -> B + E1
    B + E2 <-> BE2 -> C + E2
    B + E3 <-> BE3 -> D + E3

    Substrate A is catalyzed by E1 resulting in intermediate B. B can be catalyzed
    by either E2, resulting in product C, or E3 resulting in product D.
    All reactions are irreversible, though un/binding complex is reversible. 

            A
            | E1
            B
       E2 /   \ E3
         C     D
"""

def branchRHS(y, t, k_rates):
  # Unpack states, params
  A, E1, AE1, B, E2, BE2, C, E3, BE3, D = y
  k_E1, kr_E1, kcat_E1, k_E2, kr_E2, kcat_E2, k_E3, kr_E3, kcat_E3 = [k_rates[x] for x in
                      ['k_E1', 'kr_E1', 'kcat_E1', 'k_E2', 'kr_E2', 'kcat_E2', 'k_E3', 'kr_E3', 'kcat_E3']]
  
  dydt = [kr_E1*AE1 - k_E1*E1*A, # A
          (kr_E1 + kcat_E1)*AE1 - k_E1*E1*A, # E1
          k_E1*E1*A - (kr_E1 + kcat_E1)*AE1, # AE1
          kcat_E1*AE1 + kr_E2*BE2 - k_E2*E2*B - k_E3*B*E3 + kr_E3*BE3, # B
          (kr_E2 + kcat_E2)*BE2 - k_E2*B*E2, # E2
          k_E2*E2*B - (kr_E2 + kcat_E2)*BE2, # BE2
          kcat_E2*BE2, # C
          (kr_E3 + kcat_E3)*BE3 - k_E3*B*E3, # E3
          k_E3*E3*B - (kr_E3 + kcat_E3)*BE3, # BE3
          kcat_E3*BE3] # D
  return dydt

def branchMMRHS(y, t, k_rates, IC):
  # Unpack states, params
  A, B, C, D = y
  k_E1, kr_E1, kcat_E1, k_E2, kr_E2, kcat_E2, k_E3, kr_E3, kcat_E3 = [k_rates[x] for x in
                      ['k_E1', 'kr_E1', 'kcat_E1', 'k_E2', 'kr_E2', 'kcat_E2', 'k_E3', 'kr_E3', 'kcat_E3']]
  E1_0 = IC["E1"]
  E2_0 = IC["E2"]
  E3_0 = IC["E3"]
  
  dydt = [-(k_E1*kcat_E1*E1_0*A)/(kr_E1+kcat_E1+k_E1*A), # A
          (k_E1*kcat_E1*E1_0*A)/(kr_E1+kcat_E1+k_E1*A) - (k_E2*kcat_E2*E2_0*B)/(kr_E2+kcat_E2+k_E2*B)
              - (k_E3*kcat_E3*E3_0*B)/(kr_E3+kcat_E3+k_E3*B), # B
          (k_E2*kcat_E2*E2_0*B)/(kr_E2+kcat_E2+k_E2*B), # C
          (k_E3*kcat_E3*E3_0*B)/(kr_E3+kcat_E3+k_E3*B)] # D
  return dydt
