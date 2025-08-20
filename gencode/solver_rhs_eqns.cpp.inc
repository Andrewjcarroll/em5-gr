// Dendro: {{{ 
// Dendro: original ops:  66
// Dendro: printing temp variables
double DENDRO_0 = 4.0*PI;
double DENDRO_1 = DENDRO_0*rho_e[pp];

// Dendro: printing variables
//--
A_rhs0[pp] = -E0[pp] - grad_0_psi[pp];
//--
A_rhs1[pp] = -E1[pp] - grad_1_psi[pp];
//--
A_rhs2[pp] = -E2[pp] - grad_2_psi[pp];
//--
E_rhs0[pp] = -DENDRO_0*J0[pp] + grad_0_Gamma[pp] - grad_0_phi[pp] - grad2_0_0_A0[pp] - grad2_1_1_A0[pp] - grad2_2_2_A0[pp];
//--
E_rhs1[pp] = -DENDRO_0*J1[pp] + grad_1_Gamma[pp] - grad_1_phi[pp] - grad2_0_0_A1[pp] - grad2_1_1_A1[pp] - grad2_2_2_A1[pp];
//--
E_rhs2[pp] = -DENDRO_0*J2[pp] + grad_2_Gamma[pp] - grad_2_phi[pp] - grad2_0_0_A2[pp] - grad2_1_1_A2[pp] - grad2_2_2_A2[pp];
//--
Gamma_rhs[pp] = -DENDRO_1 - grad2_0_0_psi[pp] - grad2_1_1_psi[pp] - grad2_2_2_psi[pp];
//--
psi_rhs[pp] = -Gamma[pp];
//--
phi_rhs[pp] = -DENDRO_1 - 0.10000000000000001*phi[pp] - grad_0_E0[pp] - grad_1_E1[pp] - grad_2_E2[pp];
// Dendro: reduced ops:  61
// Dendro: }}} 
