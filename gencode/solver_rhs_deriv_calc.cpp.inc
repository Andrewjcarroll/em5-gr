// 1st derivs for psi 
  dendro_derivs::deriv_x(grad_0_psi, psi, hx, sz, bflag); 
  dendro_derivs::deriv_y(grad_1_psi, psi, hy, sz, bflag); 
  dendro_derivs::deriv_z(grad_2_psi, psi, hz, sz, bflag); 

  // 2nd derivs for psi 
  dendro_derivs::deriv_xx(grad2_0_0_psi, psi, hx, sz, bflag); 
  dendro_derivs::deriv_yy(grad2_1_1_psi, psi, hy, sz, bflag); 
  dendro_derivs::deriv_zz(grad2_2_2_psi, psi, hz, sz, bflag); 

  // 1st derivs for Gamma  
  dendro_derivs::deriv_x(grad_0_Gamma, Gamma, hx, sz, bflag); 
  dendro_derivs::deriv_y(grad_1_Gamma, Gamma, hy, sz, bflag); 
  dendro_derivs::deriv_z(grad_2_Gamma, Gamma, hz, sz, bflag); 

  // 1st derivatives for diagonal components of E
dendro_derivs::deriv_x(grad_0_E0, E0, hx, sz, bflag); // ∂x Ex
dendro_derivs::deriv_y(grad_1_E1, E1, hy, sz, bflag); // ∂y Ey
dendro_derivs::deriv_z(grad_2_E2, E2, hz, sz, bflag); // ∂z Ez


  // 2nd derivs for A0. 
  dendro_derivs::deriv_xx(grad2_0_0_A0, A0, hx, sz, bflag);
  dendro_derivs::deriv_yy(grad2_1_1_A0, A0, hy, sz, bflag);
  dendro_derivs::deriv_zz(grad2_2_2_A0, A0, hz, sz, bflag);

  // 2nd derivs for A1
  dendro_derivs::deriv_xx(grad2_0_0_A1, A1, hx, sz, bflag);
  dendro_derivs::deriv_yy(grad2_1_1_A1, A1, hy, sz, bflag);
  dendro_derivs::deriv_zz(grad2_2_2_A1, A1, hz, sz, bflag);

  // 2nd derivs for A2
  dendro_derivs::deriv_xx(grad2_0_0_A2, A2, hx, sz, bflag); 
  dendro_derivs::deriv_yy(grad2_1_1_A2, A2, hy, sz, bflag);
  dendro_derivs::deriv_zz(grad2_2_2_A2, A2, hz, sz, bflag);
  
  //derivatives for Phi the constraint damping variable
  dendro_derivs::deriv_x(grad_0_phi, phi, hx, sz, bflag);
  dendro_derivs::deriv_y(grad_1_phi, phi, hy, sz, bflag);
  dendro_derivs::deriv_z(grad_2_phi, phi, hz, sz, bflag);