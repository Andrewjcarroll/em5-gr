/**
 * This file is a basic include
 *
 *
 */

/*[[[cog
import cog
import sys
import importlib.util
import dendrosym

# the following lines will import any module directly from
spec = importlib.util.spec_from_file_location("dendroconf", CONFIG_FILE_PATH)
dendroconf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dendroconf
spec.loader.exec_module(dendroconf)

cog.outl('//// dendro_derivs::ko GRADIENT CALCULATION')
cog.outl(dendroconf.dendroConfigs.generate_dendro_derivs::dendro_derivs::ko_derivs("evolution"))

]]]*/
//// dendro_derivs::ko GRADIENT CALCULATION

dendro_derivs::ko_deriv_x(grad_0_Gamma, Gamma, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_Gamma, Gamma, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_Gamma, Gamma, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_psi, psi, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_psi, psi, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_psi, psi, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_E0, E0, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_E0, E0, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_E0, E0, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_E1, E1, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_E1, E1, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_E1, E1, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_E2, E2, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_E2, E2, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_E2, E2, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_A0, A0, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_A0, A0, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_A0, A0, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_A1, A1, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_A1, A1, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_A1, A1, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_A2, A2, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_A2, A2, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_A2, A2, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_phi, phi, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_phi, phi, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_phi, phi, hz, sz, bflag);

//[[[end]]]
