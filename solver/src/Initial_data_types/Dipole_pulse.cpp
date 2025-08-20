 const double amp1 = solver::EM5_ID_AMP1;
        const double lambda1 = solver::EM5_ID_LAMBDA1;

        double psi, A0,A1,A2, Gamma, E0,E1,E2; 
        double rho_e, J0,J1,J2;
        double r_sq,tmp_Ephiup ; 

        r_sq = x*x + y*y + z*z ; 
        tmp_Ephiup = - 8.0*amp1*lambda1*lambda1*exp(-lambda1*r_sq) ; 
        E0 = - y * tmp_Ephiup ; 
        E1 =   x * tmp_Ephiup ; 
        E2 = 0.0 ; 
 
        Gamma = 0.0 ; 

        A0 = 0.0 ;  
        A1 = 0.0 ;  
        A2 = 0.0 ;  
        
        psi = 0.0 ;  

        phi = 0.0;
        
        J0 = 0.0 ;  
        J1 = 0.0 ;  
        J2 = 0.0 ;  
        
        rho_e =  0.0 ;  
        
        var[VAR::U_PSI] = psi;
        var[VAR::U_E0] = E0;
        var[VAR::U_E1] = E1;
        var[VAR::U_E2] = E2;
        var[VAR::U_A0] = A0;
        var[VAR::U_A1] = A1;
        var[VAR::U_A2] = A2;
        var[VAR::U_GAMMA] = Gamma;
