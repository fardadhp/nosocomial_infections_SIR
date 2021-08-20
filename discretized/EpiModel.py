import numpy as np

def EpiEquations(y, t, numberUnits, omega_UC, omega_DC, omega_I, alpha_b, rho, pi, \
                 epsilon, kappa, delta, sigma_h, sigmaHat_h, rho_C, sigma_y, sigma_l, \
                 sigma_w, sigmaHat_y, sigmaHat_l, sigmaHat_w, beta_hp, beta_ph, \
                 E50, eta_p, eta_h, zeta, xi, admC, admI, admX_S, tranC, tranI, tranX_S, \
                 unitParamsValues, psi_in, psi_out, n_in, n_out, internalTransferRate, \
                 deviceTransferRate):
    (S, X, UC, DC, I, M, N0, N1, D0, D1, Y, L, W) = [y[i:i + numberUnits] for i in range(0, len(y), numberUnits)]
    P = S + X + UC + DC + I + M
    transmissionParams = CalculateTransmissionParams(unitParamsValues, beta_hp, beta_ph, eta_p, eta_h, Y, L, W, E50)
    alpha_np, alpha_dp, alpha_ep, alpha_pn, alpha_pd, alpha_py, alpha_pl, alpha_pw = transmissionParams[:8]
    alpha_ny, alpha_nl, alpha_nw, alpha_dy, alpha_dl, alpha_dw = transmissionParams[8:14]
    alpha_en, alpha_ed = transmissionParams[14:]
    statusAtAdmission = [(1-(admC+admI))/(admX_S+1), (1-(admC+admI))*admX_S/(admX_S+1), admC, admI]
    if abs(sum(statusAtAdmission)-1) > 0.01:
        raise ValueError('statusAtAdmission not at equilibrium!')
    statusAtTransfer = [(1-(tranC+tranI))/(tranX_S+1), (1-(tranC+tranI))*tranX_S/(tranX_S+1), tranC, tranI]
    if abs(sum(statusAtTransfer)-1) > 0.01:
        raise ValueError('statusAtTransfer not at equilibrium!')
    f_S, f_X, f_C, f_I = np.vstack(statusAtAdmission) * psi_in
    f_Y, f_L, f_W = [np.zeros(numberUnits) for i in range(3)]
    extTransIn_S, extTransIn_X, extTransIn_C, extTransIn_I = np.vstack(statusAtTransfer) * n_in
    m_l = deviceTransferRate
    
    S_dot = f_S + extTransIn_S + (S * internalTransferRate.transpose()/P).sum(1) + omega_UC * UC + omega_DC * DC - alpha_np * S * N1 / (N1+N0) - alpha_dp * S * D1 / (D1+D0) - (sum(internalTransferRate.transpose())/P + psi_out / P + n_out / P + alpha_b + alpha_ep) * S
    X_dot = f_X + extTransIn_X + (X * internalTransferRate.transpose()/P).sum(1) + (1- rho) * omega_I * I - alpha_np * pi * X * N1 / (N1+N0) - alpha_dp * pi * X * D1 / (D1+D0) - (sum(internalTransferRate.transpose())/P + psi_out / P + n_out / P + pi * (alpha_b + alpha_ep)) * X
    UC_dot = (1-zeta) * (f_C + extTransIn_C) + (UC * internalTransferRate.transpose()/P).sum(1) + alpha_np * (1-epsilon) * (S + pi * X) * N1 / (N1+N0) + alpha_dp * (1-epsilon) * (S + pi * X) * D1 / (D1+D0) + \
    		alpha_ep * (1-epsilon) * (S + pi * X) + alpha_b * (1-epsilon) * (S + pi * X) + rho * omega_I * I - (sum(internalTransferRate.transpose())/P + n_out / P + psi_out / P + kappa + omega_UC + xi) * UC
    DC_dot = zeta * (f_C + extTransIn_C) + (DC * internalTransferRate.transpose()/P).sum(1) + xi * UC - (sum(internalTransferRate.transpose())/P + n_out / P + psi_out / P + kappa + omega_DC) * DC
    I_dot = f_I + extTransIn_I + (I * internalTransferRate.transpose()/P).sum(1) + alpha_np * epsilon * (S + pi * X) * N1 / (N1+N0) + alpha_dp * epsilon * (S + pi * X) * D1 / (D1+D0) + \
    		alpha_ep * epsilon * (S + pi * X) + alpha_b * epsilon * (S + pi * X) + kappa * (UC + DC) - (sum(internalTransferRate.transpose())/P + n_out / P + psi_out / P + omega_I + delta) * I
    M_dot = delta * I
    N0_dot = (sigma_h + sigmaHat_h) * N1 - alpha_pn * N0 * (I + rho_C * (UC+DC)) / P - alpha_en * N0
    N1_dot = -1 * N0_dot
    D0_dot = (sigma_h + sigmaHat_h) * D1 - sum(alpha_pd * D0 * (I + rho_C * (UC+DC)) / P + alpha_ed * D0)
    D1_dot = -1 * D0_dot
    Y_dot = f_Y + alpha_py * (I + rho_C * (UC+DC)) + alpha_ny * N1 + alpha_dy * D1 - (sigma_y + sigmaHat_y) * Y
    L_dot = f_L + (L * m_l.transpose()).sum(1) + alpha_pl * (I + rho_C * (UC+DC)) + alpha_nl * N1 + alpha_dl * D1 - (sum(m_l.transpose()) + sigma_l + sigmaHat_l) * L
    W_dot = f_W + alpha_pw * (I + rho_C * (UC+DC)) + alpha_nw * N1 + alpha_dw * D1 - (sigma_w + sigmaHat_w) * W
    
    dy = np.hstack([S_dot, X_dot, UC_dot, DC_dot, I_dot, M_dot, N0_dot, N1_dot, D0_dot, D1_dot, \
    	  Y_dot, L_dot, W_dot])
    return dy

def CalculateTransmissionParams(unitParamsValues, beta_hp, beta_ph, eta_p, eta_h, Y, L, W, E50):
    lambda_n, lambda_d, lambda_py, lambda_pl, lambda_pw, lambda_ny, lambda_nl, \
        lambda_nw, lambda_dy, lambda_dl, lambda_dw, r_n, r_d = unitParamsValues
    alpha_np = lambda_n * beta_hp
    alpha_dp = lambda_d * beta_hp
    alpha_yp = probabilityContamination(doseResponse(Y, E50), lambda_py)
    alpha_lp = probabilityContamination(doseResponse(L, E50), lambda_pl)
    alpha_wp = probabilityContamination(doseResponse(W, E50), lambda_pw)
    alpha_ep = alpha_yp + alpha_lp + alpha_wp
    alpha_pn = lambda_n / r_n * beta_ph
    alpha_pd = lambda_d / r_d * beta_ph    
    alpha_py = lambda_py * eta_p
    alpha_pl = lambda_pl * eta_p
    alpha_pw = lambda_pw * eta_p
    alpha_ny = lambda_ny * eta_h
    alpha_nl = lambda_nl * eta_h
    alpha_nw = lambda_nw * eta_h
    alpha_dy = lambda_dy * eta_h
    alpha_dl = lambda_dl * eta_h
    alpha_dw = lambda_dw * eta_h
    alpha_en = lambda_ny * Y / (Y + E50) + lambda_nl * L / (L + E50) + lambda_nw * W / (W + E50)
    alpha_ed = lambda_dy * Y / (Y + E50) + lambda_dl * L / (L + E50) + lambda_dw * W / (W + E50)
    
    return [alpha_np, alpha_dp, alpha_ep, alpha_pn, alpha_pd, alpha_py, alpha_pl, alpha_pw, \
    	    alpha_ny, alpha_nl, alpha_nw, alpha_dy, alpha_dl, alpha_dw, alpha_en, alpha_ed]

def doseResponse(E, E50):
    return E / (E + E50)

def probabilityContamination(p, n):
    return 1 - (1 - p)**n