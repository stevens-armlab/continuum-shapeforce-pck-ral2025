# pck_shape_estimation_testplot_6d.py
# Adds a general 6D external-wrench solver (Option A) with rotated priors/weights.
# Keeps existing 2D solver as an option (--solver plane2) for A/B comparisons.

import numpy as np
import math, json, argparse, csv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

VERSION = "pck_shape_estimation_testplot_6D"

# --------------------- Utilities ---------------------

def Rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def blkdiag(A, B):
    Z1 = np.zeros((A.shape[0], B.shape[1]))
    Z2 = np.zeros((B.shape[0], A.shape[1]))
    return np.block([[A, Z1],
                     [Z2, B]])

def delta_from_tensions(q):
    q = np.asarray(q, float).ravel()
    return -math.atan2(q[1] - q[3], q[0] - q[2])

def tendon_offsets_Rp(Rp):
    phi = np.array([0.0, 0.5*np.pi, np.pi, 1.5*np.pi])
    r = np.stack([Rp*np.cos(phi), Rp*np.sin(phi), np.zeros_like(phi)], axis=1)
    return r

def tip_moment_from_tensions(q, Rp):
    r = tendon_offsets_Rp(Rp)
    ez = np.array([0,0,1.0])
    M = np.zeros(3)
    for i in range(4):
        M += np.cross(r[i], q[i]*ez)
    return M  # (Mx, My, 0)

# --------------------- Planar elastica (GT) ---------------------

def integrate_planar_elastica(L, EI, Fx, Fz=0.0, kappa0=0.0, N=1500):
    s = np.linspace(0.0, L, N)
    ds = s[1] - s[0]
    theta = np.zeros(N); kappa = np.zeros(N); x = np.zeros(N); z = np.zeros(N)
    theta[0] = 0.0; kappa[0] = kappa0; x[0] = 0.0; z[0] = 0.0

    def deriv(y):
        th, kap, xx, zz = y
        tx = math.sin(th); tz = math.cos(th)
        dtheta = kap
        dkappa = (tx * Fz - tz * Fx) / EI
        dx = tx; dz = tz
        return np.array([dtheta, dkappa, dx, dz], dtype=float)

    y = np.array([theta[0], kappa[0], x[0], z[0]], dtype=float)
    for i in range(N-1):
        k1 = deriv(y); k2 = deriv(y + 0.5*ds*k1)
        k3 = deriv(y + 0.5*ds*k2); k4 = deriv(y + ds*k3)
        y = y + (ds/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        theta[i+1], kappa[i+1], x[i+1], z[i+1] = y

    return {"s": s, "theta": theta, "kappa": kappa, "x": x, "z": z}

def shoot_kappa0_for_target_tip_kappa(L, EI, Fx, Fz, kappaL_target, N=1500, max_iter=60, tol=1e-10):
    def res(k0):
        sol = integrate_planar_elastica(L, EI, Fx, Fz, kappa0=k0, N=N)
        return sol["kappa"][-1] - kappaL_target, sol

    guess = float(kappaL_target)
    k0_a = 0.7*guess if guess != 0 else 0.0
    k0_b = 1.3*guess if guess != 0 else 1e-6

    fa, sola = res(k0_a); fb, solb = res(k0_b)
    if abs(fa) < tol: return k0_a, sola
    if abs(fb) < tol: return k0_b, solb

    k_prev, f_prev, sol_prev = k0_a, fa, sola
    k_curr, f_curr, sol_curr = k0_b, fb, solb
    for _ in range(max_iter):
        denom = (f_curr - f_prev) if (f_curr - f_prev) != 0 else 1e-12
        k_next = k_curr - f_curr * (k_curr - k_prev) / denom
        f_next, sol_next = res(k_next)
        if abs(f_next) < tol: return k_next, sol_next
        k_prev, f_prev, sol_prev = k_curr, f_curr, sol_curr
        k_curr, f_curr, sol_curr = k_next, f_next, sol_next
    return k_curr, sol_curr

def gt_inplane_shape(L, E, r_backbone, Rp, tensions, Fx, Fz,
                     lock_delta=False, delta_user=-math.pi/4, N=1500):
    I  = math.pi * r_backbone**4 / 4.0
    EI = E * I
    M_tip = tip_moment_from_tensions(tensions, Rp)
    M_mag = float(np.linalg.norm(M_tip[:2]))
    delta = delta_user if lock_delta else delta_from_tensions(tensions)
    kappaL_target = M_mag / EI
    k0, sol = shoot_kappa0_for_target_tip_kappa(L, EI, Fx, Fz, kappaL_target, N=N)
    return sol["x"], sol["z"], delta, sol

# --------------------- PCK integration ---------------------

def kappa_poly(S, s, order):
    s = float(s)
    if order == 0:   return S[0]
    if order == 1:   return S[0] + S[1]*s
    if order == 2:   return S[0] + S[1]*s + S[2]*(s**2)
    raise ValueError("order must be 0,1,2")

def integrate_shape_pck(S, order, L, s_eval=None, rtol=1e-10, atol=1e-12, method="DOP853"):
    def f(s, y):
        x, z, th = y
        kap = kappa_poly(S, s, order)
        return [L*math.sin(th), L*math.cos(th), kap]
    if s_eval is None: s_eval = np.linspace(0.0, 1.0, 600)
    y0 = [0.0, 0.0, 0.0]
    sol = solve_ivp(f, (0.0, 1.0), y0, method=method, t_eval=s_eval,
                    rtol=rtol, atol=atol, dense_output=False, max_step=1e-2)
    if not sol.success: raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y[0], sol.y[1], sol.y[2]

def tip_from_S(S, order, L):
    s, x, z, th = integrate_shape_pck(S, order, L, s_eval=np.array([0.0, 1.0]))
    return x[-1], z[-1], th[-1]

def reconstruct_curve(S, order, L, Nsamp=300):
    s_eval = np.linspace(0.0, 1.0, Nsamp)
    s, x, z, th = integrate_shape_pck(S, order, L, s_eval=s_eval)
    return x, z, th, s_eval

def residual_tip(S, order, xz_obs, th_obs, L, wp, wR):
    x_tip, z_tip, th_end = tip_from_S(S, order, L)
    dp = np.array([x_tip, z_tip]) - np.asarray(xz_obs).reshape(2,)
    dth = th_end - th_obs
    return np.hstack([ wp * dp, wR * dth ])

def estimate_pck(order, S0, xz_obs, th_obs, L, wp, wR):
    rfun = lambda S: residual_tip(S, order, xz_obs, th_obs, L, wp, wR)
    res = least_squares(rfun, np.asarray(S0, float), method='lm',
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
    return res.x

# --------------------- Virtual-work Jacobians ---------------------

def theta_poly_pw(S, s):
    s = np.asarray(s)
    m0 = S[0]
    m1 = S[1] if len(S) > 1 else 0.0
    m2 = S[2] if len(S) > 2 else 0.0
    return m0*s + 0.5*m1*s**2 + (1.0/3.0)*m2*s**3

def grad_U(S, E, I, L):
    m0 = S[0]
    m1 = S[1] if len(S) > 1 else 0.0
    m2 = S[2] if len(S) > 2 else 0.0
    c0 = (m0       + 0.5*m1 + (1.0/3.0)*m2)
    c1 = (0.5*m0   + (1.0/3.0)*m1 + 0.25*m2)
    c2 = ((1.0/3.0)*m0 + 0.25*m1 + 0.2*m2)
    g  = (E*I /L) * np.array([c0, c1, c2, 0.0], dtype=float)
    return g

def build_JpS(S, L, delta, n_int=400):
    """
    Returns 3x4 matrix of tip position sensitivity wrt [m0,m1,m2,delta], expressed in base frame.
    """
    s = np.linspace(0.0, 1.0, n_int)
    th = theta_poly_pw(S, s)
    cos_th, sin_th = np.cos(th), np.sin(th)
    X = L * np.trapezoid(sin_th, s)

    dth_dm0 = s
    dth_dm1 = 0.5*s**2
    dth_dm2 = (1.0/3.0)*s**3

    col0_plane = L * np.array([
        np.trapezoid(cos_th * dth_dm0, s),
        0.0,
        np.trapezoid(-sin_th * dth_dm0, s)
    ])
    col1_plane = L * np.array([
        np.trapezoid(cos_th * dth_dm1, s),
        0.0,
        np.trapezoid(-sin_th * dth_dm1, s)
    ])
    col2_plane = L * np.array([
        np.trapezoid(cos_th * dth_dm2, s),
        0.0,
        np.trapezoid(-sin_th * dth_dm2, s)
    ])

    R = Rz(-delta)  # rotate plane->base
    col0_world = R @ col0_plane
    col1_world = R @ col1_plane
    col2_world = R @ col2_plane
    col_delta_world = R @ np.array([0.0, -X, 0.0])  # d p_tip / d delta

    return np.column_stack([col0_world, col1_world, col2_world, col_delta_world])  # 3x4

def build_JqS(S, Rp, delta):
    th_end = S[0] + 0.5*(S[1] if len(S) > 1 else 0.0) + (1.0/3.0)*(S[2] if len(S) > 2 else 0.0)
    phi = np.array([0.0, 0.5*np.pi, np.pi, 1.5*np.pi], dtype=float)
    sig = phi + delta
    c = np.cos(sig); s = np.sin(sig)
    J = np.zeros((4,4), dtype=float)
    J[:,0] =  Rp * c
    J[:,1] =  0.5 * Rp * c
    J[:,2] =  (1.0/3.0) * Rp * c
    J[:,3] =  -Rp * th_end * s
    return J

# --------------------- 2D wrench solver (existing) ---------------------

def solve_wrench_vw_plane2(S, E, I, L, Rp, tau_4, delta, lam=0.0):
    S = np.asarray(S, float).ravel()
    tau = np.asarray(tau_4, float).reshape(4,)
    JpS = build_JpS(S, L, delta)     # 3x4
    JqS = build_JqS(S, Rp, delta)    # 4x4
    gU  = grad_U(S, E, I, L).reshape(4,)
    Bx = np.array([math.cos(-delta), math.sin(-delta), 0.0], dtype=float)
    Bz = np.array([0.0, 0.0, 1.0], dtype=float)
    A = np.column_stack([JpS.T @ Bx, JpS.T @ Bz])  # 4x2
    rhs = gU - (JqS.T @ tau)                       # 4,
    if lam > 0:
        A_aug = np.vstack([A, math.sqrt(lam)*np.eye(2)])
        rhs_aug = np.hstack([rhs, np.zeros(2)])
        sol, *_ = np.linalg.lstsq(A_aug, rhs_aug, rcond=None)
    else:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return float(sol[0]), float(sol[1])

# --------------------- 6D wrench solver (Option A) ---------------------

def pinv_svd(A, rtol=1e-5):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.zeros_like(A.T)
    tol = rtol * (s[0] if s.size>0 else 1.0)
    s_inv = np.array([1/si if si>tol else 0.0 for si in s])
    return (Vt.T * s_inv) @ U.T, (U, s, Vt, tol)

def build_WB_F0B(delta, Fx1_prev=0.0, Fz1_prev=0.0,
                 eps_inplane=1e-6, lam_perp=1e6, lam_M=1e6):
    """
    Build base-frame prior and weights that encode:
      - No external torques (Mx=My=Mz=0)
      - Force lies in the bending plane of Frame{1} (Fy1 = 0), but Fx1,Fz1 free.
    """
    R = Rz(-delta)
    T = blkdiag(R, R)          # 6x6
    Tt = blkdiag(R.T, R.T)
    # W in bending frame {1}
    W1 = np.diag([eps_inplane, lam_perp, eps_inplane, lam_M, lam_M, lam_M])
    F0_1 = np.array([Fx1_prev, 0.0, Fz1_prev, 0.0, 0.0, 0.0], dtype=float)
    # rotate to base
    W_B = Tt @ W1 @ T
    F0_B = T @ F0_1
    return W_B, F0_B

def solve_wrench_vw_6d(S, E, I, L, Rp, tau_4, delta,
                       eps_inplane=1e-6, lam_perp=1e6, lam_M=1e6,
                       rtol_pinv=1e-5):
    """
    General 6D base-frame solve with rotated weights/prior (Option A).
    Moments are scaled by alpha=1/L for numerical balance.
    """
    S = np.asarray(S, float).ravel()
    tau = np.asarray(tau_4, float).reshape(4,)
    # Build equilibrium
    JpS = build_JpS(S, L, delta)           # 3x4 (position sensitivity)
    JqS = build_JqS(S, Rp, delta)          # 4x4
    gU  = grad_U(S, E, I, L).reshape(4,)   # 4,
    # Assemble 6x4 Jacobian for [Fx,Fy,Fz,Mx,My,Mz]:
    JxS6 = np.vstack([JpS, np.zeros((3,4))])    # rotational rows zero (we assume no applied torque)
    A = JxS6.T                                  # 4x6
    b = gU - (JqS.T @ tau)                      # 4,

    # Scaling for moments
    alpha = 1.0 / L
    S6 = np.diag([1,1,1, alpha, alpha, alpha])      # F_tilde = S6 @ F
    Sinv = np.diag([1,1,1, 1/alpha, 1/alpha, 1/alpha])

    # Rotate weights & prior to base, then map to scaled variables
    W_B, F0_B = build_WB_F0B(delta,
                             Fx1_prev=0.0, Fz1_prev=0.0,
                             eps_inplane=eps_inplane, lam_perp=lam_perp, lam_M=lam_M)
    # Transform W and F0 into scaled coordinates:
    # A * F = b  =>  (A * S6^{-1}) * F_tilde = b
    A_s = A @ np.linalg.inv(S6)
    F0_t = S6 @ F0_B
    W_t  = np.linalg.inv(S6).T @ W_B @ np.linalg.inv(S6)

    # Particular solution in scaled space
    Aplus, (U, s, Vt, tol) = pinv_svd(A_s, rtol=rtol_pinv)
    Fp_t = Aplus @ b

    # Nullspace projector via SVD
    r = np.sum(s > tol)
    V = Vt.T
    if r < V.shape[0]:
        N = V[:, r:]            # 6 x (6-r)
    else:
        N = np.zeros((6,0))     # full rank (rare here)

    # Weighted LS in the nullspace: eta* = (N^T W N)^+ N^T W (F0 - Fp)
    if N.shape[1] > 0:
        Omega = N.T @ W_t @ N
        Omega_plus, _ = pinv_svd(Omega, rtol=rtol_pinv)
        eta = Omega_plus @ (N.T @ W_t @ (F0_t - Fp_t))
        F_t = Fp_t + N @ eta
    else:
        F_t = Fp_t

    # Unscale back to physical units
    F_B = np.linalg.inv(S6) @ F_t
    return F_B  # [FxB, FyB, FzB, MxB, MyB, MzB]

# --------------------- Shape error helpers ---------------------

def ang_diff(a, b):
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return d

def interp_gt_on_unit_s(sol_gt, L, s_eval_unit):
    s_norm = sol_gt["s"] / L
    x_gt = np.interp(s_eval_unit, s_norm, sol_gt["x"])
    z_gt = np.interp(s_eval_unit, s_eval_unit*0 + s_norm, sol_gt["z"])  # ensure same grid
    th_gt = np.interp(s_eval_unit, s_norm, sol_gt["theta"])
    return x_gt, z_gt, th_gt

def shape_metrics(S, order, L, sol_gt, Ns=200):
    _, x_tip_hat, z_tip_hat, th_hat = integrate_shape_pck(S, order, L, s_eval=np.array([0.0, 1.0]))
    x_tip_hat, z_tip_hat, th_tip_hat = x_tip_hat[-1], z_tip_hat[-1], th_hat[-1]
    x_tip_gt,  z_tip_gt,  th_tip_gt  = sol_gt["x"][-1], sol_gt["z"][-1], sol_gt["theta"][-1]
    tip_pos_mm = 1e3 * math.hypot(x_tip_hat - x_tip_gt, z_tip_hat - z_tip_gt)
    tip_rot_deg = abs(ang_diff(th_tip_hat, th_tip_gt)) * 180.0 / math.pi
    s_eval = np.linspace(0.0, 1.0, Ns)
    _, xh, zh, thh = integrate_shape_pck(S, order, L, s_eval=s_eval)
    xg, zg, thg = interp_gt_on_unit_s(sol_gt, L, s_eval)
    shape_pos_mm = 1e3 * float(np.mean(np.hypot(xh - xg, zh - zg)))
    shape_rot_deg = float(np.mean(np.abs(ang_diff(thh, thg)))) * 180.0 / math.pi
    return tip_pos_mm, tip_rot_deg, shape_pos_mm, shape_rot_deg

# --------------------- Plotting helpers ---------------------

def _draw_arrow(ax, x0, z0, Fx, Fz, L, arrow_scale, color='purple'):
    nrm = math.hypot(Fx, Fz)
    if nrm == 0:
        return
    length = arrow_scale * L * nrm
    xt = x0 + length * Fx / nrm
    zt = z0 + length * Fz / nrm
    ax.annotate('', xy=(xt, zt), xytext=(x0, z0),
                arrowprops=dict(arrowstyle='-|>', lw=2.0, color=color),
                zorder=8, annotation_clip=False)
    ax.plot([x0, xt], [z0, zt], alpha=0)

def _plot_one_cell(ax, L, E, r, Rp, q, Fx, Fz,
                   NTRIALS, sig_x, sig_z, sig_th, wp, wR,
                   lock_delta, delta_user, arrow_scale,
                   style_pck_lw,
                   gt_markers, gt_construction,
                   fs_label, fs_tick,
                   # tension noise:
                   sig_tau, bias_tau, scale_tau, drift_tau,
                   # shape table:
                   shape_records, shape_N,
                   # bookkeeping:
                   collect_forces=False, force_records=None,
                   case_idx=1, r_idx=0, c_idx=0, title_prefix="Config.", fs_title=12.0,
                   # solver & weights:
                   solver="full6d",
                   eps_inplane=1e-6, lam_perp=1e6, lam_M=1e6,
                   rtol_pinv=1e-5):
    # ---- GT shape ----
    x_gt, z_gt, delta_gt, sol = gt_inplane_shape(L, E, r, Rp, q, Fx, Fz, lock_delta, delta_user)
    th_tip = sol["theta"][-1]
    x_tip_gt, z_tip_gt = x_gt[-1], z_gt[-1]

    # ---- PCK0/1/2 from (noisy) tip ----
    S0s = []; S1s = []; S2s = []
    for _ in range(NTRIALS):
        x_meas = x_tip_gt + sig_x*np.random.randn()
        z_meas = z_tip_gt + sig_z*np.random.randn()
        th_meas = th_tip   + sig_th*np.random.randn()
        S0s.append(estimate_pck(0, [0.0],            [x_meas, z_meas], th_meas, L, wp, wR))
        S1s.append(estimate_pck(1, [0.0, 0.0],       [x_meas, z_meas], th_meas, L, wp, wR))
        S2s.append(estimate_pck(2, [0.0, 0.0, 0.0],  [x_meas, z_meas], th_meas, L, wp, wR))
    S0m = np.mean(np.vstack(S0s), axis=0)
    S1m = np.mean(np.vstack(S1s), axis=0)
    S2m = np.mean(np.vstack(S2s), axis=0)

    # ---- curves (PCK solid r/g/b) ----
    x0, z0, th0, _ = reconstruct_curve(S0m, 0, L)
    x1, z1, th1, _ = reconstruct_curve(S1m, 1, L)
    x2, z2, th2, _ = reconstruct_curve(S2m, 2, L)
    ax.plot(x0, z0, 'r-', lw=style_pck_lw, label='PCK0')
    ax.plot(x1, z1, 'g-', lw=style_pck_lw, label='PCK1')
    ax.plot(x2, z2, 'b-', lw=style_pck_lw, label='PCK2')

    # ---- GT markers ----
    if gt_markers > 0:
        N = len(x_gt)
        idx = np.unique(np.linspace(0, N-1, gt_markers, dtype=int))
        xm = x_gt[idx]; zm = z_gt[idx]
        if gt_construction and len(idx) > 1:
            ax.plot(xm, zm, color='#666', ls='--', lw=1.0, alpha=0.9, label='GT (shape)')
        ax.plot(xm, zm, 'ko', ms=3.8, mfc='w', mew=0.9, label='GT markers')
    else:
        ax.plot(x_gt, z_gt, 'k-', lw=2.6, label='GT')

    # tip marker
    ax.plot(x_tip_gt, z_tip_gt, 'ko', ms=6, mfc='w', mew=1.4)
    # applied force arrow (in-plane Fx,Fz at the tip in world)
    _draw_arrow(ax, x_tip_gt, z_tip_gt, Fx, Fz, L, arrow_scale, color='purple')

    # ---- SHAPE METRICS (per-subplot CSV) ----
    if shape_records is not None:
        t0_pos, t0_rot, s0_pos, s0_rot = shape_metrics(S0m, 0, L, sol, Ns=shape_N)
        t1_pos, t1_rot, s1_pos, s1_rot = shape_metrics(S1m, 1, L, sol, Ns=shape_N)
        t2_pos, t2_rot, s2_pos, s2_rot = shape_metrics(S2m, 2, L, sol, Ns=shape_N)
        label = f"{title_prefix} {r_idx+1}-{c_idx+1}"
        shape_records.append({
            "case_idx": case_idx, "subplot": label,
            "Fx_gt": Fx, "Fz_gt": Fz,
            "P0_tip_pos_mm": t0_pos, "P0_tip_rot_deg": t0_rot,
            "P1_tip_pos_mm": t1_pos, "P1_tip_rot_deg": t1_rot,
            "P2_tip_pos_mm": t2_pos, "P2_tip_rot_deg": t2_rot,
            "P0_shape_pos_mm": s0_pos, "P0_shape_rot_deg": s0_rot,
            "P1_shape_pos_mm": s1_pos, "P1_shape_rot_deg": s1_rot,
            "P2_shape_pos_mm": s2_pos, "P2_shape_rot_deg": s2_rot,
        })

    # ---- FORCE ESTIMATION (CSV) ----
    if collect_forces and (force_records is not None):
        I = math.pi * r**4 / 4.0
        tau_true = np.asarray(q, float).reshape(4,)

        # Tension noise model
        if abs(bias_tau) > 1e-12: b = np.full(4, bias_tau, dtype=float)
        else:                     b = np.random.uniform(-0.05, 0.05, size=4)
        eps_scale = np.random.normal(0.0, scale_tau, size=4) if abs(scale_tau)>0 else 0.0
        eta = np.random.normal(0.0, sig_tau, size=4)
        tau_meas = (1.0 + eps_scale) * tau_true + b + eta
        tau_meas = np.clip(tau_meas, 0.0, None)

        delta_used = (delta_gt if lock_delta else delta_from_tensions(tau_meas))

        S0_ext = np.array([S0m[0],          0.0,       0.0,       delta_used], dtype=float)
        S1_ext = np.array([S1m[0],          S1m[1],    0.0,       delta_used], dtype=float)
        S2_ext = np.array([S2m[0],          S2m[1],    S2m[2],    delta_used], dtype=float)

        label = f"{title_prefix} {r_idx+1}-{c_idx+1}"

        if solver == "plane2":
            Fx0, Fz0 = solve_wrench_vw_plane2(S0_ext, E, I, L, Rp, tau_meas, delta_used)
            Fx1, Fz1 = solve_wrench_vw_plane2(S1_ext, E, I, L, Rp, tau_meas, delta_used)
            Fx2, Fz2 = solve_wrench_vw_plane2(S2_ext, E, I, L, Rp, tau_meas, delta_used)

            force_records.append({
                "case_idx": case_idx, "subplot": label,
                "Fx_gt": Fx, "Fz_gt": Fz,
                "Fx_pck0": Fx0, "Fz_pck0": Fz0,
                "Fx_pck1": Fx1, "Fz_pck1": Fz1,
                "Fx_pck2": Fx2, "Fz_pck2": Fz2,
                "Fx_err_pck0": Fx0 - Fx, "Fz_err_pck0": Fz0 - Fz,
                "Fx_err_pck1": Fx1 - Fx, "Fz_err_pck1": Fz1 - Fz,
                "Fx_err_pck2": Fx2 - Fx, "Fz_err_pck2": Fz2 - Fz,
            })
        else:
            # 6D solve (base frame)
            F0 = solve_wrench_vw_6d(S0_ext, E, I, L, Rp, tau_meas, delta_used,
                                    eps_inplane=eps_inplane, lam_perp=lam_perp, lam_M=lam_M,
                                    rtol_pinv=rtol_pinv)
            F1 = solve_wrench_vw_6d(S1_ext, E, I, L, Rp, tau_meas, delta_used,
                                    eps_inplane=eps_inplane, lam_perp=lam_perp, lam_M=lam_M,
                                    rtol_pinv=rtol_pinv)
            F2 = solve_wrench_vw_6d(S2_ext, E, I, L, Rp, tau_meas, delta_used,
                                    eps_inplane=eps_inplane, lam_perp=lam_perp, lam_M=lam_M,
                                    rtol_pinv=rtol_pinv)

            # Ground-truth wrench in base frame: only force in bending plane (no torque)
            R = Rz(-delta_used)
            Fgt_B = np.hstack([R @ np.array([Fx, 0.0, Fz]), np.zeros(3)])

            force_records.append({
                "case_idx": case_idx, "subplot": label,
                # GT in base frame:
                "FxB_gt": Fgt_B[0], "FyB_gt": Fgt_B[1], "FzB_gt": Fgt_B[2],
                # PCK0/1/2 estimated (base frame):
                "FxB_pck0": F0[0], "FyB_pck0": F0[1], "FzB_pck0": F0[2],
                "MxB_pck0": F0[3], "MyB_pck0": F0[4], "MzB_pck0": F0[5],
                "FxB_pck1": F1[0], "FyB_pck1": F1[1], "FzB_pck1": F1[2],
                "MxB_pck1": F1[3], "MyB_pck1": F1[4], "MzB_pck1": F1[5],
                "FxB_pck2": F2[0], "FyB_pck2": F2[1], "FzB_pck2": F2[2],
                "MxB_pck2": F2[3], "MyB_pck2": F2[4], "MzB_pck2": F2[5],
                # errors vs GT force components
                "FxB_err_pck0": F0[0]-Fgt_B[0], "FyB_err_pck0": F0[1]-Fgt_B[1], "FzB_err_pck0": F0[2]-Fgt_B[2],
                "FxB_err_pck1": F1[0]-Fgt_B[0], "FyB_err_pck1": F1[1]-Fgt_B[1], "FzB_err_pck1": F1[2]-Fgt_B[2],
                "FxB_err_pck2": F2[0]-Fgt_B[0], "FyB_err_pck2": F2[1]-Fgt_B[1], "FzB_err_pck2": F2[2]-Fgt_B[2],
            })

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=fs_tick)
    ax.set_title(f"{title_prefix} {r_idx+1}-{c_idx+1}: Fx={Fx:.2f}, Fz={Fz:.2f}", fontsize=fs_title)

# --------------------- Figure driver ---------------------

def _force_grid_levels(Fx_levels, Fz_levels):
    Fx_levels = [float(v) for v in Fx_levels]
    Fz_levels = [float(v) for v in Fz_levels]
    grid = [[(Fx_col, Fz_row) for Fx_col in Fx_levels] for Fz_row in Fz_levels]
    return grid, Fx_levels, Fz_levels

def _force_grid_compass3(Fx_mag, Fz_mag):
    Fx = float(Fx_mag); Fz = float(Fz_mag)
    Fz_levels = [ +Fz, 0.0, -Fz ]
    Fx_levels = [ -Fx, 0.0, +Fx ]
    grid = [[(Fx_col, Fz_row) for Fx_col in Fx_levels] for Fz_row in Fz_levels]
    return grid, Fx_levels, Fz_levels

def _compute_limits_for_grid(L, E, r, Rp, q, grid, lock_delta, delta_user):
    xmin = +np.inf; xmax = -np.inf
    zmin = +np.inf; zmax = -np.inf
    for row in grid:
        for Fx, Fz in row:
            x_gt, z_gt, _, _ = gt_inplane_shape(L, E, r, Rp, q, Fx, Fz, lock_delta, delta_user)
            xmin = min(xmin, float(np.min(x_gt))); xmax = max(xmax, float(np.max(x_gt)))
            zmin = min(zmin, float(np.min(z_gt))); zmax = max(zmax, float(np.max(z_gt)))
    return xmin, xmax, zmin, zmax

def _plot_one_figure(L, E, r, Rp, q, grid,
                     NTRIALS, sig_x, sig_z, sig_th, wp, wR,
                     lock_delta, delta_user, arrow_scale,
                     axis_mode, xlim_manual, zlim_manual, axis_margin,
                     style_pck_lw,
                     gt_markers, gt_construction,
                     title_prefix, fs_title, fs_label, fs_tick,
                     suptitle_on,
                     # tension noise:
                     sig_tau, bias_tau, scale_tau, drift_tau,
                     # shape table:
                     shape_records, shape_N,
                     # table bookkeeping:
                     collect_forces=False, force_records=None, case_idx=1,
                     # solver & weights:
                     solver="full6d",
                     eps_inplane=1e-6, lam_perp=1e6, lam_M=1e6, rtol_pinv=1e-5):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    maxF = max(math.hypot(Fx, Fz) for row in grid for Fx, Fz in row)
    arrow_pad = arrow_scale * L * maxF

    xlim_fixed = zlim_fixed = None
    if axis_mode == "fixed":
        xmin, xmax, zmin, zmax = _compute_limits_for_grid(L, E, r, Rp, q, grid, lock_delta, delta_user)
        dx = xmax - xmin; dz = zmax - zmin
        if dx == 0: dx = 1e-6
        if dz == 0: dz = 1e-6
        xmin -= axis_margin*dx; xmax += axis_margin*dx
        zmin -= axis_margin*dz; zmax += axis_margin*dz
        xmin -= arrow_pad; xmax += arrow_pad
        zmin -= arrow_pad; zmax += arrow_pad
        xlim_fixed = (xmin, xmax); zlim_fixed = (zmin, zmax)

    share_axes = (axis_mode in ["fixed","auto"])
    figsize = (4.4*cols, 4.4*rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False,
                             sharex=share_axes, sharey=share_axes,
                             constrained_layout=True)

    if suptitle_on:
        delta_deg = math.degrees(delta_user if lock_delta else delta_from_tensions(q))
        fig.suptitle(
            f"tau=[{q[0]:.1f},{q[1]:.1f},{q[2]:.1f},{q[3]:.1f}] N   |   δ={delta_deg:.1f}°   |   {VERSION}",
            fontsize=fs_title
        )

    for r_idx in range(rows):
        for c_idx in range(cols):
            Fx, Fz = grid[r_idx][c_idx]
            ax = axes[r_idx, c_idx]
            _plot_one_cell(ax, L, E, r, Rp, q, Fx, Fz,
                           NTRIALS, sig_x, sig_z, sig_th, wp, wR,
                           lock_delta, delta_user, arrow_scale,
                           style_pck_lw,
                           gt_markers, gt_construction,
                           fs_label, fs_tick,
                           sig_tau, bias_tau, scale_tau, drift_tau,
                           shape_records, shape_N,
                           collect_forces=collect_forces, force_records=force_records,
                           case_idx=case_idx, r_idx=r_idx, c_idx=c_idx,
                           title_prefix=title_prefix, fs_title=fs_title,
                           solver=solver, eps_inplane=eps_inplane, lam_perp=lam_perp, lam_M=lam_M,
                           rtol_pinv=rtol_pinv)

            if axis_mode == "fixed" and xlim_fixed and zlim_fixed:
                ax.set_xlim(*xlim_fixed); ax.set_ylim(*zlim_fixed)
            elif axis_mode == "manual":
                ax.set_xlim(*xlim_manual); ax.set_ylim(*zlim_manual)
            elif axis_mode == "cell-tight":
                ax.set_aspect('equal', adjustable='datalim')
                ax.relim(); ax.autoscale_view()
                ax.margins(x=0.02, y=0.02)

            if r_idx < rows-1: ax.set_xlabel("")
            else:              ax.set_xlabel("x (m)", fontsize=fs_label)
            if c_idx > 0:      ax.set_ylabel("")
            else:              ax.set_ylabel("z (m)", fontsize=fs_label)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(handles, labels, loc='best', fontsize=fs_tick)

# --------------------- Public API ---------------------

def plot_force_grids(L, E, r, Rp, tension_cases,
                     grid_mode="compass3", Fgrid_pairs=None,
                     Fx_levels=None, Fz_levels=None,
                     NTRIALS=1, sig_x=0.01, sig_z=0.01, sig_th=0.01,
                     wp=None, wR=None, lock_delta=False, delta_user=math.pi/4,
                     arrow_scale=0.08, axis_mode="fixed",
                     xlim=None, zlim=None, axis_margin=0.05,
                     style_pck_lw=2.0,
                     gt_markers=10, gt_construction=True,
                     title_prefix="Config.", fs_title=12.0, fs_label=12.0, fs_tick=10.0,
                     suptitle_on=False,
                     # tension noise:
                     sig_tau=0.02, bias_tau=0.0, scale_tau=0.0, drift_tau=0.0,
                     # shape table:
                     shape_table=False, shape_table_path="shape_table.csv",
                     shape_overall_mae=False, shape_overall_mae_path="shape_overall_mae.csv",
                     shape_N=200,
                     # force table:
                     force_table=False, force_table_path="force_table.csv",
                     # solver & weights:
                     solver="full6d",
                     eps_inplane=1e-6, lam_perp=1e6, lam_M=1e6, rtol_pinv=1e-5):
    if wR is None: wR = 1.0 / L

    if axis_mode == "manual":
        if xlim is None or zlim is None:
            raise ValueError("axis-mode=manual requires --xlim and --zlim.")
        xlim_manual = list(map(float, json.loads(xlim)))
        zlim_manual = list(map(float, json.loads(zlim)))
        if len(xlim_manual) != 2 or len(zlim_manual) != 2:
            raise ValueError("--xlim/--zlim must be 2-element lists.")
    else:
        xlim_manual = zlim_manual = None

    # collectors
    force_records = [] if force_table else None
    shape_records = [] if shape_table else None

    for case_idx, q in enumerate(tension_cases, start=1):
        q = np.asarray(q, float)
        if grid_mode == "compass3":
            if not Fgrid_pairs:
                raise ValueError("compass3 mode needs --Fgrid pairs.")
            for (Fx_mag, Fz_mag) in Fgrid_pairs:
                grid, _, _ = _force_grid_compass3(Fx_mag, Fz_mag)
                _plot_one_figure(L, E, r, Rp, q, grid,
                                 NTRIALS, sig_x, sig_z, sig_th, wp, wR,
                                 lock_delta, delta_user, arrow_scale,
                                 axis_mode, xlim_manual, zlim_manual, axis_margin,
                                 style_pck_lw,
                                 gt_markers, gt_construction,
                                 title_prefix, fs_title, fs_label, fs_tick,
                                 suptitle_on,
                                 sig_tau, bias_tau, scale_tau, drift_tau,
                                 shape_records, shape_N,
                                 collect_forces=force_table, force_records=force_records, case_idx=case_idx,
                                 solver=solver, eps_inplane=eps_inplane, lam_perp=lam_perp, lam_M=lam_M,
                                 rtol_pinv=rtol_pinv)
        elif grid_mode == "levels":
            if Fx_levels is None or Fz_levels is None:
                raise ValueError("levels mode needs --Fx_levels and --Fz_levels.")
            grid, _, _ = _force_grid_levels(Fx_levels, Fz_levels)
            _plot_one_figure(L, E, r, Rp, q, grid,
                             NTRIALS, sig_x, sig_z, sig_th, wp, wR,
                             lock_delta, delta_user, arrow_scale,
                             axis_mode, xlim_manual, zlim_manual, axis_margin,
                             style_pck_lw,
                             gt_markers, gt_construction,
                             title_prefix, fs_title, fs_label, fs_tick,
                             suptitle_on,
                             sig_tau, bias_tau, scale_tau, drift_tau,
                             shape_records, shape_N,
                             collect_forces=force_table, force_records=force_records, case_idx=case_idx,
                             solver=solver, eps_inplane=eps_inplane, lam_perp=lam_perp, lam_M=lam_M,
                             rtol_pinv=rtol_pinv)
        else:
            raise ValueError("grid-mode must be 'compass3' or 'levels'.")

    # -------- Write CSVs --------
    if shape_table and shape_records:
        with open(shape_table_path, "w", newline="") as f:
            fieldnames = [
                "case_idx","subplot","Fx_gt","Fz_gt",
                "P0_tip_pos_mm","P0_tip_rot_deg","P1_tip_pos_mm","P1_tip_rot_deg","P2_tip_pos_mm","P2_tip_rot_deg",
                "P0_shape_pos_mm","P0_shape_rot_deg","P1_shape_pos_mm","P1_shape_rot_deg","P2_shape_pos_mm","P2_shape_rot_deg",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in shape_records:
                writer.writerow(row)

        # Overall MAE computation
        arr = lambda k: np.array([r[k] for r in shape_records], dtype=float)
        mae = {
            "P0_tip_pos_mm":   float(np.mean(np.abs(arr("P0_tip_pos_mm")))),
            "P1_tip_pos_mm":   float(np.mean(np.abs(arr("P1_tip_pos_mm")))),
            "P2_tip_pos_mm":   float(np.mean(np.abs(arr("P2_tip_pos_mm")))),
            "P0_tip_rot_deg":  float(np.mean(np.abs(arr("P0_tip_rot_deg")))),
            "P1_tip_rot_deg":  float(np.mean(np.abs(arr("P1_tip_rot_deg")))),
            "P2_tip_rot_deg":  float(np.mean(np.abs(arr("P2_tip_rot_deg")))),
            "P0_shape_pos_mm": float(np.mean(np.abs(arr("P0_shape_pos_mm")))),
            "P1_shape_pos_mm": float(np.mean(np.abs(arr("P1_shape_pos_mm")))),
            "P2_shape_pos_mm": float(np.mean(np.abs(arr("P2_shape_pos_mm")))),
            "P0_shape_rot_deg":float(np.mean(np.abs(arr("P0_shape_rot_deg")))),
            "P1_shape_rot_deg":float(np.mean(np.abs(arr("P1_shape_rot_deg")))),
            "P2_shape_rot_deg":float(np.mean(np.abs(arr("P2_shape_rot_deg")))),
        }

        print("\n=== Shape Error MAE over all configurations ===")
        print(f"PCK0: tip_pos={mae['P0_tip_pos_mm']:.3f} mm, tip_rot={mae['P0_tip_rot_deg']:.3f} deg, "
              f"shape_pos={mae['P0_shape_pos_mm']:.3f} mm, shape_rot={mae['P0_shape_rot_deg']:.3f} deg")
        print(f"PCK1: tip_pos={mae['P1_tip_pos_mm']:.3f} mm, tip_rot={mae['P1_tip_rot_deg']:.3f} deg, "
              f"shape_pos={mae['P1_shape_pos_mm']:.3f} mm, shape_rot={mae['P1_shape_rot_deg']:.3f} deg")
        print(f"PCK2: tip_pos={mae['P2_tip_pos_mm']:.3f} mm, tip_rot={mae['P2_tip_rot_deg']:.3f} deg, "
              f"shape_pos={mae['P2_shape_pos_mm']:.3f} mm, shape_rot={mae['P2_shape_rot_deg']:.3f} deg")

        if shape_overall_mae:
            with open(shape_overall_mae_path, "w", newline="") as f2:
                writer = csv.DictWriter(f2, fieldnames=[
                    "Method","MAE_tip_pos_mm","MAE_tip_rot_deg","MAE_shape_pos_mm","MAE_shape_rot_deg"])
                writer.writeheader()
                writer.writerow({"Method":"PCK0","MAE_tip_pos_mm":mae["P0_tip_pos_mm"],"MAE_tip_rot_deg":mae["P0_tip_rot_deg"],
                                 "MAE_shape_pos_mm":mae["P0_shape_pos_mm"],"MAE_shape_rot_deg":mae["P0_shape_rot_deg"]})
                writer.writerow({"Method":"PCK1","MAE_tip_pos_mm":mae["P1_tip_pos_mm"],"MAE_tip_rot_deg":mae["P1_tip_rot_deg"],
                                 "MAE_shape_pos_mm":mae["P1_shape_pos_mm"],"MAE_shape_rot_deg":mae["P1_shape_rot_deg"]})
                writer.writerow({"Method":"PCK2","MAE_tip_pos_mm":mae["P2_shape_pos_mm"],"MAE_tip_rot_deg":mae["P2_shape_rot_deg"],
                                 "MAE_shape_pos_mm":mae["P2_shape_pos_mm"],"MAE_shape_rot_deg":mae["P2_shape_rot_deg"]})

    if force_table and force_records:
        # Decide columns based on solver
        plane2 = (solver == "plane2")
        if plane2:
            headers = [
                "case_idx","subplot","Fx_gt","Fz_gt",
                "Fx_pck0","Fz_pck0","Fx_err_pck0","Fz_err_pck0",
                "Fx_pck1","Fz_pck1","Fx_err_pck1","Fz_err_pck1",
                "Fx_pck2","Fz_pck2","Fx_err_pck2","Fz_err_pck2",
            ]
        else:
            headers = [
                "case_idx","subplot",
                "FxB_gt","FyB_gt","FzB_gt",
                "FxB_pck0","FyB_pck0","FzB_pck0","MxB_pck0","MyB_pck0","MzB_pck0",
                "FxB_pck1","FyB_pck1","FzB_pck1","MxB_pck1","MyB_pck1","MzB_pck1",
                "FxB_pck2","FyB_pck2","FzB_pck2","MxB_pck2","MyB_pck2","MzB_pck2",
                "FxB_err_pck0","FyB_err_pck0","FzB_err_pck0",
                "FxB_err_pck1","FyB_err_pck1","FzB_err_pck1",
                "FxB_err_pck2","FyB_err_pck2","FzB_err_pck2",
            ]

        with open(force_table_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in force_records:
                writer.writerow(row)

        # Print quick MAEs
        if plane2:
            err0 = np.array([[r["Fx_err_pck0"], r["Fz_err_pck0"]] for r in force_records])
            err1 = np.array([[r["Fx_err_pck1"], r["Fz_err_pck1"]] for r in force_records])
            err2 = np.array([[r["Fx_err_pck2"], r["Fz_err_pck2"]] for r in force_records])
            print("\n=== Force Error MAE over all configurations (N) ===")
            print(f"PCK0: MAE(Fx)={np.nanmean(np.abs(err0[:,0])):.4f}, MAE(Fz)={np.nanmean(np.abs(err0[:,1])):.4f}, "
                  f"MAE(|e|)={np.nanmean(np.linalg.norm(err0,axis=1)):.4f}")
            print(f"PCK1: MAE(Fx)={np.nanmean(np.abs(err1[:,0])):.4f}, MAE(Fz)={np.nanmean(np.abs(err1[:,1])):.4f}, "
                  f"MAE(|e|)={np.nanmean(np.linalg.norm(err1,axis=1)):.4f}")
            print(f"PCK2: MAE(Fx)={np.nanmean(np.abs(err2[:,0])):.4f}, MAE(Fz)={np.nanmean(np.abs(err2[:,1])):.4f}, "
                  f"MAE(|e|)={np.nanmean(np.linalg.norm(err2,axis=1)):.4f}")
        else:
            # only force components MAE vs GT
            ex0 = np.array([r["FxB_err_pck0"] for r in force_records])
            ey0 = np.array([r["FyB_err_pck0"] for r in force_records])
            ez0 = np.array([r["FzB_err_pck0"] for r in force_records])
            ex1 = np.array([r["FxB_err_pck1"] for r in force_records])
            ey1 = np.array([r["FyB_err_pck1"] for r in force_records])
            ez1 = np.array([r["FzB_err_pck1"] for r in force_records])
            ex2 = np.array([r["FxB_err_pck2"] for r in force_records])
            ey2 = np.array([r["FyB_err_pck2"] for r in force_records])
            ez2 = np.array([r["FzB_err_pck2"] for r in force_records])

            print("\n=== Base-frame Force Error MAE over all configurations (N) ===")
            print(f"PCK0: MAE(Fx)={np.nanmean(np.abs(ex0)):.4f}, MAE(Fy)={np.nanmean(np.abs(ey0)):.4f}, MAE(Fz)={np.nanmean(np.abs(ez0)):.4f}")
            print(f"PCK1: MAE(Fx)={np.nanmean(np.abs(ex1)):.4f}, MAE(Fy)={np.nanmean(np.abs(ey1)):.4f}, MAE(Fz)={np.nanmean(np.abs(ez1)):.4f}")
            print(f"PCK2: MAE(Fx)={np.nanmean(np.abs(ex2)):.4f}, MAE(Fy)={np.nanmean(np.abs(ey2)):.4f}, MAE(Fz)={np.nanmean(np.abs(ez2)):.4f}")

    plt.show()

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description=f"PCK vs GT with 6D wrench solver (Option A). [{VERSION}]")
    ap.add_argument("--L", type=float, default=0.10)
    ap.add_argument("--E", type=float, default=60e9)
    ap.add_argument("--r", type=float, default=0.0005)
    ap.add_argument("--Rp", type=float, default=0.008)
    ap.add_argument("--tensions", type=str, default="[[2,2,0,0],[4,4,0,0]]")

    ap.add_argument("--grid-mode", type=str, default="compass3",
                    choices=["compass3","levels"])
    ap.add_argument("--Fgrid", type=str, default="[[1.0,1.0]]")
    ap.add_argument("--Fx_levels", type=str, default=None)
    ap.add_argument("--Fz_levels", type=str, default=None)

    ap.add_argument("--NTRIALS", type=int, default=5)
    ap.add_argument("--sig_x", type=float, default=0.0005)
    ap.add_argument("--sig_z", type=float, default=0.0005)
    ap.add_argument("--sig_th", type=float, default=0.005)
    ap.add_argument("--wp", type=float, default=2000)
    ap.add_argument("--wR", type=float, default=200)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--lock-delta", action="store_true")
    ap.add_argument("--delta", type=float, default=math.pi/4)

    ap.add_argument("--arrow-scale", type=float, default=0.08)
    ap.add_argument("--axis-mode", type=str, default="fixed",
                    choices=["fixed","auto","manual","cell-tight"])
    ap.add_argument("--xlim", type=str, default=None)
    ap.add_argument("--zlim", type=str, default=None)
    ap.add_argument("--axis-margin", type=float, default=0.05)
    ap.add_argument("--pck-lw", type=float, default=2.0)

    ap.add_argument("--gt-markers", type=int, default=10)
    ap.add_argument("--gt-construction", action="store_true")

    ap.add_argument("--title-prefix", type=str, default="Config.")
    ap.add_argument("--fs-title", type=float, default=12.0)
    ap.add_argument("--fs-label", type=float, default=12.0)
    ap.add_argument("--fs-tick",  type=float, default=10.0)
    ap.add_argument("--suptitle", action="store_true")

    # tension noise
    ap.add_argument("--sig_tau", type=float, default=0.02)
    ap.add_argument("--bias_tau", type=float, default=0.0)
    ap.add_argument("--drift_tau", type=float, default=0.0)
    ap.add_argument("--scale_tau", type=float, default=0.0)

    # shape tables
    ap.add_argument("--shape-table", action="store_true")
    ap.add_argument("--shape-table-path", type=str, default="shape_table.csv")
    ap.add_argument("--shape-overall-mae", action="store_true")
    ap.add_argument("--shape-overall-mae-path", type=str, default="shape_overall_mae.csv")
    ap.add_argument("--shape-N", type=int, default=200)

    # force table
    ap.add_argument("--force-table", action="store_true")
    ap.add_argument("--force-table-path", type=str, default="force_table.csv")

    # --- NEW: solver choice and weights (Option A) ---
    ap.add_argument("--solver", type=str, default="full6d",
                    choices=["full6d","plane2"],
                    help="Use general 6D solver (Option A) or legacy 2D planar solver.")
    ap.add_argument("--w_inplane", type=float, default=1e-6,
                    help="Small weight on in-plane force components in Frame{1}.")
    ap.add_argument("--w_perp", type=float, default=1e6,
                    help="Large weight on force component orthogonal to the bending plane (Fy1).")
    ap.add_argument("--w_torque", type=float, default=1e6,
                    help="Large weights on torque components (Mx,My,Mz).")
    ap.add_argument("--rtol_pinv", type=float, default=1e-5,
                    help="Relative SVD cutoff for pseudoinverses (after scaling).")

    args = ap.parse_args()
    try:
        tension_cases = json.loads(args.tensions)
    except Exception as e:
        raise ValueError(f"Failed to parse --tensions JSON: {e}")

    if args.wR is None: args.wR = 1.0 / args.L
    if args.seed is not None: np.random.seed(args.seed)

    grid_mode = args.grid_mode
    Fgrid_pairs = None; Fx_levels = None; Fz_levels = None

    if grid_mode == "compass3":
        try:
            Fgrid_pairs = json.loads(args.Fgrid)
            Fgrid_pairs = [list(map(float, pair)) for pair in Fgrid_pairs]
            for pair in Fgrid_pairs:
                if len(pair) != 2: raise ValueError
        except Exception:
            raise ValueError('Failed to parse --Fgrid. Use e.g. --Fgrid "[[2.0,1.0]]"')
    elif grid_mode == "levels":
        if args.Fx_levels is None or args.Fz_levels is None:
            raise ValueError("levels mode requires --Fx_levels and --Fz_levels.")
        try:
            Fx_levels = [float(v) for v in json.loads(args.Fx_levels)]
            Fz_levels = [float(v) for v in json.loads(args.Fz_levels)]
        except Exception:
            raise ValueError('Failed to parse --Fx_levels/--Fz_levels.')

    plot_force_grids(
        L=args.L, E=args.E, r=args.r, Rp=args.Rp,
        tension_cases=tension_cases,
        grid_mode=grid_mode, Fgrid_pairs=Fgrid_pairs,
        Fx_levels=Fx_levels, Fz_levels=Fz_levels,
        NTRIALS=args.NTRIALS, sig_x=args.sig_x, sig_z=args.sig_z, sig_th=args.sig_th,
        wp=args.wp, wR=args.wR, lock_delta=args.lock_delta, delta_user=args.delta,
        arrow_scale=args.arrow_scale, axis_mode=args.axis_mode,
        xlim=args.xlim, zlim=args.zlim, axis_margin=args.axis_margin,
        style_pck_lw=args.pck_lw,
        gt_markers=args.gt_markers, gt_construction=args.gt_construction,
        title_prefix=args.title_prefix, fs_title=args.fs_title, fs_label=args.fs_label, fs_tick=args.fs_tick,
        suptitle_on=args.suptitle,
        sig_tau=args.sig_tau, bias_tau=args.bias_tau, scale_tau=args.scale_tau, drift_tau=args.drift_tau,
        shape_table=args.shape_table, shape_table_path=args.shape_table_path,
        shape_overall_mae=args.shape_overall_mae, shape_overall_mae_path=args.shape_overall_mae_path,
        shape_N=args.shape_N,
        force_table=args.force_table, force_table_path=args.force_table_path,
        solver=args.solver,
        eps_inplane=args.w_inplane, lam_perp=args.w_perp, lam_M=args.w_torque, rtol_pinv=args.rtol_pinv
    )

if __name__ == "__main__":
    main()


