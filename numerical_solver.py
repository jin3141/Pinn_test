"""
Burgers方程式の数値解法（有限差分法）

Burgers方程式: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_burgers_fdm(x_range, t_range, nx=256, nt=100, nu=0.01/np.pi, method='scipy'):
    """
    有限差分法でBurgers方程式を解く

    Args:
        x_range: [x_min, x_max]
        t_range: [t_min, t_max]
        nx: 空間方向のグリッド数
        nt: 時間方向のグリッド数
        nu: 粘性係数
        method: 'implicit'（陰解法）または 'explicit'（陽解法）

    Returns:
        x: 空間座標配列
        t: 時間座標配列
        u: 解 (nt x nx)
    """
    x_min, x_max = x_range
    t_min, t_max = t_range

    # グリッドの生成
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(t_min, t_max, nt)

    dx = (x_max - x_min) / (nx - 1)
    dt = (t_max - t_min) / (nt - 1)

    # 初期条件: u(x, 0) = -sin(πx)
    u = np.zeros((nt, nx))
    u[0, :] = -np.sin(np.pi * x)

    # 境界条件: u(-1, t) = u(1, t) = 0
    u[:, 0] = 0
    u[:, -1] = 0

    if method == 'scipy':
        # SciPy ODE solverを使用（最も安定）
        from scipy.integrate import solve_ivp

        def burgers_rhs(t_val, u_vec):
            """Burgers方程式の右辺"""
            dudt = np.zeros_like(u_vec)
            for i in range(1, nx-1):
                u_x = (u_vec[i+1] - u_vec[i-1]) / (2 * dx)
                u_xx = (u_vec[i+1] - 2*u_vec[i] + u_vec[i-1]) / (dx**2)
                dudt[i] = -u_vec[i] * u_x + nu * u_xx
            dudt[0] = 0.0
            dudt[-1] = 0.0
            return dudt

        t_eval = t
        sol = solve_ivp(burgers_rhs, [t_min, t_max], u[0, :], t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)
        u = sol.y.T
        return x, sol.t, u

    elif method == 'explicit':
        # 陽解法（Forward Time, Central Space）
        # 安定性条件が厳しいので注意
        alpha = nu * dt / dx**2
        beta = dt / (2 * dx)

        print(f"FDM: dx={dx:.6f}, dt={dt:.6f}, alpha={alpha:.6f}, beta={beta:.6f}")

        for n in range(0, nt - 1):
            for i in range(1, nx - 1):
                u_x = (u[n, i+1] - u[n, i-1]) / (2 * dx)
                u_xx = (u[n, i+1] - 2 * u[n, i] + u[n, i-1]) / dx**2

                u[n+1, i] = u[n, i] - dt * u[n, i] * u_x + dt * nu * u_xx

            # 境界条件を再適用
            u[n+1, 0] = 0
            u[n+1, -1] = 0

    elif method == 'implicit':
        # 陰解法（Crank-Nicolson法）
        # より安定だが、各時間ステップで線形システムを解く必要がある
        alpha = nu * dt / (2 * dx**2)

        for n in range(0, nt - 1):
            # 非線形項の処理（線形化）
            # u^{n+1} を求めるために反復法または線形化を使用
            # ここでは簡単のため、前の時間ステップの値で非線形項を評価
            u_old = u[n, :].copy()

            # 線形化: u * ∂u/∂x ≈ u^n * ∂u^{n+1}/∂x
            # これを陰的に解く
            u_new = solve_implicit_step(u_old, dx, dt, nu, nx)
            u[n+1, :] = u_new

            # 境界条件を再適用
            u[n+1, 0] = 0
            u[n+1, -1] = 0

    return x, t, u


def solve_implicit_step(u_old, dx, dt, nu, nx):
    """
    Crank-Nicolson法で1ステップ進める（線形化版）
    """
    # 係数
    alpha = nu * dt / (dx**2)
    beta = dt / (2 * dx)

    # 係数行列の構築 (陰的Crank-Nicolson)
    # (1 + α)u^{n+1}_i - α/2(u^{n+1}_{i+1} + u^{n+1}_{i-1}) - β/2*u^n_i*(u^{n+1}_{i+1} - u^{n+1}_{i-1})
    # = (1 - α)u^n_i + α/2(u^n_{i+1} + u^n_{i-1}) + β/2*u^n_i*(u^n_{i+1} - u^n_{i-1})

    # 三重対角行列の要素
    main_diag = np.ones(nx) * (1.0 + alpha)
    upper_diag = np.zeros(nx - 1)
    lower_diag = np.zeros(nx - 1)

    # 右辺ベクトル
    b = np.zeros(nx)

    for i in range(1, nx - 1):
        # 左辺（陰的部分）の係数
        lower_diag[i-1] = -alpha / 2.0 + beta / 2.0 * u_old[i]
        upper_diag[i] = -alpha / 2.0 - beta / 2.0 * u_old[i]

        # 右辺（陽的部分）
        b[i] = (1.0 - alpha) * u_old[i] + \
               (alpha / 2.0) * (u_old[i+1] + u_old[i-1]) + \
               (beta / 2.0) * u_old[i] * (u_old[i+1] - u_old[i-1])

    # 境界条件
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    b[0] = 0.0
    b[-1] = 0.0

    if nx > 1:
        upper_diag[0] = 0.0
    if nx > 1:
        lower_diag[-1] = 0.0

    # 三重対角行列
    A = sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')

    # 線形システムを解く
    u_new = spsolve(A, b)

    return u_new


def solve_burgers_analytical(x_range, t_range, nx=256, nt=100, nu=0.01/np.pi):
    """
    Burgers方程式の解析解（Cole-Hopf変換による）

    初期条件: u(x, 0) = -sin(πx)
    境界条件: u(-1, t) = u(1, t) = 0

    Cole-Hopf変換: u = -2ν * ∂φ/∂x / φ
    これによりBurgers方程式は熱方程式に変換される: ∂φ/∂t = ν * ∂²φ/∂x²

    Args:
        x_range: [x_min, x_max]
        t_range: [t_min, t_max]
        nx: 空間方向のグリッド数
        nt: 時間方向のグリッド数
        nu: 粘性係数

    Returns:
        x: 空間座標配列
        t: 時間座標配列
        u: 解析解 (nt x nx)
    """
    from scipy.integrate import quad
    from scipy.special import erf

    x_min, x_max = x_range
    t_min, t_max = t_range

    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(t_min, t_max, nt)

    u = np.zeros((nt, nx))

    # Cole-Hopf変換を使った解析解の計算
    # φ(x,t)を熱方程式の基本解を使って計算
    for i_t, t_val in enumerate(t):
        if t_val < 1e-10:  # t = 0 の場合は初期条件を使用
            u[i_t, :] = -np.sin(np.pi * x)
        else:
            for i_x, x_val in enumerate(x):
                # φとその導関数を数値積分で計算
                # φ(x,t) = ∫ G(x,y,t) * φ0(y) dy
                # G(x,y,t) = 1/√(4πνt) * exp(-(x-y)²/(4νt)) : 熱方程式の基本解
                # φ0(y) = exp(∫[0 to y] u0(s)/(2ν) ds) = exp(cos(πy)/(2νπ))

                def integrand_phi(y):
                    """φの積分"""
                    G = 1.0 / np.sqrt(4 * np.pi * nu * t_val) * \
                        np.exp(-(x_val - y)**2 / (4 * nu * t_val))
                    # u0(y) = -sin(πy) なので、∫ u0(s)/(2ν) ds = (cos(πy) - 1)/(2νπ)
                    # より安定な計算のため、exp((cos(πy) - 1)/(2νπ)) を使用
                    phi0 = np.exp((np.cos(np.pi * y) - 1.0) / (2 * nu * np.pi))
                    return G * phi0

                def integrand_dphi(y):
                    """∂φ/∂x の積分"""
                    G = 1.0 / np.sqrt(4 * np.pi * nu * t_val) * \
                        np.exp(-(x_val - y)**2 / (4 * nu * t_val))
                    dG_dx = -(x_val - y) / (2 * nu * t_val) * G
                    # u0(y) = -sin(πy) より φ0 を計算
                    phi0 = np.exp((np.cos(np.pi * y) - 1.0) / (2 * nu * np.pi))
                    return dG_dx * phi0

                # 数値積分で計算（積分範囲を [-3, 3] 程度に制限して効率化）
                y_min = max(x_min, x_val - 5 * np.sqrt(nu * t_val))
                y_max = min(x_max, x_val + 5 * np.sqrt(nu * t_val))

                try:
                    phi, _ = quad(integrand_phi, y_min, y_max, limit=100)
                    dphi_dx, _ = quad(integrand_dphi, y_min, y_max, limit=100)

                    # u = -2ν * (∂φ/∂x) / φ
                    if abs(phi) > 1e-10:
                        u[i_t, i_x] = -2 * nu * dphi_dx / phi
                    else:
                        u[i_t, i_x] = 0.0
                except:
                    u[i_t, i_x] = 0.0

        # 境界条件を適用
        u[i_t, 0] = 0.0
        u[i_t, -1] = 0.0

    return x, t, u


def compute_error_metrics(u_test, u_reference):
    """
    誤差指標を計算

    Args:
        u_test: 検証対象の解（例：PINNやFDM）
        u_reference: 参照解（例：解析解やFDM）

    Returns:
        L2相対誤差、最大絶対誤差、平均絶対誤差
    """
    # L2相対誤差
    l2_error = np.linalg.norm(u_test - u_reference) / np.linalg.norm(u_reference)

    # 最大絶対誤差
    max_error = np.max(np.abs(u_test - u_reference))

    # 平均絶対誤差
    mean_error = np.mean(np.abs(u_test - u_reference))

    return {
        'L2_relative_error': l2_error,
        'max_absolute_error': max_error,
        'mean_absolute_error': mean_error
    }
