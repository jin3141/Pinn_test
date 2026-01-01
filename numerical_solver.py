"""
Burgers方程式の数値解法（有限差分法）

Burgers方程式: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_burgers_fdm(x_range, t_range, nx=256, nt=100, nu=0.01/np.pi, method='implicit'):
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

    if method == 'explicit':
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
    alpha = nu * dt / (2 * dx**2)
    beta = dt / (4 * dx)

    # 係数行列の構築
    # u^{n+1}_i - α(u^{n+1}_{i+1} - 2u^{n+1}_i + u^{n+1}_{i-1})
    # - β*u^n_i*(u^{n+1}_{i+1} - u^{n+1}_{i-1})
    # = u^n_i + α(u^n_{i+1} - 2u^n_i + u^n_{i-1})
    #   + β*u^n_i*(u^n_{i+1} - u^n_{i-1})

    # 左辺の係数行列（陰的部分）
    main_diag = np.ones(nx)
    upper_diag = np.zeros(nx - 1)
    lower_diag = np.zeros(nx - 1)

    for i in range(1, nx - 1):
        main_diag[i] = 1 + 2 * alpha
        upper_diag[i] = -alpha - beta * u_old[i]
        if i > 0:
            lower_diag[i-1] = -alpha + beta * u_old[i]

    # 境界条件
    main_diag[0] = 1
    main_diag[-1] = 1
    if nx > 1:
        upper_diag[0] = 0
        lower_diag[-1] = 0

    # 三重対角行列
    A = sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')

    # 右辺の計算（陽的部分）
    b = u_old.copy()
    for i in range(1, nx - 1):
        u_xx = (u_old[i+1] - 2 * u_old[i] + u_old[i-1]) / dx**2
        u_x = (u_old[i+1] - u_old[i-1]) / (2 * dx)
        b[i] = u_old[i] + alpha * (u_old[i+1] - 2 * u_old[i] + u_old[i-1]) \
               + beta * u_old[i] * (u_old[i+1] - u_old[i-1])

    # 境界条件
    b[0] = 0
    b[-1] = 0

    # 線形システムを解く
    u_new = spsolve(A, b)

    return u_new


def compute_error_metrics(u_pinn, u_fdm):
    """
    誤差指標を計算

    Args:
        u_pinn: PINNによる解
        u_fdm: 数値解法による解

    Returns:
        L2相対誤差、最大絶対誤差
    """
    # L2相対誤差
    l2_error = np.linalg.norm(u_pinn - u_fdm) / np.linalg.norm(u_fdm)

    # 最大絶対誤差
    max_error = np.max(np.abs(u_pinn - u_fdm))

    # 平均絶対誤差
    mean_error = np.mean(np.abs(u_pinn - u_fdm))

    return {
        'L2_relative_error': l2_error,
        'max_absolute_error': max_error,
        'mean_absolute_error': mean_error
    }
