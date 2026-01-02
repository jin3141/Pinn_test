"""
Burgers方程式の数値解法（SciPy ODE solverを使用）
"""
import numpy as np
from scipy.integrate import solve_ivp


def solve_burgers_scipy(x_range, t_range, nx=100, nt=100, nu=0.01/np.pi):
    """
    Burgers方程式をSciPyのODEソルバーで解く（Method of Lines）

    ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²

    初期条件: u(x, 0) = -sin(πx)
    境界条件: u(-1, t) = u(1, t) = 0

    Args:
        x_range: [x_min, x_max]
        t_range: [t_min, t_max]
        nx: 空間グリッド数
        nt: 時間グリッド数
        nu: 粘性係数

    Returns:
        x, t, u
    """
    x_min, x_max = x_range
    t_min, t_max = t_range

    # 空間グリッド
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]

    # 時間グリッド
    t_eval = np.linspace(t_min, t_max, nt)

    # 初期条件
    u0 = -np.sin(np.pi * x)
    u0[0] = 0.0   # 境界条件
    u0[-1] = 0.0  # 境界条件

    def burgers_rhs(t, u):
        """
        Burgers方程式の右辺
        ∂u/∂t = -u * ∂u/∂x + ν * ∂²u/∂x²
        """
        dudt = np.zeros_like(u)

        # 内部点
        for i in range(1, nx-1):
            # 空間微分（中心差分）
            u_x = (u[i+1] - u[i-1]) / (2 * dx)
            u_xx = (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)

            # Burgers方程式
            dudt[i] = -u[i] * u_x + nu * u_xx

        # 境界条件（ディリクレ: u = 0）
        dudt[0] = 0.0
        dudt[-1] = 0.0

        return dudt

    print(f"SciPy solver: nx={nx}, nt={nt}, solving...")

    # ODEを解く（BDF法：陰的で安定）
    sol = solve_ivp(
        burgers_rhs,
        [t_min, t_max],
        u0,
        t_eval=t_eval,
        method='BDF',  # Backward Differentiation Formula (stiff problems用)
        rtol=1e-6,
        atol=1e-8
    )

    if not sol.success:
        print(f"Warning: ODE solver did not converge: {sol.message}")

    # 結果を整形
    u = sol.y.T  # (nt, nx)

    print(f"  Solved {sol.nfev} function evaluations")
    print(f"  Status: {sol.message}")

    return x, sol.t, u
