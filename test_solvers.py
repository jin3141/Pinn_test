#!/usr/bin/env python3
"""
Burgers方程式の各ソルバーを検証するテスト
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要
import matplotlib.pyplot as plt
from numerical_solver import solve_burgers_fdm, solve_burgers_analytical, compute_error_metrics

print("=" * 60)
print("Burgers方程式ソルバーの検証テスト")
print("=" * 60)

# パラメータ設定
x_range = [-1, 1]
t_range = [0, 1]
nu = 0.01 / np.pi
nx = 100
nt = 50

print(f"\nパラメータ:")
print(f"  空間範囲: {x_range}")
print(f"  時間範囲: {t_range}")
print(f"  粘性係数 ν: {nu:.6f}")
print(f"  空間グリッド数: {nx}")
print(f"  時間グリッド数: {nt}")

# 1. 数値解法（有限差分法）のテスト
print("\n" + "=" * 60)
print("1. 数値解法（有限差分法）のテスト")
print("=" * 60)
try:
    x_fdm, t_fdm, u_fdm = solve_burgers_fdm(
        x_range, t_range, nx=nx, nt=nt, nu=nu, method='implicit'
    )
    print(f"✓ FDM計算完了")
    print(f"  解の形状: {u_fdm.shape}")
    print(f"  初期値 u(0, 0): {u_fdm[0, nx//2]:.6f} (期待値: 0.0)")
    print(f"  最終値 u({t_range[1]}, 0): {u_fdm[-1, nx//2]:.6f}")
    print(f"  最小値: {np.min(u_fdm):.6f}")
    print(f"  最大値: {np.max(u_fdm):.6f}")

    # 初期条件の確認
    x_test = x_fdm[nx//2]
    u_initial_expected = -np.sin(np.pi * x_test)
    u_initial_actual = u_fdm[0, nx//2]
    print(f"  初期条件チェック: u({x_test:.3f}, 0) = {u_initial_actual:.6f} (期待: {u_initial_expected:.6f})")

    fdm_success = True
except Exception as e:
    print(f"✗ FDMエラー: {e}")
    fdm_success = False
    import traceback
    traceback.print_exc()

# 2. 解析解のテスト
print("\n" + "=" * 60)
print("2. 解析解（Cole-Hopf変換）のテスト")
print("=" * 60)
try:
    # 解析解は計算コストが高いので、粗いグリッドでテスト
    nx_ana = 30
    nt_ana = 20
    print(f"  計算コスト削減のため nx={nx_ana}, nt={nt_ana} で計算")

    x_ana, t_ana, u_ana = solve_burgers_analytical(
        x_range, t_range, nx=nx_ana, nt=nt_ana, nu=nu
    )
    print(f"✓ 解析解計算完了")
    print(f"  解の形状: {u_ana.shape}")
    print(f"  初期値 u(0, 0): {u_ana[0, nx_ana//2]:.6f} (期待値: 0.0)")
    print(f"  最終値 u({t_range[1]}, 0): {u_ana[-1, nx_ana//2]:.6f}")
    print(f"  最小値: {np.min(u_ana):.6f}")
    print(f"  最大値: {np.max(u_ana):.6f}")

    analytical_success = True
except Exception as e:
    print(f"✗ 解析解エラー: {e}")
    analytical_success = False
    import traceback
    traceback.print_exc()

# 3. FDMと解析解の比較
if fdm_success and analytical_success:
    print("\n" + "=" * 60)
    print("3. FDMと解析解の比較")
    print("=" * 60)

    # 同じグリッドで再計算
    nx_comp = 50
    nt_comp = 30
    print(f"  比較用に nx={nx_comp}, nt={nt_comp} で再計算")

    x_fdm_comp, t_fdm_comp, u_fdm_comp = solve_burgers_fdm(
        x_range, t_range, nx=nx_comp, nt=nt_comp, nu=nu, method='implicit'
    )

    x_ana_comp, t_ana_comp, u_ana_comp = solve_burgers_analytical(
        x_range, t_range, nx=nx_comp, nt=nt_comp, nu=nu
    )

    errors = compute_error_metrics(u_fdm_comp, u_ana_comp)
    print(f"  L2相対誤差: {errors['L2_relative_error']:.6e}")
    print(f"  最大絶対誤差: {errors['max_absolute_error']:.6e}")
    print(f"  平均絶対誤差: {errors['mean_absolute_error']:.6e}")

    if errors['L2_relative_error'] < 0.1:
        print("  ✓ FDMと解析解は良好に一致しています")
    elif errors['L2_relative_error'] < 0.3:
        print("  ⚠ FDMと解析解にやや大きな誤差があります")
    else:
        print("  ✗ FDMと解析解の誤差が大きすぎます - 実装に問題がある可能性")

# 4. 物理的妥当性のチェック
print("\n" + "=" * 60)
print("4. 物理的妥当性のチェック")
print("=" * 60)

if fdm_success:
    # エネルギー減衰のチェック（粘性による散逸）
    energy = np.sum(u_fdm**2, axis=1)
    energy_decrease = energy[0] > energy[-1]
    print(f"  初期エネルギー: {energy[0]:.6f}")
    print(f"  最終エネルギー: {energy[-1]:.6f}")
    print(f"  エネルギー減衰: {'✓ はい' if energy_decrease else '✗ いいえ (問題!)'}")

    # 境界条件のチェック
    bc_satisfied = np.allclose(u_fdm[:, 0], 0.0) and np.allclose(u_fdm[:, -1], 0.0)
    print(f"  境界条件 u(-1,t)=0: {'✓ 満たす' if np.allclose(u_fdm[:, 0], 0.0) else '✗ 満たさない'}")
    print(f"  境界条件 u(1,t)=0: {'✓ 満たす' if np.allclose(u_fdm[:, -1], 0.0) else '✗ 満たさない'}")

# 5. プロット生成
print("\n" + "=" * 60)
print("5. 結果の可視化")
print("=" * 60)

if fdm_success:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # FDMの時空間プロット
    ax = axes[0, 0]
    im = ax.contourf(x_fdm, t_fdm, u_fdm, levels=20, cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('FDM Solution')
    plt.colorbar(im, ax=ax)

    # FDMの時間スナップショット
    ax = axes[0, 1]
    times_to_plot = [0, nt//3, 2*nt//3, nt-1]
    for i in times_to_plot:
        ax.plot(x_fdm, u_fdm[i, :], label=f't={t_fdm[i]:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('FDM: u(x,t) at different times')
    ax.legend()
    ax.grid(True)

    if analytical_success:
        # 解析解の時空間プロット
        ax = axes[1, 0]
        im = ax.contourf(x_ana_comp, t_ana_comp, u_ana_comp, levels=20, cmap='RdBu_r')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title('Analytical Solution')
        plt.colorbar(im, ax=ax)

        # 誤差プロット
        ax = axes[1, 1]
        error_map = np.abs(u_fdm_comp - u_ana_comp)
        im = ax.contourf(x_fdm_comp, t_fdm_comp, error_map, levels=20, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title('Absolute Error: |FDM - Analytical|')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('burgers_validation.png', dpi=100)
    print("  ✓ プロットを burgers_validation.png に保存しました")

print("\n" + "=" * 60)
print("検証完了")
print("=" * 60)
