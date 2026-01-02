#!/usr/bin/env python3
"""
PINN、数値解法、解析解の統合テスト
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pinn_burgers import BurgersPINN
from numerical_solver import solve_burgers_fdm, compute_error_metrics

print("="*70)
print("Burgers方程式: PINN vs 数値解法 統合テスト")
print("="*70)

# パラメータ
x_range = [-1, 1]
t_range = [0, 0.5]  # 短い時間で検証
nu = 0.01 / np.pi

print(f"\nパラメータ:")
print(f"  空間範囲: {x_range}")
print(f"  時間範囲: {t_range}")
print(f"  粘性係数 ν: {nu:.6f}")

# 1. 数値解法で参照解を計算
print("\n" + "="*70)
print("1. 数値解法（参照解）")
print("="*70)

nx_ref = 100
nt_ref = 50

x_ref, t_ref, u_ref = solve_burgers_fdm(
    x_range, t_range, nx=nx_ref, nt=nt_ref, nu=nu, method='scipy'
)

print(f"  グリッド: nx={nx_ref}, nt={nt_ref}")
print(f"  解の範囲: [{np.min(u_ref):.6f}, {np.max(u_ref):.6f}]")

energy_ref = np.sum(u_ref**2, axis=1) * (x_ref[1] - x_ref[0])
print(f"  初期エネルギー: {energy_ref[0]:.6f}")
print(f"  最終エネルギー: {energy_ref[-1]:.6f}")
print(f"  エネルギー減衰: {(energy_ref[0] - energy_ref[-1])/energy_ref[0]*100:.2f}%")

# 2. PINNで解を計算
print("\n" + "="*70)
print("2. PINN")
print("="*70)

# 訓練データの準備
n_bc = 50
n_ic = 50
n_f = 500

# 境界条件データ
np.random.seed(42)
t_bc = np.random.uniform(t_range[0], t_range[1], (n_bc, 1))
x_bc_left = np.ones((n_bc//2, 1)) * x_range[0]
x_bc_right = np.ones((n_bc - n_bc//2, 1)) * x_range[1]
x_bc = np.vstack([x_bc_left, x_bc_right])
u_bc = np.zeros((n_bc, 1))

# 初期条件データ
x_ic = np.random.uniform(x_range[0], x_range[1], (n_ic, 1))
t_ic = np.zeros((n_ic, 1))
u_ic = -np.sin(np.pi * x_ic)

# PDE残差計算用データ
x_f = np.random.uniform(x_range[0], x_range[1], (n_f, 1))
t_f = np.random.uniform(t_range[0], t_range[1], (n_f, 1))

print(f"  訓練データ: BC={n_bc}, IC={n_ic}, PDE={n_f}")

# PINNモデル
pinn = BurgersPINN(layers_dims=[2, 40, 40, 40, 1], nu=nu, lr=0.001)

print(f"  訓練開始...")
pinn.fit(
    x_bc, t_bc, u_bc,
    x_ic, t_ic, u_ic,
    x_f, t_f,
    epochs=2000,
    print_every=500
)

# PINNで予測
u_pinn = np.zeros((nt_ref, nx_ref))
for i_t in range(nt_ref):
    for i_x in range(nx_ref):
        x_test = np.array([[x_ref[i_x]]])
        t_test = np.array([[t_ref[i_t]]])
        u_pinn[i_t, i_x] = pinn.predict(x_test, t_test)[0, 0]

print(f"  予測完了")
print(f"  解の範囲: [{np.min(u_pinn):.6f}, {np.max(u_pinn):.6f}]")

# 3. 誤差評価
print("\n" + "="*70)
print("3. 誤差評価")
print("="*70)

errors = compute_error_metrics(u_pinn, u_ref)
print(f"  L2相対誤差: {errors['L2_relative_error']:.6e}")
print(f"  最大絶対誤差: {errors['max_absolute_error']:.6e}")
print(f"  平均絶対誤差: {errors['mean_absolute_error']:.6e}")

if errors['L2_relative_error'] < 0.1:
    print(f"\n  ✓✓✓ SUCCESS! PINNと数値解法が良好に一致しています!")
    success = True
elif errors['L2_relative_error'] < 0.3:
    print(f"\n  ⚠ ACCEPTABLE: PINNと数値解法にやや誤差がありますが許容範囲です")
    success = True
else:
    print(f"\n  ✗ FAILED: 誤差が大きすぎます。PINN訓練パラメータの調整が必要です")
    success = False

# 4. 可視化
print("\n" + "="*70)
print("4. 結果の可視化")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 数値解法
ax = axes[0, 0]
im = ax.contourf(x_ref, t_ref, u_ref, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('Numerical Solution (Reference)')
plt.colorbar(im, ax=ax)

# PINN
ax = axes[0, 1]
im = ax.contourf(x_ref, t_ref, u_pinn, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('PINN Solution')
plt.colorbar(im, ax=ax)

# 誤差
ax = axes[0, 2]
error_map = np.abs(u_pinn - u_ref)
im = ax.contourf(x_ref, t_ref, error_map, levels=20, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title(f'Absolute Error (L2={errors["L2_relative_error"]:.4f})')
plt.colorbar(im, ax=ax)

# 時間スナップショット
ax = axes[1, 0]
time_indices = [0, nt_ref//3, 2*nt_ref//3, nt_ref-1]
for idx in time_indices:
    ax.plot(x_ref, u_ref[idx, :], 'o-', label=f't={t_ref[idx]:.2f} (Num)', markersize=3)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Numerical: u(x,t)')
ax.legend()
ax.grid(True)

ax = axes[1, 1]
for idx in time_indices:
    ax.plot(x_ref, u_pinn[idx, :], 's-', label=f't={t_ref[idx]:.2f} (PINN)', markersize=3)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('PINN: u(x,t)')
ax.legend()
ax.grid(True)

# 訓練履歴
ax = axes[1, 2]
ax.semilogy(pinn.loss_history)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('PINN Training History')
ax.grid(True)

plt.tight_layout()
plt.savefig('pinn_vs_numerical.png', dpi=100)
print(f"  ✓ プロットを pinn_vs_numerical.png に保存")

print("\n" + "="*70)
if success:
    print("✓✓✓ 全テスト成功！")
else:
    print("✗ テスト失敗 - PINNの訓練パラメータを調整してください")
print("="*70)
