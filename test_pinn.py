#!/usr/bin/env python3
"""
PINNモデルの基本動作をテスト
"""
import numpy as np
from pinn_burgers import BurgersPINN

print("Testing BurgersPINN...")

# モデルの初期化
pinn = BurgersPINN(
    layers_dims=[2, 20, 20, 1],
    nu=0.01/np.pi,
    lr=0.001
)
print("✓ Model initialized")

# 小さなデータセットを作成
n_bc = 10
n_ic = 10
n_f = 20

# 境界条件データ
x_bc = np.random.uniform(-1, 1, (n_bc, 1))
t_bc = np.random.uniform(0, 1, (n_bc, 1))
u_bc = np.zeros((n_bc, 1))

# 初期条件データ
x_ic = np.random.uniform(-1, 1, (n_ic, 1))
t_ic = np.zeros((n_ic, 1))
u_ic = -np.sin(np.pi * x_ic)

# PDE残差計算用データ
x_f = np.random.uniform(-1, 1, (n_f, 1))
t_f = np.random.uniform(0, 1, (n_f, 1))

print("✓ Training data created")

# 短時間訓練
pinn.fit(
    x_bc, t_bc, u_bc,
    x_ic, t_ic, u_ic,
    x_f, t_f,
    epochs=10,
    print_every=5
)
print("✓ Training completed")

# 予測テスト
x_test = np.array([[0.0]])
t_test = np.array([[0.5]])
u_pred = pinn.predict(x_test, t_test)
print(f"✓ Prediction test: u(0, 0.5) = {u_pred[0, 0]:.6f}")

print("\n✅ All tests passed!")
