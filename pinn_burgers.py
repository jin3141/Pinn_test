"""
PINN (Physics-Informed Neural Network) for Burgers Equation - PyTorch版

Burgers方程式: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
"""

import numpy as np
import torch
import torch.nn as nn
import time


class BurgersPINN(nn.Module):
    """
    Physics-Informed Neural Network for Burgers Equation (PyTorch実装)
    """

    def __init__(self, layers_dims=[2, 50, 50, 50, 1], nu=0.01/np.pi, lr=0.001):
        """
        Args:
            layers_dims: ニューラルネットワークの層の次元 [入力, 隠れ層..., 出力]
            nu: 粘性係数
            lr: 学習率
        """
        super(BurgersPINN, self).__init__()

        self.nu = nu
        self.layers_dims = layers_dims

        # ニューラルネットワークの構築
        layers = []
        for i in range(len(layers_dims) - 1):
            layers.append(nn.Linear(layers_dims[i], layers_dims[i+1]))
            if i < len(layers_dims) - 2:  # 最終層以外は活性化関数を追加
                layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

        # パラメータの初期化（Xavier/Glorot）
        self._initialize_weights()

        # オプティマイザ
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # 損失履歴
        self.loss_history = []

        # デバイス設定（CPUを使用）
        self.device = torch.device('cpu')
        self.to(self.device)

    def _initialize_weights(self):
        """重みの初期化（Xavier/Glorot）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """
        順伝播
        Args:
            x: 空間座標 [N, 1]
            t: 時間座標 [N, 1]
        Returns:
            u: 解 [N, 1]
        """
        xt = torch.cat([x, t], dim=1)
        return self.network(xt)

    def compute_pde_residual(self, x, t):
        """
        物理法則の残差を計算
        Burgers方程式: ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0

        Args:
            x: 空間座標 [N, 1]
            t: 時間座標 [N, 1]
        Returns:
            residual: PDE残差 [N, 1]
        """
        # 勾配計算を有効化
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        # u(x,t)を計算
        u = self.forward(x, t)

        # 1階微分: ∂u/∂x, ∂u/∂t
        u_x = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        u_t = torch.autograd.grad(
            outputs=u, inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        # 2階微分: ∂²u/∂x²
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]

        # Burgers方程式の残差
        residual = u_t + u * u_x - self.nu * u_xx

        return residual

    def compute_loss(self, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f):
        """
        損失関数を計算

        Args:
            x_bc, t_bc, u_bc: 境界条件のデータ
            x_ic, t_ic, u_ic: 初期条件のデータ
            x_f, t_f: コロケーションポイント（物理法則を満たすべき点）
        Returns:
            total_loss, loss_bc, loss_ic, loss_pde
        """
        # 境界条件の損失
        u_bc_pred = self.forward(x_bc, t_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)

        # 初期条件の損失
        u_ic_pred = self.forward(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)

        # 物理法則（PDE）の損失
        residual = self.compute_pde_residual(x_f, t_f)
        loss_pde = torch.mean(residual ** 2)

        # 総損失
        total_loss = loss_bc + loss_ic + loss_pde

        return total_loss, loss_bc, loss_ic, loss_pde

    def train_step(self, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f):
        """1ステップの訓練"""
        self.optimizer.zero_grad()

        loss, loss_bc, loss_ic, loss_pde = self.compute_loss(
            x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f
        )

        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_bc.item(), loss_ic.item(), loss_pde.item()

    def train(self, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f,
              epochs=10000, print_every=1000):
        """
        モデルを訓練
        """
        print("Training PINN model...")
        start_time = time.time()

        self.train_mode = True

        for epoch in range(epochs):
            loss, loss_bc, loss_ic, loss_pde = self.train_step(
                x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f
            )

            self.loss_history.append(loss)

            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}, "
                      f"BC: {loss_bc:.6f}, IC: {loss_ic:.6f}, "
                      f"PDE: {loss_pde:.6f}, Time: {elapsed:.2f}s")

        print(f"Training completed in {time.time() - start_time:.2f}s")

        self.eval()

    def predict(self, x, t):
        """
        予測を実行（NumPy配列で入出力）

        Args:
            x: 空間座標 [N, 1] (numpy array)
            t: 時間座標 [N, 1] (numpy array)
        Returns:
            u: 解 [N, 1] (numpy array)
        """
        self.eval()

        with torch.no_grad():
            # NumPy → Tensor
            x_tensor = torch.from_numpy(x).float().to(self.device)
            t_tensor = torch.from_numpy(t).float().to(self.device)

            # 予測
            u_tensor = self.forward(x_tensor, t_tensor)

            # Tensor → NumPy
            u_pred = u_tensor.cpu().numpy()

        return u_pred


def generate_training_data(x_range, t_range, n_bc=100, n_ic=256, n_f=10000):
    """
    訓練データを生成（PyTorchテンソルとして）

    Args:
        x_range: [x_min, x_max]
        t_range: [t_min, t_max]
        n_bc: 境界条件のサンプル数
        n_ic: 初期条件のサンプル数
        n_f: コロケーションポイント数

    Returns:
        境界条件、初期条件、コロケーションポイントのデータ（PyTorchテンソル）
    """
    x_min, x_max = x_range
    t_min, t_max = t_range

    # 境界条件: x = x_min と x = x_max で u = 0
    t_bc = np.random.uniform(t_min, t_max, (n_bc, 1)).astype(np.float32)
    x_bc_left = np.full((n_bc // 2, 1), x_min, dtype=np.float32)
    x_bc_right = np.full((n_bc // 2, 1), x_max, dtype=np.float32)
    x_bc = np.vstack([x_bc_left, x_bc_right])
    t_bc = np.vstack([t_bc[:n_bc//2], t_bc[n_bc//2:]])
    u_bc = np.zeros((n_bc, 1), dtype=np.float32)

    # 初期条件: t = 0 で u = -sin(πx)
    x_ic = np.random.uniform(x_min, x_max, (n_ic, 1)).astype(np.float32)
    t_ic = np.zeros((n_ic, 1), dtype=np.float32)
    u_ic = -np.sin(np.pi * x_ic).astype(np.float32)

    # コロケーションポイント（物理法則を満たすべき点）
    x_f = np.random.uniform(x_min, x_max, (n_f, 1)).astype(np.float32)
    t_f = np.random.uniform(t_min, t_max, (n_f, 1)).astype(np.float32)

    # NumPy → PyTorchテンソルに変換
    x_bc = torch.from_numpy(x_bc).float()
    t_bc = torch.from_numpy(t_bc).float()
    u_bc = torch.from_numpy(u_bc).float()

    x_ic = torch.from_numpy(x_ic).float()
    t_ic = torch.from_numpy(t_ic).float()
    u_ic = torch.from_numpy(u_ic).float()

    x_f = torch.from_numpy(x_f).float()
    t_f = torch.from_numpy(t_f).float()

    return (x_bc, t_bc, u_bc), (x_ic, t_ic, u_ic), (x_f, t_f)
