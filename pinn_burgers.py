"""
PINN (Physics-Informed Neural Network) for Burgers Equation

Burgers方程式: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


class BurgersPINN:
    """
    Physics-Informed Neural Network for Burgers Equation
    """

    def __init__(self, layers_dims=[2, 50, 50, 50, 1], nu=0.01/np.pi, lr=0.001):
        """
        Args:
            layers_dims: ニューラルネットワークの層の次元 [入力, 隠れ層..., 出力]
            nu: 粘性係数
            lr: 学習率
        """
        self.nu = nu
        self.layers_dims = layers_dims
        self.model = self._build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_history = []

    def _build_model(self):
        """ニューラルネットワークモデルを構築"""
        inputs = keras.Input(shape=(2,))  # [x, t]
        x = inputs

        # 隠れ層
        for units in self.layers_dims[1:-1]:
            x = layers.Dense(units, activation='tanh',
                           kernel_initializer='glorot_normal')(x)

        # 出力層
        outputs = layers.Dense(self.layers_dims[-1],
                             kernel_initializer='glorot_normal')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def predict_u(self, x, t):
        """u(x,t)を予測"""
        xt = tf.concat([x, t], axis=1)
        return self.model(xt)

    def compute_residual(self, x, t):
        """
        物理法則の残差を計算
        Burgers方程式: ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, t])
                u = self.predict_u(x, t)

            u_x = tape1.gradient(u, x)
            u_t = tape1.gradient(u, t)

        u_xx = tape2.gradient(u_x, x)

        del tape1, tape2

        # Burgers方程式の残差
        residual = u_t + u * u_x - self.nu * u_xx
        return residual

    @tf.function
    def compute_loss(self, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f):
        """
        損失関数を計算
        Args:
            x_bc, t_bc, u_bc: 境界条件のデータ
            x_ic, t_ic, u_ic: 初期条件のデータ
            x_f, t_f: コロケーションポイント（物理法則を満たすべき点）
        """
        # 境界条件の損失
        u_bc_pred = self.predict_u(x_bc, t_bc)
        loss_bc = tf.reduce_mean(tf.square(u_bc_pred - u_bc))

        # 初期条件の損失
        u_ic_pred = self.predict_u(x_ic, t_ic)
        loss_ic = tf.reduce_mean(tf.square(u_ic_pred - u_ic))

        # 物理法則（PDE）の損失
        residual = self.compute_residual(x_f, t_f)
        loss_pde = tf.reduce_mean(tf.square(residual))

        # 総損失
        total_loss = loss_bc + loss_ic + loss_pde

        return total_loss, loss_bc, loss_ic, loss_pde

    @tf.function
    def train_step(self, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f):
        """1ステップの訓練"""
        with tf.GradientTape() as tape:
            loss, loss_bc, loss_ic, loss_pde = self.compute_loss(
                x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, loss_bc, loss_ic, loss_pde

    def train(self, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f,
              epochs=10000, print_every=1000):
        """
        モデルを訓練
        """
        print("Training PINN model...")
        start_time = time.time()

        for epoch in range(epochs):
            loss, loss_bc, loss_ic, loss_pde = self.train_step(
                x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f
            )

            self.loss_history.append(loss.numpy())

            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.numpy():.6f}, "
                      f"BC: {loss_bc.numpy():.6f}, IC: {loss_ic.numpy():.6f}, "
                      f"PDE: {loss_pde.numpy():.6f}, Time: {elapsed:.2f}s")

        print(f"Training completed in {time.time() - start_time:.2f}s")

    def predict(self, x, t):
        """予測を実行（NumPy配列で）"""
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
        u_pred = self.predict_u(x_tf, t_tf)
        return u_pred.numpy()


def generate_training_data(x_range, t_range, n_bc=100, n_ic=256, n_f=10000):
    """
    訓練データを生成

    Args:
        x_range: [x_min, x_max]
        t_range: [t_min, t_max]
        n_bc: 境界条件のサンプル数
        n_ic: 初期条件のサンプル数
        n_f: コロケーションポイント数

    Returns:
        境界条件、初期条件、コロケーションポイントのデータ
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

    # TensorFlowテンソルに変換
    x_bc = tf.convert_to_tensor(x_bc)
    t_bc = tf.convert_to_tensor(t_bc)
    u_bc = tf.convert_to_tensor(u_bc)

    x_ic = tf.convert_to_tensor(x_ic)
    t_ic = tf.convert_to_tensor(t_ic)
    u_ic = tf.convert_to_tensor(u_ic)

    x_f = tf.convert_to_tensor(x_f)
    t_f = tf.convert_to_tensor(t_f)

    return (x_bc, t_bc, u_bc), (x_ic, t_ic, u_ic), (x_f, t_f)


def burgers_analytical(x, t, nu=0.01/np.pi):
    """
    Burgers方程式の解析解（存在する場合）
    初期条件: u(x,0) = -sin(πx)
    境界条件: u(-1,t) = u(1,t) = 0

    Note: 完全な解析解は複雑なので、ここでは数値解を返します
    """
    # この問題の正確な解析解は複雑なので、数値解法を別途使用します
    pass
