"""
Burgers方程式を解くPINNアプリケーション（Streamlit）
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pinn_burgers import BurgersPINN, generate_training_data
from numerical_solver import solve_burgers_fdm, compute_error_metrics
import time

# ページ設定
st.set_page_config(page_title="PINN Burgers Equation Solver", layout="wide")

st.title("🌊 PINN による Burgers方程式ソルバー")
st.write("Physics-Informed Neural Networks (PINN) でBurgers方程式を解きます")

# Burgers方程式の説明
with st.expander("📚 Burgers方程式について"):
    st.latex(r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}")
    st.write("""
    - **u**: 速度場
    - **t**: 時間
    - **x**: 空間座標
    - **ν**: 粘性係数

    **初期条件**: $u(x, 0) = -\sin(\pi x)$

    **境界条件**: $u(-1, t) = u(1, t) = 0$
    """)

# サイドバーでパラメータ設定
st.sidebar.header("⚙️ パラメータ設定")

# 物理パラメータ
nu = st.sidebar.number_input("粘性係数 ν", min_value=0.001, max_value=0.1,
                             value=0.01/np.pi, format="%.6f")

# 領域設定
x_min = st.sidebar.number_input("x の最小値", value=-1.0)
x_max = st.sidebar.number_input("x の最大値", value=1.0)
t_min = st.sidebar.number_input("t の最小値", value=0.0)
t_max = st.sidebar.number_input("t の最大値", value=1.0)

# PINN設定
st.sidebar.subheader("PINN設定")
epochs = st.sidebar.slider("エポック数", 100, 20000, 5000, 100)
learning_rate = st.sidebar.select_slider(
    "学習率",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001
)
n_collocation = st.sidebar.slider("コロケーションポイント数", 1000, 20000, 10000, 1000)

# 数値解法設定
st.sidebar.subheader("数値解法設定")
nx_fdm = st.sidebar.slider("空間グリッド数", 50, 500, 256, 10)
nt_fdm = st.sidebar.slider("時間グリッド数", 50, 500, 100, 10)

# 訓練開始ボタン
if st.sidebar.button("🚀 訓練開始", type="primary"):
    st.session_state.training = True

# メインエリア
if 'training' not in st.session_state:
    st.info("👈 左のサイドバーでパラメータを設定し、「訓練開始」ボタンを押してください")
else:
    # タブの作成
    tab1, tab2, tab3, tab4 = st.tabs(["📊 結果比較", "🧠 PINN結果", "🔢 数値解法結果", "📈 誤差分析"])

    with st.spinner("モデルを訓練中..."):
        # データ生成
        st.write("### データ生成")
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("訓練データを生成中...")
        progress_bar.progress(10)

        (x_bc, t_bc, u_bc), (x_ic, t_ic, u_ic), (x_f, t_f) = generate_training_data(
            x_range=[x_min, x_max],
            t_range=[t_min, t_max],
            n_bc=100,
            n_ic=256,
            n_f=n_collocation
        )

        # PINNモデルの訓練
        status_text.text("PINNモデルを訓練中...")
        progress_bar.progress(20)

        pinn = BurgersPINN(
            layers_dims=[2, 50, 50, 50, 1],
            nu=nu,
            lr=learning_rate
        )

        # 訓練
        loss_container = st.empty()

        for epoch in range(0, epochs, max(1, epochs // 10)):
            remaining_epochs = min(max(1, epochs // 10), epochs - epoch)
            pinn.train(
                x_bc, t_bc, u_bc, x_ic, t_ic, u_ic, x_f, t_f,
                epochs=remaining_epochs,
                print_every=max(1, remaining_epochs // 2)
            )
            progress = 20 + int((epoch / epochs) * 50)
            progress_bar.progress(progress)

            if len(pinn.loss_history) > 0:
                loss_container.metric("現在の損失", f"{pinn.loss_history[-1]:.6f}")

        status_text.text("PINNモデルの訓練完了")
        progress_bar.progress(70)

        # 数値解法で解く
        status_text.text("数値解法で計算中...")
        progress_bar.progress(75)

        x_fdm, t_fdm, u_fdm = solve_burgers_fdm(
            x_range=[x_min, x_max],
            t_range=[t_min, t_max],
            nx=nx_fdm,
            nt=nt_fdm,
            nu=nu,
            method='implicit'
        )

        progress_bar.progress(90)

        # PINNで予測
        status_text.text("PINNで予測中...")
        X_grid, T_grid = np.meshgrid(x_fdm, t_fdm)
        x_flat = X_grid.flatten()[:, None].astype(np.float32)
        t_flat = T_grid.flatten()[:, None].astype(np.float32)

        u_pinn_flat = pinn.predict(x_flat, t_flat)
        u_pinn = u_pinn_flat.reshape(X_grid.shape)

        progress_bar.progress(100)
        status_text.text("完了！")

        # 誤差計算
        errors = compute_error_metrics(u_pinn, u_fdm)

    # 結果の表示
    with tab1:
        st.header("結果比較")

        col1, col2, col3 = st.columns(3)
        col1.metric("L2相対誤差", f"{errors['L2_relative_error']:.6f}")
        col2.metric("最大絶対誤差", f"{errors['max_absolute_error']:.6f}")
        col3.metric("平均絶対誤差", f"{errors['mean_absolute_error']:.6f}")

        # 特定の時刻での比較
        st.subheader("特定の時刻での比較")
        time_idx = st.slider("時刻インデックス", 0, len(t_fdm)-1, len(t_fdm)//2)
        selected_time = t_fdm[time_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_fdm, u_fdm[time_idx, :], 'b-', label='数値解法 (FDM)', linewidth=2)
        ax.plot(x_fdm, u_pinn[time_idx, :], 'r--', label='PINN', linewidth=2)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u', fontsize=12)
        ax.set_title(f't = {selected_time:.3f} での比較', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # ヒートマップ比較
        st.subheader("時空間での解の分布")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**PINN**")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            im1 = ax1.contourf(X_grid, T_grid, u_pinn, levels=50, cmap='viridis')
            ax1.set_xlabel('x')
            ax1.set_ylabel('t')
            ax1.set_title('PINN Solution')
            plt.colorbar(im1, ax=ax1)
            st.pyplot(fig1)

        with col2:
            st.write("**数値解法 (FDM)**")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            im2 = ax2.contourf(X_grid, T_grid, u_fdm, levels=50, cmap='viridis')
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            ax2.set_title('FDM Solution')
            plt.colorbar(im2, ax=ax2)
            st.pyplot(fig2)

        with col3:
            st.write("**誤差分布**")
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            error_map = np.abs(u_pinn - u_fdm)
            im3 = ax3.contourf(X_grid, T_grid, error_map, levels=50, cmap='hot')
            ax3.set_xlabel('x')
            ax3.set_ylabel('t')
            ax3.set_title('Absolute Error')
            plt.colorbar(im3, ax=ax3)
            st.pyplot(fig3)

    with tab2:
        st.header("PINN結果")

        # 損失の履歴
        st.subheader("訓練の損失履歴")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pinn.loss_history)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss History')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # PINN解のヒートマップ
        st.subheader("PINN解の時空間分布")
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.contourf(X_grid, T_grid, u_pinn, levels=50, cmap='viridis')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('t', fontsize=12)
        ax.set_title('PINN Solution u(x,t)', fontsize=14)
        plt.colorbar(im, ax=ax, label='u')
        st.pyplot(fig)

    with tab3:
        st.header("数値解法 (有限差分法) 結果")

        # FDM解のヒートマップ
        st.subheader("FDM解の時空間分布")
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.contourf(X_grid, T_grid, u_fdm, levels=50, cmap='viridis')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('t', fontsize=12)
        ax.set_title('FDM Solution u(x,t)', fontsize=14)
        plt.colorbar(im, ax=ax, label='u')
        st.pyplot(fig)

        # 複数の時刻でのスナップショット
        st.subheader("時刻ごとのスナップショット")
        time_snapshots = [0, len(t_fdm)//4, len(t_fdm)//2, 3*len(t_fdm)//4, len(t_fdm)-1]

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, idx in enumerate(time_snapshots):
            axes[i].plot(x_fdm, u_fdm[idx, :], 'b-', linewidth=2)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('u')
            axes[i].set_title(f't = {t_fdm[idx]:.3f}')
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.header("誤差分析")

        # 誤差統計
        st.subheader("誤差統計")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("L2相対誤差", f"{errors['L2_relative_error']:.6f}")
            st.metric("最大絶対誤差", f"{errors['max_absolute_error']:.6f}")
            st.metric("平均絶対誤差", f"{errors['mean_absolute_error']:.6f}")

        with col2:
            error_std = np.std(np.abs(u_pinn - u_fdm))
            st.metric("誤差の標準偏差", f"{error_std:.6f}")
            st.metric("相対誤差 (%)", f"{errors['L2_relative_error']*100:.2f}%")

        # 誤差のヒストグラム
        st.subheader("誤差の分布")
        fig, ax = plt.subplots(figsize=(10, 4))
        error_flat = (u_pinn - u_fdm).flatten()
        ax.hist(error_flat, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Error (u_PINN - u_FDM)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 時間ごとの誤差推移
        st.subheader("時間ごとの誤差推移")
        time_errors = [np.linalg.norm(u_pinn[i, :] - u_fdm[i, :]) for i in range(len(t_fdm))]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_fdm, time_errors, 'r-', linewidth=2)
        ax.set_xlabel('時間 t')
        ax.set_ylabel('L2ノルム誤差')
        ax.set_title('時間発展に伴う誤差')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# フッター
st.markdown("---")
st.markdown("**Physics-Informed Neural Networks (PINN)** で偏微分方程式を解くデモアプリケーション")
