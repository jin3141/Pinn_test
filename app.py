"""
Burgers方程式を解くPINNアプリケーション（Streamlit）
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pinn_burgers import BurgersPINN, generate_training_data
from numerical_solver import solve_burgers_fdm, solve_burgers_analytical, compute_error_metrics
import time

# ページ設定
st.set_page_config(
    page_title="PINN Burgers Equation Solver",
    layout="centered",  # モバイル対応のため centered に変更
    initial_sidebar_state="auto"
)

# モバイル対応のカスタムCSS
st.markdown("""
<style>
    /* モバイル対応 */
    @media (max-width: 768px) {
        /* メインコンテンツエリア */
        .main .block-container {
            padding-top: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
            padding-bottom: 3rem;
            max-width: 100%;
        }

        /* サイドバーのスクロール改善 */
        section[data-testid="stSidebar"] {
            z-index: 999;
        }

        section[data-testid="stSidebar"] > div {
            overflow-y: auto;
            -webkit-overflow-scrolling: touch;
            max-height: 100vh;
        }

        /* ボタンのサイズを大きく（タップしやすく） */
        .stButton > button {
            width: 100%;
            min-height: 3.5rem;
            font-size: 1.3rem;
            padding: 1rem;
        }

        /* タブのサイズ調整 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
            padding: 0.5rem 0.8rem;
            white-space: nowrap;
        }

        /* 画像とプロットのサイズ調整 */
        img {
            max-width: 100%;
            height: auto;
        }

        /* テキストのサイズ調整 */
        h1 {
            font-size: 1.8rem;
        }

        h2 {
            font-size: 1.5rem;
        }

        h3 {
            font-size: 1.2rem;
        }
    }

    /* タブレット対応 */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            max-width: 95%;
        }
    }

    /* 全デバイス共通 */
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #FF6B6B;
        transform: scale(1.02);
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    /* スクロールバーのスタイル */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 PINN Burgers方程式")
st.write("Physics-Informed Neural Networks (PINN) でBurgers方程式を解きます")

# モバイルユーザー向けヒント
st.info("💡 **スマホの方**: サイドバーを開いてパラメータを調整できます（画面左上の「>」をタップ）")

# Burgers方程式の説明
with st.expander("📚 Burgers方程式について", expanded=False):
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
st.sidebar.info("💡 スマホの方: パラメータ変更後、メインページに戻って「訓練開始」ボタンを押してください")

# クイック設定
with st.sidebar.expander("⚡ クイック設定", expanded=True):
    quick_mode = st.radio(
        "計算モード",
        ["🚀 高速（テスト用）", "⚖️ 標準", "🎯 高精度"],
        index=1,
        help="計算速度と精度のバランスを選択"
    )

# クイック設定に応じたパラメータ
if quick_mode == "🚀 高速（テスト用）":
    default_epochs = 1000
    default_nx = 100
    default_nt = 50
    default_collocation = 5000
elif quick_mode == "🎯 高精度":
    default_epochs = 10000
    default_nx = 300
    default_nt = 150
    default_collocation = 15000
else:  # 標準
    default_epochs = 5000
    default_nx = 200
    default_nt = 100
    default_collocation = 10000

# 物理パラメータ
with st.sidebar.expander("🔬 物理パラメータ", expanded=False):
    nu = st.number_input("粘性係数 ν", min_value=0.001, max_value=0.1,
                         value=0.01/np.pi, format="%.6f")
    st.markdown("---")
    x_min = st.number_input("x の最小値", value=-1.0)
    x_max = st.number_input("x の最大値", value=1.0)
    t_min = st.number_input("t の最小値", value=0.0)
    t_max = st.number_input("t の最大値", value=1.0)

# PINN設定
with st.sidebar.expander("🧠 PINN設定", expanded=False):
    epochs = st.slider("エポック数", 100, 20000, default_epochs, 100)
    learning_rate = st.select_slider(
        "学習率",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001
    )
    n_collocation = st.slider("コロケーションポイント数", 1000, 20000, default_collocation, 1000)

# 数値解法設定
with st.sidebar.expander("🔢 数値解法設定", expanded=False):
    nx_fdm = st.slider("空間グリッド数", 50, 500, default_nx, 10)
    nt_fdm = st.slider("時間グリッド数", 50, 500, default_nt, 10)

# 解析解設定
with st.sidebar.expander("✨ 解析解設定", expanded=False):
    compute_analytical = st.checkbox("解析解を計算する", value=True)
    if compute_analytical:
        nx_analytical = st.slider("解析解の空間グリッド数", 30, 200, 100, 10)
        nt_analytical = st.slider("解析解の時間グリッド数", 20, 100, 50, 10)
    else:
        nx_analytical = 100
        nt_analytical = 50

# メインエリアの訓練開始ボタン（モバイル対応）
st.markdown("---")
st.markdown("### 🚀 計算を開始")
st.write("下のボタンをクリックして、PINNの訓練と解の計算を開始します。")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    main_start_button = st.button(
        "🚀 訓練開始",
        type="primary",
        key="main_button",
        use_container_width=True,
        help="PINNの訓練と数値解法・解析解の計算を開始します"
    )

# サイドバーの訓練開始ボタン（デスクトップ用）
sidebar_start_button = st.sidebar.button("🚀 訓練開始", type="primary", key="sidebar_button")

# どちらかのボタンが押されたら訓練開始
if main_start_button or sidebar_start_button:
    st.session_state.training = True

st.markdown("---")

# メインエリア
if 'training' not in st.session_state:
    st.info("⬆️ 上の「訓練開始」ボタンを押すか、左のサイドバーでパラメータを調整してください")
else:
    # タブの作成
    if compute_analytical:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 3手法比較", "✨ 解析解", "🧠 PINN結果", "🔢 数値解法結果", "📈 誤差分析", "📉 詳細比較"
        ])
    else:
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
            pinn.fit(
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

        progress_bar.progress(80)

        # 解析解を計算
        u_analytical = None
        x_analytical = None
        t_analytical = None
        if compute_analytical:
            status_text.text("解析解を計算中（Cole-Hopf変換）...")
            x_analytical, t_analytical, u_analytical = solve_burgers_analytical(
                x_range=[x_min, x_max],
                t_range=[t_min, t_max],
                nx=nx_analytical,
                nt=nt_analytical,
                nu=nu
            )
            progress_bar.progress(85)

        # PINNで予測
        status_text.text("PINNで予測中...")
        X_grid, T_grid = np.meshgrid(x_fdm, t_fdm)
        x_flat = X_grid.flatten()[:, None].astype(np.float32)
        t_flat = T_grid.flatten()[:, None].astype(np.float32)

        u_pinn_flat = pinn.predict(x_flat, t_flat)
        u_pinn = u_pinn_flat.reshape(X_grid.shape)

        progress_bar.progress(95)

        # 誤差計算
        errors_vs_fdm = compute_error_metrics(u_pinn, u_fdm)

        # 解析解との誤差計算（同じグリッドで比較）
        errors_pinn_vs_analytical = None
        errors_fdm_vs_analytical = None
        if compute_analytical:
            # 解析解のグリッドでPINNとFDMを評価
            X_ana, T_ana = np.meshgrid(x_analytical, t_analytical)
            x_ana_flat = X_ana.flatten()[:, None].astype(np.float32)
            t_ana_flat = T_ana.flatten()[:, None].astype(np.float32)

            u_pinn_ana = pinn.predict(x_ana_flat, t_ana_flat).reshape(X_ana.shape)

            # FDMの結果を解析解のグリッドに補間
            from scipy.interpolate import RectBivariateSpline
            fdm_interp = RectBivariateSpline(t_fdm, x_fdm, u_fdm)
            u_fdm_ana = fdm_interp(t_analytical, x_analytical)

            errors_pinn_vs_analytical = compute_error_metrics(u_pinn_ana, u_analytical)
            errors_fdm_vs_analytical = compute_error_metrics(u_fdm_ana, u_analytical)

        progress_bar.progress(100)
        status_text.text("完了！")

    # 結果の表示
    with tab1:
        if compute_analytical:
            st.header("3手法の比較（PINN vs FDM vs 解析解）")

            # 解析解との誤差比較
            st.subheader("📊 解析解との誤差比較")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**PINN vs 解析解**")
                st.metric("L2相対誤差", f"{errors_pinn_vs_analytical['L2_relative_error']:.6f}")
                st.metric("最大絶対誤差", f"{errors_pinn_vs_analytical['max_absolute_error']:.6f}")
                st.metric("平均絶対誤差", f"{errors_pinn_vs_analytical['mean_absolute_error']:.6f}")

            with col2:
                st.write("**FDM vs 解析解**")
                st.metric("L2相対誤差", f"{errors_fdm_vs_analytical['L2_relative_error']:.6f}")
                st.metric("最大絶対誤差", f"{errors_fdm_vs_analytical['max_absolute_error']:.6f}")
                st.metric("平均絶対誤差", f"{errors_fdm_vs_analytical['mean_absolute_error']:.6f}")

            # 特定の時刻での3手法比較
            st.subheader("特定の時刻での3手法比較")
            time_idx_ana = st.slider("時刻インデックス", 0, len(t_analytical)-1, len(t_analytical)//2, key="time_idx_ana")
            selected_time_ana = t_analytical[time_idx_ana]

            # 対応するFDMの時刻インデックスを見つける
            fdm_time_idx = np.argmin(np.abs(t_fdm - selected_time_ana))

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(x_analytical, u_analytical[time_idx_ana, :], 'k-', label='解析解', linewidth=2.5, alpha=0.8)
            ax.plot(x_fdm, u_fdm[fdm_time_idx, :], 'b--', label='数値解法 (FDM)', linewidth=2)
            ax.plot(x_analytical, u_pinn_ana[time_idx_ana, :], 'r:', label='PINN', linewidth=2)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('u', fontsize=12)
            ax.set_title(f't = {selected_time_ana:.3f} での3手法比較', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # ヒートマップ比較
            st.subheader("時空間での解の分布")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**解析解**")
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                X_ana, T_ana = np.meshgrid(x_analytical, t_analytical)
                im1 = ax1.contourf(X_ana, T_ana, u_analytical, levels=50, cmap='viridis')
                ax1.set_xlabel('x')
                ax1.set_ylabel('t')
                ax1.set_title('Analytical Solution')
                plt.colorbar(im1, ax=ax1)
                st.pyplot(fig1)

            with col2:
                st.write("**PINN**")
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                im2 = ax2.contourf(X_ana, T_ana, u_pinn_ana, levels=50, cmap='viridis')
                ax2.set_xlabel('x')
                ax2.set_ylabel('t')
                ax2.set_title('PINN Solution')
                plt.colorbar(im2, ax=ax2)
                st.pyplot(fig2)

            with col3:
                st.write("**FDM**")
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                im3 = ax3.contourf(X_ana, T_ana, u_fdm_ana, levels=50, cmap='viridis')
                ax3.set_xlabel('x')
                ax3.set_ylabel('t')
                ax3.set_title('FDM Solution')
                plt.colorbar(im3, ax=ax3)
                st.pyplot(fig3)

        else:
            st.header("結果比較（PINN vs FDM）")

            col1, col2, col3 = st.columns(3)
            col1.metric("L2相対誤差", f"{errors_vs_fdm['L2_relative_error']:.6f}")
            col2.metric("最大絶対誤差", f"{errors_vs_fdm['max_absolute_error']:.6f}")
            col3.metric("平均絶対誤差", f"{errors_vs_fdm['mean_absolute_error']:.6f}")

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
                ax3.set_title('Absolute Error (PINN - FDM)')
                plt.colorbar(im3, ax=ax3)
                st.pyplot(fig3)

    # 解析解タブ（解析解を計算する場合のみ）
    if compute_analytical:
        with tab2:
            st.header("✨ 解析解（Cole-Hopf変換）")

            st.write("""
            **Cole-Hopf変換**を用いてBurgers方程式を熱方程式に変換し、解析的に解いています。

            変換: $u = -2\\nu \\frac{\\partial \\phi}{\\partial x} / \\phi$

            これによりBurgers方程式は熱方程式 $\\frac{\\partial \\phi}{\\partial t} = \\nu \\frac{\\partial^2 \\phi}{\\partial x^2}$ に変換されます。
            """)

            # 解析解のヒートマップ
            st.subheader("解析解の時空間分布")
            fig, ax = plt.subplots(figsize=(10, 6))
            X_ana, T_ana = np.meshgrid(x_analytical, t_analytical)
            im = ax.contourf(X_ana, T_ana, u_analytical, levels=50, cmap='viridis')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('t', fontsize=12)
            ax.set_title('Analytical Solution u(x,t)', fontsize=14)
            plt.colorbar(im, ax=ax, label='u')
            st.pyplot(fig)

            # 複数の時刻でのスナップショット
            st.subheader("時刻ごとのスナップショット（解析解）")
            time_snapshots = [0, len(t_analytical)//4, len(t_analytical)//2, 3*len(t_analytical)//4, len(t_analytical)-1]

            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i, idx in enumerate(time_snapshots):
                axes[i].plot(x_analytical, u_analytical[idx, :], 'k-', linewidth=2)
                axes[i].set_xlabel('x')
                axes[i].set_ylabel('u')
                axes[i].set_title(f't = {t_analytical[idx]:.3f}')
                axes[i].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    with (tab3 if compute_analytical else tab2):
        st.header("🧠 PINN結果")

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

    with (tab4 if compute_analytical else tab3):
        st.header("🔢 数値解法 (有限差分法) 結果")

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

    with (tab5 if compute_analytical else tab4):
        st.header("📈 誤差分析")

        if compute_analytical:
            st.subheader("解析解との誤差（最も信頼できる比較）")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**PINN vs 解析解**")
                st.metric("L2相対誤差", f"{errors_pinn_vs_analytical['L2_relative_error']:.6f}")
                st.metric("最大絶対誤差", f"{errors_pinn_vs_analytical['max_absolute_error']:.6f}")
                st.metric("平均絶対誤差", f"{errors_pinn_vs_analytical['mean_absolute_error']:.6f}")

            with col2:
                st.write("**FDM vs 解析解**")
                st.metric("L2相対誤差", f"{errors_fdm_vs_analytical['L2_relative_error']:.6f}")
                st.metric("最大絶対誤差", f"{errors_fdm_vs_analytical['max_absolute_error']:.6f}")
                st.metric("平均絶対誤差", f"{errors_fdm_vs_analytical['mean_absolute_error']:.6f}")

            # 誤差のヒートマップ
            st.subheader("誤差分布（vs 解析解）")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**PINN - 解析解**")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                error_pinn = np.abs(u_pinn_ana - u_analytical)
                im1 = ax1.contourf(X_ana, T_ana, error_pinn, levels=50, cmap='hot')
                ax1.set_xlabel('x')
                ax1.set_ylabel('t')
                ax1.set_title('|PINN - Analytical|')
                plt.colorbar(im1, ax=ax1)
                st.pyplot(fig1)

            with col2:
                st.write("**FDM - 解析解**")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                error_fdm = np.abs(u_fdm_ana - u_analytical)
                im2 = ax2.contourf(X_ana, T_ana, error_fdm, levels=50, cmap='hot')
                ax2.set_xlabel('x')
                ax2.set_ylabel('t')
                ax2.set_title('|FDM - Analytical|')
                plt.colorbar(im2, ax=ax2)
                st.pyplot(fig2)

        else:
            # 誤差統計（PINN vs FDM）
            st.subheader("誤差統計（PINN vs FDM）")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("L2相対誤差", f"{errors_vs_fdm['L2_relative_error']:.6f}")
                st.metric("最大絶対誤差", f"{errors_vs_fdm['max_absolute_error']:.6f}")
                st.metric("平均絶対誤差", f"{errors_vs_fdm['mean_absolute_error']:.6f}")

            with col2:
                error_std = np.std(np.abs(u_pinn - u_fdm))
                st.metric("誤差の標準偏差", f"{error_std:.6f}")
                st.metric("相対誤差 (%)", f"{errors_vs_fdm['L2_relative_error']*100:.2f}%")

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

    # 詳細比較タブ（解析解を計算する場合のみ）
    if compute_analytical:
        with tab6:
            st.header("📉 詳細比較")

            st.subheader("時間ごとの誤差推移（vs 解析解）")

            # PINNとFDMの誤差を時間ごとに計算
            time_errors_pinn = [np.linalg.norm(u_pinn_ana[i, :] - u_analytical[i, :]) for i in range(len(t_analytical))]
            time_errors_fdm = [np.linalg.norm(u_fdm_ana[i, :] - u_analytical[i, :]) for i in range(len(t_analytical))]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t_analytical, time_errors_pinn, 'r-', linewidth=2, label='PINN vs 解析解')
            ax.plot(t_analytical, time_errors_fdm, 'b-', linewidth=2, label='FDM vs 解析解')
            ax.set_xlabel('時間 t')
            ax.set_ylabel('L2ノルム誤差')
            ax.set_title('時間発展に伴う誤差の比較')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # 誤差の比較表
            st.subheader("誤差の比較表")

            import pandas as pd

            comparison_df = pd.DataFrame({
                '手法': ['PINN', 'FDM'],
                'L2相対誤差': [
                    f"{errors_pinn_vs_analytical['L2_relative_error']:.6f}",
                    f"{errors_fdm_vs_analytical['L2_relative_error']:.6f}"
                ],
                '最大絶対誤差': [
                    f"{errors_pinn_vs_analytical['max_absolute_error']:.6f}",
                    f"{errors_fdm_vs_analytical['max_absolute_error']:.6f}"
                ],
                '平均絶対誤差': [
                    f"{errors_pinn_vs_analytical['mean_absolute_error']:.6f}",
                    f"{errors_fdm_vs_analytical['mean_absolute_error']:.6f}"
                ]
            })

            st.table(comparison_df)

            # どちらが正確か判定
            st.subheader("精度評価")

            if errors_pinn_vs_analytical['L2_relative_error'] < errors_fdm_vs_analytical['L2_relative_error']:
                st.success("✅ PINNの方がFDMより解析解に近い結果を得ています！")
            else:
                st.info("ℹ️ FDMの方がPINNより解析解に近い結果を得ています。")

            st.write(f"""
            - PINN誤差: {errors_pinn_vs_analytical['L2_relative_error']:.6f}
            - FDM誤差: {errors_fdm_vs_analytical['L2_relative_error']:.6f}
            - 誤差の差: {abs(errors_pinn_vs_analytical['L2_relative_error'] - errors_fdm_vs_analytical['L2_relative_error']):.6f}
            """)

# フッター
st.markdown("---")
st.markdown("**Physics-Informed Neural Networks (PINN)** で偏微分方程式を解くデモアプリケーション")
