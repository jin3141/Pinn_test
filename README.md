# PINN Burgers方程式ソルバー

Physics-Informed Neural Networks (PINN) を使ってBurgers方程式を解くStreamlitアプリケーション

## 概要

このアプリケーションは、PINNを使ってBurgers方程式を解き、従来の数値解法（有限差分法）と比較します。

### Burgers方程式

```
∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
```

- **初期条件**: u(x, 0) = -sin(πx)
- **境界条件**: u(-1, t) = u(1, t) = 0
- **粘性係数**: ν = 0.01/π

## 機能

1. **PINN解法**: ニューラルネットワークで偏微分方程式を直接解く
2. **数値解法**: 有限差分法（Crank-Nicolson法）で解く
3. **比較分析**: 両手法の結果を比較し、誤差を計算
4. **可視化**: 時空間での解の分布、誤差分布、訓練履歴を表示

## インストール

```bash
pip install -r requirements.txt
```

## 実行方法

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスします。

## 使い方

1. サイドバーでパラメータを設定
   - 粘性係数 ν
   - 空間・時間の範囲
   - PINNの訓練パラメータ（エポック数、学習率など）
   - 数値解法のグリッド数

2. 「訓練開始」ボタンをクリック

3. 結果を4つのタブで確認
   - **結果比較**: PINNと数値解法の比較
   - **PINN結果**: 訓練履歴と解の可視化
   - **数値解法結果**: 有限差分法の結果
   - **誤差分析**: 詳細な誤差統計

## ファイル構成

- `app.py`: Streamlitアプリケーションのメインファイル
- `pinn_burgers.py`: PINNモデルの実装
- `numerical_solver.py`: 有限差分法の実装
- `requirements.txt`: 必要なPythonパッケージ

## PINNについて

Physics-Informed Neural Networks (PINN) は、物理法則（偏微分方程式）を損失関数に組み込むことで、データが少ない状況でも物理的に妥当な解を得ることができる手法です。

### 損失関数

PINNの損失関数は以下の3つの項から構成されます：

1. **初期条件の損失**: t=0での値が初期条件を満たすか
2. **境界条件の損失**: 境界での値が境界条件を満たすか
3. **PDE残差の損失**: 領域内の点で物理法則（PDE）を満たすか

## デプロイ

### Streamlit Community Cloudへのデプロイ

1. GitHubリポジトリにコードをプッシュ
2. [Streamlit Community Cloud](https://streamlit.io/cloud)にアクセス
3. GitHubリポジトリを接続
4. `app.py`を指定してデプロイ

## 参考文献

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
