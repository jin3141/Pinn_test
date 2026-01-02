# Hugging Face Spacesへのデプロイ手順

このドキュメントでは、PINN Burgers方程式ソルバーをHugging Face Spacesにデプロイする手順を説明します。

## 🎯 なぜHugging Face Spacesなのか？

- ✅ **無料枠が寛大**: 機械学習アプリに最適化された無料枠
- ✅ **簡単デプロイ**: ファイルをアップロードするだけ
- ✅ **永続的**: アプリが常時稼働
- ✅ **高速**: GPU/TPUも利用可能（有料）
- ✅ **自動更新**: GitHubと連携可能

## 📋 必要なもの

- Hugging Face アカウント（無料）
- このリポジトリのファイル

## 🚀 デプロイ手順

### ステップ1: Hugging Face アカウントを作成

1. [Hugging Face](https://huggingface.co/join)にアクセス
2. メールアドレスで無料アカウントを作成
3. メール認証を完了

### ステップ2: 新しいSpaceを作成

1. [新しいSpaceを作成](https://huggingface.co/new-space)にアクセス
2. 以下の設定を入力：

   ```
   Owner: あなたのユーザー名
   Space name: pinn-burgers-solver（または任意の名前）
   License: MIT
   Select the SDK: Streamlit
   Space hardware: CPU basic - Free（無料）
   ```

3. **Create Space** をクリック

### ステップ3: ファイルをアップロード

作成されたSpaceで、以下のファイルをアップロードします。

#### 方法A: Webインターフェースでアップロード

1. 「Files」タブをクリック
2. 「Add file」→「Upload files」を選択
3. 以下の5つのファイルをドラッグ&ドロップ：
   - `app.py`
   - `pinn_burgers.py`
   - `numerical_solver.py`
   - `requirements.txt`
   - `README.md`
4. 「Commit changes to main」をクリック

#### 方法B: Git CLIでプッシュ（上級者向け）

```bash
# Hugging Face Spaceをクローン
git clone https://huggingface.co/spaces/YOUR_USERNAME/pinn-burgers-solver
cd pinn-burgers-solver

# ファイルをコピー
cp /path/to/app.py .
cp /path/to/pinn_burgers.py .
cp /path/to/numerical_solver.py .
cp /path/to/requirements.txt .
cp /path/to/README.md .

# プッシュ
git add .
git commit -m "Initial commit"
git push
```

### ステップ4: デプロイを待つ

1. ファイルをアップロードすると、自動的にビルドが開始されます
2. 「Building」→「Running」のステータス変化を確認
3. 初回は**5-10分程度**かかります（TensorFlowのインストールに時間がかかる）
4. 完了すると「Running」となり、アプリが表示されます

### ステップ5: アプリを使う

1. Spaceのページでアプリが表示されます
2. サイドバーでパラメータを設定
3. 「訓練開始」ボタンをクリック
4. 結果を確認

## 🔄 アプリの更新

### 既存のSpaceを更新する

1. Spaceの「Files」タブ
2. 更新したいファイルをクリック
3. 「Edit」で編集、または「Upload file」で上書き
4. 「Commit changes」で保存
5. 自動的に再デプロイされます

### GitHubと連携して自動更新

1. Spaceの「Settings」タブ
2. 「Repository」セクションで「Link to a GitHub repository」
3. GitHubリポジトリを接続
4. 今後GitHubにpushすると自動的にSpaceが更新されます

## ⚙️ Spaceの設定

### ハードウェアのアップグレード（オプション）

無料のCPU basicで十分動作しますが、より高速な処理が必要な場合：

1. Spaceの「Settings」タブ
2. 「Hardware」セクション
3. CPU upgrade や GPU を選択（有料）

### プライベートSpaceにする

1. Spaceの「Settings」タブ
2. 「Visibility」を「Private」に変更
3. 自分だけがアクセス可能になります

## 🐛 トラブルシューティング

### ビルドエラーが発生する

1. 「Logs」タブでエラーメッセージを確認
2. `requirements.txt`のパッケージバージョンを確認
3. エラーメッセージをコピーして検索

### アプリが起動しない

1. ログで `streamlit run app.py` が実行されているか確認
2. README.mdの先頭にYAMLフロントマターがあるか確認：
   ```yaml
   ---
   title: PINN Burgers Equation Solver
   sdk: streamlit
   sdk_version: "1.28.0"
   app_file: app.py
   ---
   ```

### メモリ不足エラー

- 解析解のグリッド数を減らす（デフォルトで調整済み）
- エポック数を減らす
- 無料版では2GB RAMの制限があります

### TensorFlow関連のエラー

`requirements.txt`で互換性のあるバージョンを指定：
```
tensorflow>=2.13.0
```

## 📊 パフォーマンス最適化

### 無料枠での推奨設定

アプリのサイドバーで以下の設定を推奨：

- **エポック数**: 3000-5000（デフォルト: 5000）
- **解析解グリッド**: 50-100（デフォルト: 100）
- **数値解法グリッド**: 128-256（デフォルト: 256）

### 初回実行を高速化

- エポック数を少なくして試す
- 解析解の計算をオフにする（チェックボックス）

## 🔗 便利なリンク

- [Hugging Face Spaces ドキュメント](https://huggingface.co/docs/hub/spaces)
- [Streamlit on Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit)
- [Spaces FAQ](https://huggingface.co/docs/hub/spaces-overview)

## 💡 ヒント

1. **Space名は後で変更できます**（Settings → Rename）
2. **無料版でも十分動作します**（TensorFlowの訓練に対応）
3. **ログを見て進捗確認**（Logs タブ）
4. **README.mdがSpaceのホームページになります**

## 📸 スクリーンショット例

デプロイ後、Spaceのページには以下が表示されます：

```
🌊 PINN Burgers Equation Solver
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[アプリのプレビュー]

📚 Burgers方程式について
⚙️ パラメータ設定
🚀 訓練開始
```

デプロイ成功おめでとうございます！🎉
