#!/bin/bash
# Hugging Face Spacesへのデプロイスクリプト

echo "Hugging Face Spacesへデプロイします..."
echo ""
echo "事前準備："
echo "1. https://huggingface.co/new-space でSpaceを作成してください"
echo "   - Space name: pinn-burgers-solver (または任意)"
echo "   - SDK: Streamlit"
echo "   - Hardware: CPU basic"
echo ""
echo "2. https://huggingface.co/settings/tokens でAccess Tokenを生成してください"
echo ""
read -p "Spaceを作成しましたか？ (y/n): " created

if [ "$created" != "y" ]; then
    echo "まずSpaceを作成してください"
    exit 1
fi

read -p "Hugging Faceユーザー名: " username
read -p "Space名 (例: pinn-burgers-solver): " space_name

echo ""
echo "以下のコマンドを実行してください："
echo ""
echo "git remote add hf https://huggingface.co/spaces/$username/$space_name"
echo "git push hf claude/pinn-burgers-equation-CGuOc:main --force"
echo ""
echo "認証が求められたら："
echo "- Username: $username"
echo "- Password: [Access Token from https://huggingface.co/settings/tokens]"
