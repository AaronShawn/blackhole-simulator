@echo off
chcp 65001 >nul
title 发布到 GitHub - Schwarzschild Black Hole Simulator
cd /d "%~dp0"

echo ============================================================
echo   Schwarzschild Black Hole Simulator - 一键发布到 GitHub
echo ============================================================
echo.
echo  需要一枚 GitHub Personal Access Token（只用于本次发布）：
echo    classic token      : 勾选 repo（如需改 CI 再勾 workflow）
echo    fine-grained token : 仓库权限 Contents=RW, Administration=RW, Metadata=R
echo  创建地址: https://github.com/settings/tokens
echo.
echo  粘贴时不会显示字符，回车开始发布。
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish.ps1" %*

echo.
echo 按任意键关闭窗口...
pause >nul
