@echo off
chcp 65001 >nul
echo ============================================
echo  商品图片自动分类工具  PyInstaller 打包脚本
echo ============================================
echo.

:: 确认 PyInstaller 已安装
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装 PyInstaller...
    pip install pyinstaller
)

echo [打包] 开始打包，请稍候...
echo.

pyinstaller ^
    --noconfirm ^
    --onefile ^
    --windowed ^
    --name "商品图片分类工具" ^
    --hidden-import=sklearn.utils._cython_blas ^
    --hidden-import=sklearn.neighbors.typedefs ^
    --hidden-import=sklearn.neighbors._partition_nodes ^
    --hidden-import=sklearn.tree._utils ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=transformers.models.clip ^
    --hidden-import=huggingface_hub ^
    --hidden-import=openai ^
    image_classifier.py

echo.
if exist "dist\商品图片分类工具.exe" (
    echo [完成] 打包成功！
    echo        EXE 路径：%cd%\dist\商品图片分类工具.exe
) else (
    echo [失败] 打包未成功，请检查上方错误输出。
)
echo.
pause
