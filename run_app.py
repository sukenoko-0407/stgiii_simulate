"""
Streamlit app runner script
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# アプリを実行
from app.main import main

main()
