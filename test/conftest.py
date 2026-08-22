"""
pytest 全局配置（自动加载，不需要手动 import）。

作用：
    1. 把项目根目录加入 sys.path，让 test 目录下所有测试文件都能 import 根目录的模块
    2. 强制 stdout/stderr 用 UTF-8 编码，避免 Windows 终端 GBK 报错（Bug 4）

有了这个文件，每个测试文件就不用重复写 sys.path 和编码处理的样板代码。
"""

import sys
import os

# Windows 终端 UTF-8 兜底（和 main.py 的 Bug 4 修复一致）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 项目根目录 = test 目录的上一级
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
