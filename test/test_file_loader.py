"""
file_loader.py 单元测试。

验证源码文件加载器的核心逻辑：
    - 只读取 .py 文件
    - 跳过缓存目录（__pycache__ / .pytest_cache 等）
    - 目录不存在时返回空列表
    - 读取内容正确

注意：临时目录要建在项目所在盘（D:），避免 os.path.relpath 跨盘报错。

运行方式：
    python -m pytest test/test_file_loader.py -v
"""

import os
import tempfile

from ingestion.file_loader import load_source_files


def _make_tmp_project() -> str:
    """在项目目录下创建临时项目目录（同盘，避免 relpath 跨盘错误）。"""
    # dir="." 保证临时目录在 cwd（D:\code_RAG）下，和项目同盘
    tmp = tempfile.mkdtemp(prefix="coderag_test_", dir=".")
    with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as f:
        f.write("print('hello')\n")
    # 非 .py 文件（应被跳过）
    with open(os.path.join(tmp, "readme.md"), "w", encoding="utf-8") as f:
        f.write("# README")
    # 子目录里的 .py
    sub = os.path.join(tmp, "pkg")
    os.makedirs(sub)
    with open(os.path.join(sub, "helper.py"), "w", encoding="utf-8") as f:
        f.write("def help(): pass\n")
    # __pycache__ 缓存目录（应被跳过）
    cache = os.path.join(tmp, "__pycache__")
    os.makedirs(cache)
    with open(os.path.join(cache, "main.cpython.pyc"), "wb") as f:
        f.write(b"\x00")
    return tmp


class TestLoadSourceFiles:
    """测 load_source_files 的核心行为。"""

    def test_only_py_files_loaded(self):
        """只加载 .py 文件，.md 等非 Python 文件被跳过。"""
        tmp = _make_tmp_project()
        try:
            docs = load_source_files(tmp)
            file_names = [os.path.basename(d["file_path"]) for d in docs]
            assert "main.py" in file_names
            assert "helper.py" in file_names
            assert "readme.md" not in file_names
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_skip_pycache_dir(self):
        """__pycache__ 目录应被跳过，不读取 .pyc。"""
        tmp = _make_tmp_project()
        try:
            docs = load_source_files(tmp)
            for d in docs:
                assert "__pycache__" not in d["file_path"]
                assert not d["file_path"].endswith(".pyc")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_nonexistent_dir_returns_empty(self):
        """目录不存在时返回空列表，不抛异常。"""
        docs = load_source_files("this_dir_does_not_exist_xyz")
        assert docs == []

    def test_content_and_start_line(self):
        """返回的文档项应包含正确的内容和 start_line=1。"""
        tmp = _make_tmp_project()
        try:
            docs = load_source_files(tmp)
            main_doc = next(d for d in docs if d["file_path"].endswith("main.py"))
            assert "print('hello')" in main_doc["content"]
            assert main_doc["start_line"] == 1
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
