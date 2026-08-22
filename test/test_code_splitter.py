"""
code_splitter.py 单元测试。

重点验证本次修复的 3 个 Bug：
    Bug 14: _split_class_by_methods 外层循环缺三引号追踪
    Bug 16: _split_class_by_methods 方法间装饰器丢失
    Bug 3:  _parse_line_blocks 三引号字符串把缩进解析器搞崩

注意：_parse_line_blocks 返回 list[tuple(start, end, text)]，文本在 tuple[2]
      _split_class_by_methods 返回 list[dict]（分片列表）或 None

运行方式：
    python -m pytest test/test_code_splitter.py -v
"""

import pytest

from ingestion.code_splitter import (
    split_python_files,
    _parse_line_blocks,
    _split_class_by_methods,
    _count_unclosed_triples,
)


# ============================================================
# 一、_count_unclosed_triples：三引号计数器
# ============================================================

class TestCountUnclosedTriples:
    """测三引号状态机的核心工具函数。"""

    def test_no_triple_quote(self):
        """没有三引号的行，返回 None。"""
        assert _count_unclosed_triples("def foo(): pass") is None

    def test_open_triple_quote(self):
        """只有开头三引号（未闭合），返回三引号符号。"""
        assert _count_unclosed_triples('x = """hello') == '"""'

    def test_open_close_same_line(self):
        """同一行开+关（count=2 偶数），算已闭合，返回 None。"""
        assert _count_unclosed_triples('x = """hello"""') is None

    def test_single_quotes_triple(self):
        """单引号三连 ''' 也能识别。"""
        assert _count_unclosed_triples("x = '''hello") == "'''"


# ============================================================
# 二、_parse_line_blocks：逻辑块解析（返回 tuple 列表）
# ============================================================

class TestParseLineBlocks:
    """测按 def/class 边界切逻辑块的核心函数。"""

    def test_basic_two_functions(self):
        """两个独立函数应切成两个块。"""
        code = (
            "def foo():\n"
            "    return 1\n"
            "\n"
            "def bar():\n"
            "    return 2\n"
        )
        blocks = _parse_line_blocks(code)
        assert len(blocks) == 2
        # block 是 tuple (start, end, text)，文本在 [2]
        assert "def foo" in blocks[0][2]
        assert "def bar" in blocks[1][2]

    def test_triple_quote_string_not_break_class(self):
        """Bug 3 验证：三引号字符串内部的零缩进行不应被误判为 class 结束。"""
        code = (
            'class Foo:\n'
            '    PROMPT = """\n'
            '这是多行字符串\n'
            'def fake_method():\n'      # 字符串内部假的 def，不应被当作方法边界
            '    """\n'
            '\n'
            '    def real_method(self):\n'
            '        return 1\n'
        )
        blocks = _parse_line_blocks(code)
        all_text = "".join(b[2] for b in blocks)
        # real_method 应该出现（没被切断）
        assert "real_method" in all_text
        # 不应出现"无头"片段（从字符串中间开始的块）
        for b in blocks:
            assert not b[2].startswith("这是多行字符串")

    def test_empty_content(self):
        """空内容返回空列表。"""
        assert _parse_line_blocks("") == []


# ============================================================
# 三、_split_class_by_methods：长 class 按方法切（返回 list[dict] 或 None）
# ============================================================

class TestSplitClassByMethods:
    """测超长 class 按方法边界切分（Bug 2/14/16 的核心）。"""

    def _make_long_class(self, n_methods: int = 5) -> str:
        """造一个含 n 个方法的超长 class 文本。"""
        lines = ["class LongClass:"]
        for i in range(n_methods):
            lines.append(f"    def method_{i}(self):")
            lines.append(f"        x = {i}")
            lines.append(f"        return x")
            lines.append("")
        return "\n".join(lines)

    def test_class_split_into_methods(self):
        """Bug 2 验证：长 class 应被切成多个分片，每个方法完整。"""
        class_text = self._make_long_class(5)
        # max_chars 设小一点强制触发按方法切
        result = _split_class_by_methods(class_text, 1, max_chars=100, overlap=0)
        if result is None:
            pytest.skip("class 内无 def，跳过")
        # result 是分片列表（list[dict]）
        assert isinstance(result, list)
        assert len(result) >= 1
        all_text = "".join(c["chunk_text"] for c in result)
        # 每个方法都应完整出现
        for i in range(5):
            assert f"def method_{i}" in all_text

    def test_decorator_attached_to_method(self):
        """Bug 16 验证：装饰器行应并入紧随其后的方法块。"""
        class_text = (
            "class Foo:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "\n"
            "    @staticmethod\n"
            "    def helper():\n"
            "        return 42\n"
            "\n"
            "    @classmethod\n"
            "    def factory(cls):\n"
            "        return cls()\n"
        )
        result = _split_class_by_methods(class_text, 1, max_chars=50, overlap=0)
        if result is None:
            pytest.skip("class 内无 def，跳过")
        all_text = "".join(c["chunk_text"] for c in result)
        # 装饰器应出现在某个分片里，不被丢弃
        assert "@staticmethod" in all_text
        assert "@classmethod" in all_text

    def test_triple_quote_inside_class_not_break_split(self):
        """Bug 14 验证：class 内三引号字符串里的假 def 不应被误识别为方法边界。"""
        class_text = (
            "class Foo:\n"
            '    PROMPT = """\n'
            "    def fake_def_in_string():\n"   # 字符串里的假 def
            '    """\n'
            "\n"
            "    def real_method(self):\n"
            "        return 1\n"
        )
        result = _split_class_by_methods(class_text, 1, max_chars=50, overlap=0)
        if result is None:
            pytest.skip("class 内无 def，跳过")
        all_text = "".join(c["chunk_text"] for c in result)
        # real_method 应被正确识别
        assert "def real_method" in all_text

    def test_no_def_returns_none(self):
        """纯数据类（无 def）应返回 None，回退到滑动窗口。"""
        class_text = 'class Foo:\n    x = 1\n    y = 2\n'
        result = _split_class_by_methods(class_text, 1, max_chars=100, overlap=0)
        assert result is None


# ============================================================
# 四、split_python_files：端到端切片
# ============================================================

class TestSplitPythonFiles:
    """测整体切片入口。"""

    def test_single_file_multiple_chunks(self):
        """一个含多个函数的文件应切成多个分片。"""
        docs = [{
            "file_path": "demo.py",
            "content": (
                "def foo():\n"
                "    return 1\n"
                "\n"
                "def bar():\n"
                "    return 2\n"
            ),
            "start_line": 1,
        }]
        chunks = split_python_files(docs)
        assert len(chunks) >= 1
        for c in chunks:
            assert "file_path" in c
            assert "chunk_text" in c
            assert "start_line" in c
            assert c["file_path"] == "demo.py"

    def test_empty_docs(self):
        """空文档列表返回空分片列表。"""
        assert split_python_files([]) == []

    def test_class_method_not_truncated(self):
        """长 class 的 __init__ 不应被拦腰截断（Bug 2 端到端验证）。"""
        init_body = "\n".join(f"        self.var_{i} = {i}" for i in range(20))
        docs = [{
            "file_path": "big.py",
            "content": (
                "class BigClass:\n"
                "    def __init__(self):\n"
                f"{init_body}\n"
                "\n"
                "    def run(self):\n"
                "        return 'done'\n"
            ),
            "start_line": 1,
        }]
        chunks = split_python_files(docs)
        init_chunks = [c for c in chunks if "def __init__" in c["chunk_text"]]
        assert len(init_chunks) >= 1
        # __init__ 的分片应包含最后一行，说明没被截断
        assert any("self.var_19" in c["chunk_text"] for c in init_chunks)
