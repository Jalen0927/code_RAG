"""
multi_retriever.py 单元测试。

重点验证本次修复的 3 个 Bug：
    Bug 12: _compute_keyword_boost 子串匹配误命中（改用词边界）
    Bug 13: _extract_key_terms 停用词表不完整（只对英文标识符生效）
    Bug 15: _compute_source_boost 用 test 子串误判测试文件

还验证：
    Bug 10: 乘法增强常量配置（KEYWORD_BOOST_MULTIPLIER=3.0, SOURCE_BOOST_MULTIPLIER=0.8）

注意：_extract_key_terms 对中文是整段提取（>=2 字连续汉字作为一个 term），
      停用词过滤只对英文标识符生效，中文段不过滤。

运行方式：
    python -m pytest test/test_multi_retriever.py -v
"""

import pytest

from retriever.multi_retriever import MultiRetriever


# ============================================================
# 一、_extract_key_terms：关键词提取（Bug 13 停用词）
# ============================================================

class TestExtractKeyTerms:
    """测从查询里提取关键词。"""

    def test_english_identifier_extracted(self):
        """英文标识符（>=3 字符）应被提取。"""
        terms = MultiRetriever._extract_key_terms("how does run method work")
        lowered = [t.lower() for t in terms]
        assert "run" in lowered
        assert "method" in lowered
        assert "work" in lowered

    def test_dunder_method_extracted(self):
        """双下划线方法名 __init__ 应被提取。"""
        terms = MultiRetriever._extract_key_terms("查询 __init__ 方法")
        lowered = [t.lower() for t in terms]
        assert "__init__" in lowered

    def test_chinese_segment_extracted_whole(self):
        """中文连续段（>=2 字）应整体提取为一个 term。"""
        terms = MultiRetriever._extract_key_terms("加载文件")
        # "加载文件" 作为一个整体中文段
        assert "加载文件" in terms or "加载" in terms or "文件" in terms

    def test_english_stopwords_filtered(self):
        """Bug 13 验证：英文停用词应被过滤，不进入 key_terms。"""
        # Bug 13 扩充的英文停用词
        for stopword in ["where", "when", "why", "with", "from",
                         "this", "that", "are", "have", "would", "should"]:
            terms = MultiRetriever._extract_key_terms(f"{stopword} run method")
            lowered = [t.lower() for t in terms]
            assert stopword.lower() not in lowered, f"英文停用词 '{stopword}' 没被过滤掉"


# ============================================================
# 二、_compute_keyword_boost：关键词加权（Bug 12 词边界）
# ============================================================

class TestComputeKeywordBoost:
    """测关键词加权分数，重点验证词边界匹配。"""

    def test_exact_word_match(self):
        """Bug 12 验证：chunk 含 def run，问 run 应命中。"""
        chunk = "def run(self, query):\n    return result"
        boost = MultiRetriever._compute_keyword_boost(chunk, ["run"])
        assert boost > 0

    def test_substring_not_matched(self):
        """Bug 12 核心验证：含 current 的 chunk，问 run 不应被算命中。"""
        chunk = "current_step = 1\nrunning = True"
        boost = MultiRetriever._compute_keyword_boost(chunk, ["run"])
        # 修复前：子串匹配会命中（current 里有 run），boost=1.0
        # 修复后：词边界断言，current 里的 run 不算命中
        assert boost == 0.0

    def test_initialize_not_match_init(self):
        """Bug 12 验证：含 initialize 的 chunk，问 init 不应命中。"""
        chunk = "def initialize_system():\n    pass"
        boost = MultiRetriever._compute_keyword_boost(chunk, ["init"])
        assert boost == 0.0

    def test_chinese_substring_match(self):
        """中文用子串匹配（没有词边界概念）。"""
        chunk = "这是初始化逻辑"
        boost = MultiRetriever._compute_keyword_boost(chunk, ["初始化"])
        assert boost > 0

    def test_empty_key_terms_returns_zero(self):
        """空关键词列表返回 0。"""
        assert MultiRetriever._compute_keyword_boost("any text", []) == 0.0

    def test_partial_hit(self):
        """部分命中应给部分分数。"""
        chunk = "def run(self): pass"
        # 两个词，只命中一个，boost 应 = 0.5
        boost = MultiRetriever._compute_keyword_boost(chunk, ["run", "nonexistent_xyz"])
        assert 0 < boost <= 0.5 + 1e-9


# ============================================================
# 三、_compute_source_boost：源文件加权（Bug 15 路径判断）
# ============================================================

class TestComputeSourceBoost:
    """测源文件/测试文件判断，重点验证精确路径判断。"""

    def test_normal_source_file(self):
        """正常源文件应得 1.0。"""
        assert MultiRetriever._compute_source_boost("source_code/file_agent/react_scheduler.py") == 1.0

    def test_test_file_in_tests_dir(self):
        """tests/ 目录下的文件应得 0.0。"""
        assert MultiRetriever._compute_source_boost("source_code/file_agent/tests/test_basic.py") == 0.0

    def test_test_file_prefix(self):
        """test_ 开头的文件应得 0.0。"""
        assert MultiRetriever._compute_source_boost("test_react_scheduler.py") == 0.0

    def test_test_file_suffix(self):
        """_test.py 结尾的文件应得 0.0。"""
        assert MultiRetriever._compute_source_boost("react_test.py") == 0.0

    def test_latest_not_judged_as_test(self):
        """Bug 15 核心验证：latest.py 含 test 子串但不是测试文件，应得 1.0。"""
        assert MultiRetriever._compute_source_boost("latest.py") == 1.0

    def test_protest_not_judged_as_test(self):
        """Bug 15 验证：protest.py 含 test 子串但不是测试文件。"""
        assert MultiRetriever._compute_source_boost("protest.py") == 1.0

    def test_pycache_filtered(self):
        """__pycache__ 路径应得 0.0。"""
        assert MultiRetriever._compute_source_boost("__pycache__/main.cpython.pyc") == 0.0

    def test_test_dir_exact_match(self):
        """test/ 目录（单数）应得 0.0。"""
        assert MultiRetriever._compute_source_boost("test/example.py") == 0.0


# ============================================================
# 四、乘法增强常量配置（Bug 10）
# ============================================================

class TestBoostMultipliers:
    """Bug 10 验证：乘法增强的常量配置。

    公式：final = base × (1 + KEYWORD_BOOST_MULTIPLIER × kw_boost)
                        × (1 + SOURCE_BOOST_MULTIPLIER × src_boost)
    """

    def test_keyword_boost_multiplier_is_3(self):
        """关键词乘数应为 3.0（命中时放大 4 倍）。"""
        assert MultiRetriever.KEYWORD_BOOST_MULTIPLIER == 3.0

    def test_source_boost_multiplier_is_0_8(self):
        """源文件乘数应为 0.8（源文件放大 1.8 倍）。"""
        assert MultiRetriever.SOURCE_BOOST_MULTIPLIER == 0.8

    def test_keyword_base_weight_positive(self):
        """关键词基础权重应为正数。"""
        assert MultiRetriever.KEYWORD_BASE_WEIGHT > 0

    def test_amplification_ratio(self):
        """验证放大倍数：kw=1 + src=1 时总放大 4×1.8=7.2 倍。"""
        base = 1.0
        kw_mul = 1.0 + MultiRetriever.KEYWORD_BOOST_MULTIPLIER * 1.0  # 4.0
        src_mul = 1.0 + MultiRetriever.SOURCE_BOOST_MULTIPLIER * 1.0  # 1.8
        final = base * kw_mul * src_mul
        assert abs(final - 7.2) < 1e-6
