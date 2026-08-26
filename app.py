"""
app.py — CodeRAG 的 Streamlit Web 演示界面。

职责：
  - 调用 main.build_index() 一次性初始化全部 RAG 组件（缓存复用）
  - 提供浏览器可视化的提问界面，替代命令行交互
  - 将检索 Top3 代码片段与 LLM 回答结构化为可折叠面板展示
  - 附带 3 个一键示例问题，方便面试/演示快速展示效果
  - 新增"源码浏览器"标签页：支持浏览 source_code 目录下任意文件的
    完整源码，并支持根据行号快速跳转

运行方式（先 cd 到项目根目录）：
    首次安装：pip install streamlit
    启动命令：python -m streamlit run app.py
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
import zipfile
from contextlib import redirect_stdout

# 保证项目根目录在搜索路径里，与 main.py 保持一致
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 确保 Windows 终端能正确显示中文
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# 外部库
# ============================================================
try:
    import streamlit as st
except ImportError:
    print("❌ 未检测到 streamlit，请先执行：pip install streamlit")
    print("   然后再运行：python -m streamlit run app.py")
    sys.exit(1)

# ============================================================
# 项目内部模块
# ============================================================
from main import build_index
from embedding_client import EmbeddingError
from llm_client import LLMRequestError


# ============================================================
# 页面基础配置
# ============================================================
st.set_page_config(
    page_title="CodeRAG · 代码知识库问答",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# 主题增强：Trae / VS Code Dark+ 风格（中性深灰系，用户截图参考色）
# 核心色板：
#   全局主底 #1e1e1e（编辑器底）、侧边栏 #252526、面板 #2d2d30
#   主色 #0e639c（选中蓝）、边框 #3c3c3c、字色 #cccccc / #9cdcfe
# 通过稳定 data-* / aria-* 属性覆盖，不去改 Streamlit 内部类名
# 保留代码块 Pygments 语法高亮彩色（用 VSCode Dark+ 那套配色）
# ============================================================
def _inject_theme_css() -> None:
    st.markdown(
        """
<style>
    /* 全局：VSCode Dark+ 主底 #1e1e1e + 柔和行距 */
    html, body, [class*="css"], section.main, .block-container {
        background-color: #1e1e1e !important;
        color: #cccccc;
        line-height: 1.65;
    }

    /* 标题与正文：中性深灰系，不把 span/div 写进来以防覆盖 Pygments 高亮 */
    h1, h2, h3, h4, h5, h6, p, li, label, small {
        color: #cccccc !important;
    }
    /* 主标题用 VSCode 的变量浅蓝（#9cdcfe），和截图里标题色一致 */
    h1, h2 { color: #9cdcfe !important; }
    h3, h4 { color: #569cd6 !important; }

    /* 代码块：VSCode 底 #1e1e1e（和页面背景几乎同色）+ 细边 #3c3c3c
       只改背景和边框，不动 padding/margin 让 Streamlit 自己管布局
       注意：不单独给 pre 加样式，避免干扰 React reconciliation */
    .stCodeBlock {
        background-color: #1e1e1e !important;
        border: 1px solid #3c3c3c !important;
        border-radius: 6px !important;
    }
    .stCodeBlock pre {
        background-color: #1e1e1e !important;
    }
    /* 行号列：VSCode 灰 #858585，细一点 */
    .stCodeBlock [data-testid="stCodeBlockLineNumbers"],
    .stCodeBlock .linenos,
    .stCodeBlock .line-numbers {
        color: #858585 !important;
    }
    /* 代码里没被 span 包到的普通字符，用 VSCode 主文字色
       只改 code 标签本身，不动 pre 让 Streamlit 自己管 pre 的布局 */
    .stCodeBlock code, .stCodeBlock pre code {
        color: #d4d4d4 !important;
    }

    /* ========== 代码语法高亮：VSCode Dark+ 标准配色 ==========
       每条都加 !important 覆盖 Pygments 的 inline style */
    /* 关键字：def / if / return / import / for / while / try / raise ...（#569cd6） */
    .stCodeBlock span[style*="color: rgb(0, 128,"],
    .stCodeBlock span[style*="color: rgb(0,0,255)"],
    .stCodeBlock span[style*="color: rgb(86, 156,"],
    .stCodeBlock span[style*="color:#0080"],
    .stCodeBlock span[style*="color:#00f"],
    .stCodeBlock span[style*="color:#0000ff"] {
        color: #569cd6 !important;
    }
    /* 内置名 / 装饰器 / 异常类（浅紫 #c586c0） */
    .stCodeBlock span[style*="color: rgb(128, 0,"],
    .stCodeBlock span[style*="color: rgb(102, 0,"],
    .stCodeBlock span[style*="color: rgb(197, 134,"],
    .stCodeBlock span[style*="color:#800080"],
    .stCodeBlock span[style*="color:#660087"] {
        color: #c586c0 !important;
    }
    /* 字符串 / docstring（橙棕 #ce9178，VSCode 字符串色） */
    .stCodeBlock span[style*="color: rgb(163, 21,"],
    .stCodeBlock span[style*="color: rgb(0, 128"],
    .stCodeBlock span[style*="color: rgb(4, 130"],
    .stCodeBlock span[style*="color: rgb(206, 145,"],
    .stCodeBlock span[style*="color:#008000"],
    .stCodeBlock span[style*="color:#0451a5"] {
        color: #ce9178 !important;
    }
    /* 数字 / None / True / False（黄 #dcdcaa） */
    .stCodeBlock span[style*="color: rgb(0, 0,"],
    .stCodeBlock span[style*="color: rgb(220, 220,"],
    .stCodeBlock span[style*="color:#098658"],
    .stCodeBlock span[style*="color:#0451a5"]:not([style*="0,128"]) {
        color: #dcdcaa !important;
    }
    /* 函数名本身（金黄 #dcdcaa / 部分主题给米黄） */
    .stCodeBlock span[style*="color: rgb(121,"],
    .stCodeBlock span[style*="color: rgb(220,220,170"],
    .stCodeBlock span[style*="color: rgb(236, 196,"] {
        color: #dcdcaa !important;
    }
    /* 括号/逗号/普通代码字（保持白灰） */
    .stCodeBlock span[style*="color: rgb(0, 0, 0)"],
    .stCodeBlock span[style*="color: rgb(31, 31,"] {
        color: #d4d4d4 !important;
    }
    /* 注释 / docstring（草绿斜体 #6a9955，VSCode 默认） */
    .stCodeBlock span[style*="color: rgb(107, 113,"],
    .stCodeBlock span[style*="color: rgb(128, 128,"],
    .stCodeBlock span[style*="color: rgb(106, 153,"],
    .stCodeBlock span[style*="color: #0080"],
    .stCodeBlock span[style*="color:#808080"] {
        color: #6a9955 !important;
        font-style: italic;
    }
    /* 操作符 / 标点红 */
    .stCodeBlock span[style*="color: rgb(163, 21, 21"] {
        color: #f44747 !important;
    }
    /* hljs 类名兼容 */
    .stCodeBlock .hljs-keyword { color: #569cd6 !important; }
    .stCodeBlock .hljs-string  { color: #ce9178 !important; }
    .stCodeBlock .hljs-number  { color: #dcdcaa !important; }
    .stCodeBlock .hljs-comment { color: #6a9955 !important; font-style: italic; }
    .stCodeBlock .hljs-built_in,.stCodeBlock .hljs-type { color: #c586c0 !important; }
    .stCodeBlock .hljs-function,.stCodeBlock .hljs-title { color: #dcdcaa !important; }
    .stCodeBlock .hljs-name    { color: #9cdcfe !important; }
    .stCodeBlock .hljs-meta    { color: #569cd6 !important; }
    .stCodeBlock .hljs-params  { color: #9cdcfe !important; }
    .stCodeBlock .hljs-literal { color: #dcdcaa !important; }
    .stCodeBlock .hljs-operator{ color: #d4d4d4 !important; }
    .stCodeBlock .hljs-punctuation { color: #d4d4d4 !important; }

    /* 主按钮：VSCode 活动栏蓝 #0e639c，hover 更亮一点 */
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background-color: #0e639c;
        border-color: #1177bb;
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        background-color: #1177bb;
        border-color: #007acc;
    }

    /* 普通按钮：VSCode 面板灰底 + 浅蓝字，hover 加选中蓝底 */
    .stButton > button:not([kind="primary"]) {
        background-color: #2d2d30;
        color: #cccccc;
        border: 1px solid #3c3c3c;
    }
    .stButton > button:not([kind="primary"]):hover {
        background-color: #094771;
        color: #ffffff;
        border-color: #0e639c;
    }

    /* 进度条：选中蓝 */
    div[data-testid="stProgress"] > div > div > div {
        background-color: #0e639c !important;
    }

    /* 提示卡片 info/error/success：统一面板灰底 + 灰边框，
       图标只留蓝/红，彻底禁绿 */
    div[data-testid="stAlertContainer"] > div[data-testid="stAlert"] {
        background-color: #252526 !important;
        border: 1px solid #3c3c3c !important;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        color: #cccccc !important;
    }
    div[data-testid="stAlertContainer"] > div[data-testid="stAlert"] div,
    div[data-testid="stAlertContainer"] > div[data-testid="stAlert"] p,
    div[data-testid="stAlertContainer"] > div[data-testid="stAlert"] li {
        color: #cccccc !important;
    }
    div[data-testid="stAlertContainer"] > div[data-testid="stAlert"] svg {
        color: #569cd6 !important;
    }

    /* Tabs：VSCode 标签页风格 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.1rem;
        border-bottom: 1px solid #3c3c3c;
        background-color: #252526;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: transparent !important;
        color: #969696 !important;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #ffffff !important;
        background-color: #1e1e1e !important;
        border-bottom: 2px solid #007acc;
    }

    /* 侧边栏：#252526（VSCode 资源管理器）*/
    section[data-testid="stSidebar"] {
        background-color: #252526 !important;
        border-right: 1px solid #3c3c3c;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] small {
        color: #cccccc !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #9cdcfe !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] * {
        color: #969696 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] * {
        color: #e0e0e0 !important;
    }

    /* Metric 卡片：面板灰底 + 细灰边 */
    div[data-testid="stMetric"] {
        background-color: #2d2d30;
        border: 1px solid #3c3c3c;
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }

    /* 输入框 / 下拉 / 数字输入：VSCode 编辑器输入框样式 */
    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="textarea"],
    textarea, input, select {
        background-color: #3c3c3c !important;
        color: #cccccc !important;
        caret-color: #aeafad !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #3c3c3c !important;
        border-color: #3c3c3c !important;
        color: #cccccc !important;
    }
    /* focus 时上蓝边，和 VSCode 输入一致 */
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {
        border-color: #007acc !important;
        box-shadow: inset 0 0 0 1px #007acc;
    }
    textarea::placeholder, input::placeholder {
        color: #858585 !important;
    }
    li[role="option"] {
        background-color: #252526 !important;
        color: #cccccc !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #094771 !important;
        color: #ffffff !important;
    }

    /* radio 单选按钮（视图模式切换） */
    div[role="radiogroup"] label {
        color: #cccccc !important;
    }
    div[role="radiogroup"] input[type="radio"] + div {
        background-color: #2d2d30 !important;
        border-color: #3c3c3c !important;
    }

    /* 分隔线 */
    hr {
        border-color: #3c3c3c;
    }

    /* expander 折叠面板：面板灰底 */
    details {
        background-color: #252526 !important;
        border: 1px solid #3c3c3c !important;
        border-radius: 6px;
        padding: 0.25rem 0.75rem;
    }
    summary {
        color: #9cdcfe !important;
    }

    /* toast 提示 */
    div[data-testid="stToast"] {
        background-color: #252526 !important;
        border: 1px solid #3c3c3c !important;
        color: #cccccc !important;
    }

    /* ============================================================
       源码浏览器"一层一层叠起来"的树形样式
       用法：外层 div 给 .tree-layer，文件条目 .tree-item
       利用左边框颜色从深到浅，每层缩进一级，视觉上就是一层层叠起来
       ============================================================ */
    .tree-container {
        background-color: #252526;
        border: 1px solid #3c3c3c;
        border-radius: 6px;
        padding: 0.5rem 0;
    }
    .tree-layer {
        padding: 0 0 0 14px;
        border-left: 1px solid #3c3c3c;
        margin-left: 8px;
        margin-top: 2px;
    }
    .tree-layer.level-0 { border-left-color: #3c3c3c; margin-left: 4px; }
    .tree-layer.level-1 { border-left-color: #454545; }
    .tree-layer.level-2 { border-left-color: #505050; }
    .tree-layer.level-3 { border-left-color: #5a5a5a; }
    .tree-layer.level-4 { border-left-color: #666666; }

    /* 文件夹标题（折叠式） */
    .tree-folder-header {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 2px 6px;
        border-radius: 4px;
        color: #cccccc;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .tree-folder-header:hover {
        background-color: #2a2d2e;
    }
    .tree-folder-icon {
        color: #dcb67a;  /* VSCode 文件夹黄色 */
        display: inline-block;
        width: 1rem;
        text-align: center;
    }

    /* 文件条目（可点击选中） */
    .tree-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 2px 6px 2px calc(1.1rem + 6px);
        border-radius: 4px;
        font-size: 0.92rem;
        cursor: pointer;
        color: #cccccc;
    }
    .tree-item:hover {
        background-color: #2a2d2e;
    }
    .tree-item.tree-item-active {
        background-color: #094771;
        color: #ffffff;
    }
    .tree-item-icon {
        display: inline-block;
        width: 1rem;
        text-align: center;
        color: #569cd6;  /* VSCode Python 文件图标色 */
    }
    .tree-item-path {
        font-size: 0.78rem;
        color: #858585;
        margin-left: auto;
        padding-left: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 40%;
    }
    .tree-item.tree-item-active .tree-item-path {
        color: #c8c8c8;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


_inject_theme_css()


# ============================================================
# 源码浏览器：辅助函数
# ============================================================
_DEFAULT_SOURCE_ROOT = os.path.join(ROOT_DIR, "source_code")


def _get_source_root() -> str:
    """获取当前使用的源码根目录（支持动态切换）。"""
    return st.session_state.get("_current_source_dir", _DEFAULT_SOURCE_ROOT)


@st.cache_data(show_spinner=False)
def _list_source_files(source_root: str) -> list[str]:
    """列出指定目录下所有 .py 文件（相对路径，排序）。"""
    files: list[str] = []
    if not os.path.isdir(source_root):
        return files
    for root, _dirs, fnames in os.walk(source_root):
        for fn in fnames:
            if not fn.endswith(".py"):
                continue
            abs_path = os.path.join(root, fn)
            rel_path = os.path.relpath(abs_path, source_root)
            files.append(rel_path)
    return sorted(files)


@st.cache_data(show_spinner=False)
def _build_source_tree(files: list[str], root_name: str = "source_code") -> dict:
    """
    把相对路径的文件列表转成嵌套字典的树形结构，方便一层一层渲染。

    返回示例：
    {
        "source_code": {
            "__type__": "dir",
            "__children__": {
                "ingestion": {
                    "__type__": "dir",
                    "__children__": {
                        "code_splitter.py": {"__type__": "file", "path": "source_code/ingestion/code_splitter.py"},
                        ...
                    }
                },
                ...
            }
        }
    }
    """
    root_node: dict = {}
    for rel_path in files:
        parts = rel_path.replace("\\", "/").split("/")
        node = root_node
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            if is_last:
                node[part] = {"__type__": "file", "path": rel_path}
            else:
                if part not in node:
                    node[part] = {"__type__": "dir", "__children__": {}}
                node = node[part]["__children__"]
    return {root_name: {"__type__": "dir", "__children__": root_node}}


def _read_source(rel_path: str, source_root: str) -> str:
    """根据相对路径读源码，读不到返回空串。"""
    abs_path = os.path.join(source_root, rel_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(abs_path, "r", encoding="gbk") as f:
                return f.read()
        except Exception:
            return ""


def _build_line_view(text: str, target_line: int | None, context: int = 30) -> tuple[str, int, int]:
    """
    当用户点了"跳转到第 N 行"时，只显示 target_line 前后 context 行，
    并给出高亮行在局部视图里是第几行（方便肉眼定位）。
    返回 (局部文本, 局部起始行号, 需要高亮的局部行号或 0)。
    """
    all_lines = text.splitlines()
    total = len(all_lines)

    if target_line is None or target_line < 1:
        return text, 1, 0
    target_line = max(1, min(target_line, total))

    start = max(1, target_line - context)
    end = min(total, target_line + context)
    local_lines = all_lines[start - 1: end]
    highlight_local = target_line - start + 1
    return "\n".join(local_lines), start, highlight_local


# ============================================================
# 索引初始化（缓存：支持按 source_dir 动态切换）
# ============================================================
@st.cache_resource(show_spinner=False)
def _load_components_and_logs(source_dir: str = _DEFAULT_SOURCE_ROOT):
    """
    初始化 RAG 组件（加缓存确保不会因页面刷新重复向量化）。
    同时捕获 build_index 过程中的 print 日志，用于侧边栏展示。
    """
    log_buffer = io.StringIO()
    t0 = time.time()
    with redirect_stdout(log_buffer):
        components = build_index(source_dir=source_dir)
    init_seconds = round(time.time() - t0, 1)
    build_log = log_buffer.getvalue()
    return components, build_log, init_seconds


def get_components(source_dir: str | None = None):
    """外层包装：如果初始化失败，展示错误信息并中止。"""
    if source_dir is None:
        source_dir = _get_source_root()
    components, build_log, init_seconds = _load_components_and_logs(source_dir)
    if components is None:
        st.error(f"索引初始化失败，请检查目录 `{source_dir}` 是否有 .py 文件、网络和 .env 配置。")
        with st.expander("查看启动日志"):
            st.text(build_log)
        st.stop()
    return components, build_log, init_seconds


# ============================================================
# RAG 问答流水线（完全复用 main.py 中的逻辑，仅不使用 print）
# ============================================================
def run_rag_pipeline(query: str, components: dict):
    """
    执行一次完整 RAG 问答，不打印、只返回结构化结果。
    与 main.answer_query 的算法/阈值/流程完全一致。
    """
    emb_client = components["emb_client"]
    multi = components["multi"]
    reranker = components["reranker"]
    llm = components["llm"]

    result = {"success": False, "error": None, "stats": {}, "top_n": [], "answer": ""}

    try:
        yield {"stage": "向量化你的问题（1/4）…", "progress": 0.15}
        query_vec = emb_client.get_embedding(query)

        yield {"stage": "多路召回中 · 语义 + 关键词（2/4）…", "progress": 0.35}
        candidates = multi.search(
            query_vector=query_vec,
            query_text=query,
            top_k=20,
        )
        result["stats"]["召回候选数"] = len(candidates)
        if not candidates:
            result["success"] = True
            result["error"] = "没有检索到相关代码片段，试试换个说法。"
            yield {"stage": "完成", "progress": 1.0, "done": True, "data": result}
            return

        yield {"stage": f"召回 {len(candidates)} 条，扩展上下文（2.5/4）…", "progress": 0.55}
        candidates = multi.expand_with_neighbors(candidates, window=2)
        result["stats"]["邻域扩充后总数"] = len(candidates)

        yield {"stage": "Cross-Encoder 精排 + 源文件兜底（3/4）…", "progress": 0.75}
        top_n = reranker.rerank(query, candidates, top_n=8)
        result["stats"]["重排有效结果"] = len(top_n) if top_n else 0
        if not top_n:
            result["success"] = True
            result["error"] = "重排后没有有效结果。"
            yield {"stage": "完成", "progress": 1.0, "done": True, "data": result}
            return
        top_n = multi.ensure_source_file_in_top_n(candidates, top_n, query, quota=2)

        yield {"stage": "大模型参考代码生成回答（4/4）…", "progress": 0.92}
        context_parts = []
        for i, r in enumerate(top_n, start=1):
            context_parts.append(
                f"[片段 {i}] 来源: {r['file_path']} (起始行: {r['start_line']})\n"
                f"{r['chunk_text']}"
            )
        context = "\n\n".join(context_parts)
        prompt = (
            f"你是一个代码助手。请根据下方代码片段回答用户的问题。\n"
            f"如果片段中没有相关信息，请如实说明。回答时请引用来源文件路径和行号。\n\n"
            f"===== 代码片段 =====\n"
            f"{context}\n\n"
            f"===== 用户问题 =====\n"
            f"{query}\n"
        )
        answer = llm.chat_completion([{"role": "user", "content": prompt}])

        result["success"] = True
        result["answer"] = answer
        result["top_n"] = top_n
        yield {"stage": "完成", "progress": 1.0, "done": True, "data": result}

    except EmbeddingError as e:
        result["error"] = f"查询向量化失败：{e}"
        yield {"stage": "出错", "progress": 1.0, "done": True, "data": result}
    except LLMRequestError as e:
        result["error"] = f"大模型调用失败：{e}"
        yield {"stage": "出错", "progress": 1.0, "done": True, "data": result}
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        yield {"stage": "出错", "progress": 1.0, "done": True, "data": result}


# ============================================================
# 侧边栏：展示索引状态、启动日志、代码库管理
# ============================================================
with st.sidebar:
    st.markdown("## 🤖 CodeRAG")
    st.caption("私有代码知识库 RAG 问答系统")
    st.divider()

    current_dir = _get_source_root()
    st.caption(f"📁 当前源码目录：`{current_dir}`")

    with st.status("正在加载 RAG 索引…", expanded=True) as status_box:
        components, build_log, init_seconds = get_components()
        chunks = components["chunks"]
        source_files = sorted({c["file_path"] for c in chunks})
        status_box.update(
            label=f"✅ 索引已就绪 · 耗时 {init_seconds}s",
            state="complete",
            expanded=False,
        )

    st.metric("索引代码文件", f"{len(source_files)} 个")
    st.metric("代码分片总数", f"{len(chunks)} 个")

    with st.expander("📋 查看启动日志"):
        st.text(build_log)

    # ---- 代码库管理 ----
    st.divider()
    st.markdown("### 📁 代码库管理")

    # 方式一：输入本地路径
    path_input = st.text_input(
        "本地代码目录路径",
        value=st.session_state.get("_custom_dir_input", ""),
        placeholder=r"例如：D:\my_project",
        help="输入本机磁盘上的代码目录，将从该目录读取 .py 文件重建索引",
    )

    col_path_btn, col_path_reset = st.columns(2)
    if col_path_btn.button("🔄 使用此目录重建", key="btn_rebuild_from_path", use_container_width=True):
        if not path_input.strip():
            st.warning("请先输入一个目录路径")
        elif not os.path.isdir(path_input.strip()):
            st.error(f"目录不存在：`{path_input.strip()}`")
        else:
            with st.spinner("正在重建索引…"):
                st.session_state["_current_source_dir"] = path_input.strip()
                st.session_state["_custom_dir_input"] = path_input.strip()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

    if col_path_reset.button("↩️ 恢复默认", key="btn_reset_default", use_container_width=True):
        st.session_state["_current_source_dir"] = _DEFAULT_SOURCE_ROOT
        st.session_state["_custom_dir_input"] = ""
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    # 方式二：上传 zip
    st.markdown("##### 或 上传 ZIP 压缩包")
    uploaded_zip = st.file_uploader(
        "上传包含 .py 文件的 ZIP 压缩包",
        type=["zip"],
        key="zip_uploader",
        help="ZIP 内的目录结构会原样保留，支持子文件夹",
    )

    if uploaded_zip is not None:
        st.caption(f"已上传：{uploaded_zip.name}（{uploaded_zip.size:,} 字节）")
        if st.button("📦 解压并重建索引", key="btn_extract_zip", use_container_width=True):
            with st.spinner("解压 ZIP 并重建索引…"):
                extract_dir = tempfile.mkdtemp(prefix="coderag_")
                try:
                    with zipfile.ZipFile(uploaded_zip) as zf:
                        zf.extractall(extract_dir)
                    st.session_state["_current_source_dir"] = extract_dir
                    st.session_state["_zip_extract_dir"] = extract_dir
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success(f"解压完成 → {extract_dir}")
                    st.rerun()
                except Exception as e:
                    st.error(f"解压失败：{e}")

    st.divider()
    st.caption("面向 Agent/AI 应用开发岗位的简历项目")
    st.caption("技术栈：Python · NumPy · BM25 · Cross-Encoder · MiMo")


# ============================================================
# 主区域：两个标签页（问答 / 源码浏览）
# ============================================================
st.markdown(
    """
<div style="padding: 1rem 1.25rem; border-radius: 6px;
            background-color: #252526;
            border: 1px solid #3c3c3c;">
  <div style="font-size: 1.3rem; font-weight: 600; color: #9cdcfe;">🤖 CodeRAG · 私有代码知识库问答</div>
  <div style="margin-top: 0.35rem; color: #cccccc; font-size: 0.95rem;">
    把源码喂进来，用自然语言提问。系统自动检索最相关的 3 段代码，
    并交给大模型生成带出处的回答。当前索引目录：<code style="background:#1e1e1e;border:1px solid #3c3c3c;padding:1px .4rem;border-radius:3px;color:#ce9178;">source_code/</code>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)
st.write("")

tab_qa, tab_browse = st.tabs(["💬 智能问答", "📂 源码浏览器"])


# ============================================================
# 标签页 1：智能问答
# ============================================================
EXAMPLES = [
    "react_scheduler 的 run 方法是干什么的？",
    "registry.py 的注册机制是怎么实现的？",
    "_build_system_prompt 作用是什么？",
]

with tab_qa:
    # ---- 一键示例问题 ----
    st.markdown("#### 💡 快速提问（点一下直接填入输入框）")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["_user_question"] = ex

    # ---- 输入框 ----
    st.markdown("#### 🔍 你的问题")
    default_value = st.session_state.get("_user_question", "")
    query = st.text_area(
        "输入框",
        value=default_value,
        height=110,
        label_visibility="collapsed",
        placeholder="例如：registry.py 中怎么拿到 TOOL_DEFINITIONS 变量？",
    )

    submit_col, _ = st.columns([1, 5])
    ask_button = submit_col.button("🚀 开始提问", type="primary", use_container_width=True)

    # ---- 执行提问（两阶段渲染：进度 → 结果，避免 React removeChild） ----
    pending_result = st.session_state.pop("_pending_result", None)

    if pending_result is not None:
        # 阶段 2：展示结果（独立渲染周期，无进度条/无 spinner，DOM 结构稳定）
        final_result = pending_result
        if final_result.get("error") or not final_result.get("success"):
            st.error(final_result.get("error") or "未知错误")
        else:
            st.markdown("#### 🤖 回答")
            answer_text = final_result.get("answer") or "（模型返回空内容）"
            st.info(answer_text)

            st.markdown("###### ")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            stats_col1.metric("召回候选", final_result["stats"].get("召回候选数", "-"))
            stats_col2.metric("邻域扩充后", final_result["stats"].get("邻域扩充后总数", "-"))
            stats_col3.metric("精排 TopN", final_result["stats"].get("重排有效结果", "-"))

            st.markdown("#### 📎 参考代码片段")
            top_n = final_result.get("top_n", [])
            if top_n:
                tab_labels = [
                    f"[{i+1}] {r['file_path']}:{r['start_line']}"
                    for i, r in enumerate(top_n)
                ]
                tabs = st.tabs(tab_labels)
                for i, (r, tab) in enumerate(zip(top_n, tabs)):
                    with tab:
                        meta_cols = st.columns([2, 1, 1])
                        meta_cols[0].markdown(
                            f"**来源：** `{r['file_path']}` · 起始行 **第 {r['start_line']} 行**"
                        )
                        meta_cols[1].markdown(
                            f"**相关度：** `{r.get('rerank_score', 0.0):+.4f}`"
                        )
                        tags = []
                        if r.get("boosted_by_source_filter"):
                            tags.append("🔧 源文件兜底提升")
                        if r.get("expanded_by_neighbor"):
                            tags.append("🔗 邻居扩充")
                        if r.get("chunk_type"):
                            tags.append(f"🏷 {r['chunk_type']}")
                        meta_cols[2].markdown(" · ".join(tags) if tags else "-")

                        lines = r["chunk_text"].splitlines()
                        st.code(r["chunk_text"], language="python")
                        st.caption(
                            f"📄 片段共 {len(lines)} 行 · "
                            f"起始第 {r['start_line']} 行 · "
                            f"来源: `{r['file_path']}`"
                        )

                        btns = st.columns([1.3, 1, 3.7])
                        btns[0].caption(f"片段共 {len(lines)} 行 · 起始第 {r['start_line']} 行")
                        open_key = f"open_src_{i}_{r['start_line']}"
                        if btns[1].button(
                            "📂 在源码浏览器打开",
                            key=open_key,
                            use_container_width=True,
                        ):
                            st.session_state["browse_file"] = r["file_path"]
                            st.session_state["browse_line"] = int(r["start_line"])
                            st.toast(
                                f"已切换到「源码浏览器」→ {r['file_path']}:{r['start_line']}",
                                icon="✅",
                            )
        if "_user_question" in st.session_state:
            del st.session_state["_user_question"]

    elif ask_button and query.strip():
        # 阶段 1：执行 RAG 流水线（只用进度条 + spinner，不混合 tabs）
        st.divider()
        st.markdown(f"**❓ 你的问题：** {query.strip()}")

        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        final_result = None

        with st.spinner("思考中，请稍候…"):
            for payload in run_rag_pipeline(query.strip(), components):
                progress_bar.progress(payload.get("progress", 0.0))
                if payload.get("done"):
                    final_result = payload["data"]
                    break
                progress_text.caption(f"⏳  {payload.get('stage', '')}")

        progress_bar.progress(1.0)

        if final_result is None:
            progress_bar.empty()
            progress_text.empty()
            st.error("流水线未返回结果，请重试。")
        else:
            # 把结果存入 session_state，然后触发一次 rerun
            # 下次渲染走"阶段 2"的分支，单独展示结果，避免混合 DOM
            progress_bar.empty()
            progress_text.empty()
            st.session_state["_pending_result"] = final_result
            if "_user_question" in st.session_state:
                del st.session_state["_user_question"]
            st.rerun()


# ============================================================
# 标签页 2：源码浏览器
# 布局：左侧"一层一层叠起来"的文件树（缩进线+卡片层叠感）
#      右侧：文件信息、跳行、视图模式、源码内容
# ============================================================
with tab_browse:
    source_root = _get_source_root()
    root_name = os.path.basename(os.path.normpath(source_root)) or "source_code"
    st.markdown(
        f"""
浏览 `{source_root}` 下被索引的任意 Python 源文件。
左侧是一层一层展开的文件树，点击文件即可查看源码；
也支持从「智能问答」的检索结果一键跳转到对应文件+起始行号。
"""
    )

    all_rel_files = _list_source_files(source_root)
    if not all_rel_files:
        st.warning(f"没有在 `{source_root}` 下找到任何 .py 文件。")
    else:
        source_tree = _build_source_tree(all_rel_files, root_name=root_name)

        # 初始化默认选中文件（兼容之前从问答页联动写进来的 browse_file）
        if "browse_file" not in st.session_state or st.session_state["browse_file"] not in all_rel_files:
            st.session_state["browse_file"] = all_rel_files[0]
        if "browse_line" not in st.session_state:
            st.session_state["browse_line"] = None

        # ---- 渲染一层一层的文件树（递归，深度最大 6 层足以）----
        def _render_tree_level(
            children: dict, level: int, parent_prefix: str = ""
        ) -> None:
            """
            渲染目录树的一层：
            - dir 节点用 st.expander 折叠（一层一层从外到内嵌进去，每层缩进+左边框）
            - file 节点直接是一个 Streamlit 原生按钮，aria-label 携带 "文件: ... @Lx 【已选中】..."
              信息给外面 CSS 做"伪装成列表行 + 缩进 + 激活背景"。完全不做 DOM 移位。
            """
            # 用 Streamlit 原生 markdown 包一层带缩进的卡片感容器
            level_class = f"level-{min(level, 4)}"
            st.markdown(
                f'<div class="tree-layer {level_class}">',
                unsafe_allow_html=True,
            )

            # 排序：文件夹在前，文件在后；各自内部按字母
            def _sort_key(item):
                name, node = item
                is_dir = 0 if node["__type__"] == "dir" else 1
                return (is_dir, name.lower())

            for name, node in sorted(children.items(), key=_sort_key):
                if node["__type__"] == "dir":
                    default_open = (level <= 0)
                    with st.expander(f"📁 {name}", expanded=default_open):
                        _render_tree_level(
                            node["__children__"],
                            level + 1,
                            f"{parent_prefix}/{name}" if parent_prefix else name,
                        )
                else:
                    path = node["path"]
                    is_active = (st.session_state.get("browse_file") == path)

                    # 关键修复：st.button 的 label 里放 HTML 会被 escape（Streamlit 安全策略），
                    # 所以按钮 label 只保留纯文本（emoji + 文件名），样式完全交给 CSS。
                    # 选中态用 Streamlit 原生 type="primary"，DOM 里会有 kind="primary" 属性，
                    # CSS 通过 [kind="primary"] 精准命中选中行。
                    indent_px = 6 + 18 * level  # 缩进由 Python 端精确算好，包一层外层 div
                    st.markdown(
                        f'<div class="tree-row-wrap" data-level="{level}" '
                        f'style="padding-left: {indent_px}px;">',
                        unsafe_allow_html=True,
                    )
                    # 按钮显示纯文本：🐍 前缀 + 文件名  + 末尾灰色路径（路径用 help 悬停）
                    clicked = st.button(
                        f"🐍  {name}",
                        key=f"tree__{path}__L{level}",
                        use_container_width=True,
                        help=f"打开：{path}",
                        type="primary" if is_active else "secondary",
                    )
                    # 路径小字：在按钮下方用 markdown 渲染（不进按钮 label 就不会被 escape）
                    st.markdown(
                        f'<div class="tree-file-path '
                        f'{"tree-file-path-active" if is_active else ""}">{path}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if clicked:
                        st.session_state["browse_file"] = path
                        st.session_state["browse_line"] = None
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # 左右分栏：左文件树 3.5 份宽度，右代码 6 份宽度
        col_tree, col_code = st.columns([3.2, 6.8])

        with col_tree:
            # 给整棵树包一个容器（面板灰底 + 细灰边）
            st.markdown('<div class="tree-container">', unsafe_allow_html=True)
            # 从 source_code 根开始渲染。注意 source_tree 最外层是 {"source_code": {...}}
            root_name, root_node = next(iter(source_tree.items()))
            with st.expander(f"📁 {root_name}", expanded=True):
                _render_tree_level(root_node["__children__"], level=0, parent_prefix=root_name)
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption(
                f"共收录 **{len(all_rel_files)}** 个 Python 文件。"
                "点击 📁 展开子文件夹，点击 🐍 文件打开源码。"
            )

        with col_code:
            selected_file = st.session_state["browse_file"]
            total_text = _read_source(selected_file, source_root)
            total_lines = len(total_text.splitlines()) if total_text else 0

            # ---- 第 1 行：正在查看文件（整行独占，不再和行号/视图模式挤同一栏）----
            st.markdown(
                f'<div style="background:#252526;border:1px solid #3c3c3c;border-radius:6px;'
                f'padding:6px 12px;margin-bottom:2px;">'
                f'<span style="color:#858585;font-size:.8rem;">📄 正在查看：</span>&nbsp;'
                f'<span style="color:#dcdcaa;font-family:Consolas,monospace;font-size:.95rem;word-break:break-all;">'
                f"{selected_file}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ---- 视图模式常量 & 普通 state 初始化
            # 注意：这里 browse_view_mode 是一个普通 session_state key，
            # 不再和任何 widget 的 key= 绑定（上面那个容易卡 bug 的 radio 已移除）
            # 所以下方按钮直接 st.session_state["browse_view_mode"] = xxx 不会冲突。
            VIEW_NEARBY = "只看目标行附近（约 60 行）"
            VIEW_FULL = "查看文件完整内容"
            if "browse_view_mode" not in st.session_state or st.session_state["browse_view_mode"] not in (VIEW_NEARBY, VIEW_FULL):
                st.session_state["browse_view_mode"] = VIEW_NEARBY
            view_mode = st.session_state["browse_view_mode"]

            # ---- 第 2 行：跳转到行号 + 当前视图模式徽章
            head_left, head_mid, head_right = st.columns([2.2, 3, 2])
            jump_line = head_left.number_input(
                "跳转到行号",
                min_value=1,
                max_value=max(1, total_lines),
                value=max(1, min(st.session_state.get("browse_line") or 1, max(1, total_lines))),
                step=1,
                key="browse_line_input",
                label_visibility="visible",
            )
            st.session_state["browse_line"] = jump_line

            # 徽章：显示当前视图模式（替代移除的 radio，让用户清楚当前是什么模式）
            is_nearby = (view_mode == VIEW_NEARBY)
            badge_bg = "#0e639c" if is_nearby else "#6a9955"
            badge_icon = "🎯" if is_nearby else "📖"
            badge_label = "附近视图（60 行）" if is_nearby else "完整视图"
            head_mid.markdown(
                f'<div style="margin-top:1.9rem;padding:4px 10px;border-radius:999px;'
                f'display:inline-flex;align-items:center;gap:.35rem;'
                f'background-color:{badge_bg};color:#ffffff;font-size:.82rem;font-weight:500;'
                f'box-shadow:0 0 0 1px rgba(255,255,255,.04) inset;">'
                f'<span>{badge_icon}</span><span>{badge_label}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            head_right.caption("")  # 占位，保持三栏对齐

            # 信息栏：总行数 / 大小 / 模式
            info_a, info_b, info_c = st.columns([2, 2, 2.2])
            info_a.caption(f"📏 文件总行数：**{total_lines}**")
            info_b.caption(f"💾 大小：**{len(total_text.encode('utf-8', errors='ignore')):,}** 字节")
            mode_tip = ("点下方『↩️ 切换到完整视图』看完整文件" if is_nearby else "点下方『🎯 切到附近视图』定位跳转行")
            info_c.caption(f"🧭 {mode_tip}")

            # ---- 渲染源码视图 ----
            if not total_text:
                st.error("文件读不出来，可能是编码问题或已被删除。")
            else:
                if is_nearby and jump_line:
                    local_text, local_start, hl_local = _build_line_view(
                        total_text, int(jump_line), context=30
                    )
                    st.code(local_text, language="python", line_numbers=True)
                    meta = st.columns([2, 1.4, 3])
                    meta[0].caption(
                        f"显示范围：第 {local_start} ~ {local_start + len(local_text.splitlines()) - 1} 行"
                    )
                    meta[1].caption(f"高亮：第 {jump_line} 行 = 局部第 {hl_local} 行")
                    if meta[2].button(
                        "↩️ 切换到完整视图",
                        key="to_full_view",
                        use_container_width=True,
                    ):
                        # browse_view_mode 已不再绑定任何 widget key，直接改安全
                        st.session_state["browse_view_mode"] = VIEW_FULL
                        st.rerun()
                else:
                    st.code(total_text, language="python", line_numbers=True)
                    if jump_line:
                        col_x, col_y = st.columns([3, 1])
                        col_x.caption(
                            f"💡 文件共 {total_lines} 行，当前跳转到第 {jump_line} 行。"
                            "文件太长时切到『附近视图』更方便定位。"
                        )
                        if col_y.button(
                            "🎯 切到附近视图",
                            key="to_nearby_view",
                            use_container_width=True,
                        ):
                            st.session_state["browse_view_mode"] = VIEW_NEARBY
                            st.rerun()

        # 底部：联动跳转说明
        st.divider()
        with st.expander("💡 小技巧：从问答页一键跳转过来"):
            st.markdown(
                """
在「💬 智能问答」标签页里，每个参考代码片段下方都有一个蓝色按钮 **「📂 在源码浏览器打开」**。
点一下就会自动切到本标签页、左侧文件树自动匹配、右侧定位到对应文件+起始行。面试时用来对比"检索片段 vs 完整源码上下文"特别好用。
"""
            )

# 源码浏览器的"卡片层叠"相关：不再用 DOM 移位 trick，改成给原生 st.button 套样式。
st.markdown(
    """
<style>
    /* 文件树里的 expander 面板去掉背景/边框，让 tree-layer 的缩进竖线显现出来 */
    div[class*="block-container"] .tree-container details {
        background-color: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0;
        margin: 0;
    }
    div[class*="block-container"] .tree-container summary {
        padding: 2px 4px;
        color: #cccccc !important;
        list-style: none;
    }
    div[class*="block-container"] .tree-container summary::-webkit-details-marker {
        display: none;
    }

    /* 文件条目按钮：Streamlit 原生按钮（type=secondary → 普通行；type=primary → 选中行）
       统一伪装成资源管理器列表行：透明底、无边框、左对齐，只保留 hover/选中高亮 */
    div[class*="block-container"] .tree-layer .tree-row-wrap button[kind="secondary"],
    div[class*="block-container"] .tree-layer .tree-row-wrap button[kind="primary"] {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        width: 100%;
        border: 0 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        padding: 3px 6px !important;
        margin: 2px 0 0 0 !important;
        text-align: left;
        justify-content: flex-start !important;
        line-height: 1.35;
        height: auto !important;
        min-height: 28px;
        font-size: 0.92rem;
    }
    /* 普通文件行（未选中）：透明底 + 灰字，hover 变浅灰 */
    div[class*="block-container"] .tree-layer .tree-row-wrap button[kind="secondary"] {
        background-color: transparent !important;
        color: #cccccc !important;
        font-weight: 500;
    }
    div[class*="block-container"] .tree-layer .tree-row-wrap button[kind="secondary"]:hover {
        background-color: #2a2d2e !important;
        color: #ffffff !important;
    }
    /* 选中文件行（primary）：VSCode 选中深蓝底 + 白字，bold 一点 */
    div[class*="block-container"] .tree-layer .tree-row-wrap button[kind="primary"] {
        background-color: #094771 !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    div[class*="block-container"] .tree-layer .tree-row-wrap button[kind="primary"]:hover {
        background-color: #0a548a !important;
    }
    /* 按钮里 emoji 🐍 单独给一点蓝色（按钮文字全改成灰/白后 emoji 没变）*/
    div[class*="block-container"] .tree-layer .tree-row-wrap button::first-letter {
        /* first-letter 对 emoji 可能没用，保持原样即可，视觉上够区分了 */
    }

    /* 缩进由 Python 端的外层 .tree-row-wrap 的 padding-left 精确控制，
       CSS 这里不再重复计算缩进；.tree-row-wrap 再叠一层小外边距让行与行之间呼吸一点 */
    div[class*="block-container"] .tree-layer .tree-row-wrap {
        margin-bottom: 1px;
    }

    /* 文件按钮下方的"灰色路径小字"（不进按钮 label，独立渲染，避免被 escape）*/
    div[class*="block-container"] .tree-layer .tree-row-wrap .tree-file-path {
        font-size: 0.76rem;
        color: #858585;
        padding: 1px 6px 2px 1.6rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.2;
    }
    /* 选中行下面的路径字也亮一点，和深蓝底协调 */
    div[class*="block-container"] .tree-layer .tree-row-wrap .tree-file-path-active {
        color: #c8c8c8;
    }
</style>
    """,
    unsafe_allow_html=True,
)
