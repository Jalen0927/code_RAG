"""
chunk_viewer.py — 代码分片质量查看调试工具。

职责：
  - 接收一个 .py 源码文件路径，调用项目的 code_splitter 切片函数
  - 输出每个分片的序号、长度、起止行号、是否是断点、开头/末尾预览
  - 最后统计断点率，帮助判断切片质量是否良好

使用方法：
    python chunk_viewer.py <源码文件路径>

示例：
    python chunk_viewer.py source_code/file_agent/react_scheduler.py
    python chunk_viewer.py source_code/file_agent/registry.py
"""
import os
import sys

# Windows 控制台默认 GBK 编码，无法打印 emoji（📄❌✅），
# 这里把 stdout 重新配置成 utf-8，避免 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 把项目根目录加入 sys.path，让脚本能 import 项目内的模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入项目的切片函数
from ingestion.code_splitter import split_python_file


# ============ 工具函数 ============

def read_source_file(file_path: str) -> str:
    """
    读取源码文件全部内容。
    用 utf-8 编码读取，避免 Windows 中文乱码。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def is_breakpoint(chunk_text: str) -> bool:
    """
    判断这个分片是否是"断点"——即切割在了 def/class 的内部。

    判断逻辑：找到分片里第一个非空行，看它的特征：
      - 如果以 def / class / import / from / @ / # 开头 → 不是断点
        （说明这是一个新语句的开始，分片从语句边界开始）
      - 如果行首是空格（缩进行）→ 进一步看内容：
          - 以 { } [ ] " 开头 → 不是断点
            （列表/字典元素边界，code_splitter 的括号感知已对齐到这里）
          - 其他缩进行 → 是断点（切在函数/类语句中间）
      - 其他情况（如顶层变量赋值 TOOL_DEFINITIONS = [...]）→ 不是断点
    """
    # 把分片按行拆开，逐行检查
    for line in chunk_text.split("\n"):
        stripped = line.strip()  # 去掉首尾空白，看实际内容
        if not stripped:  # 跳过空行
            continue
        # 找到了第一个非空行，开始判断
        # 情况1：以这些关键字开头 → 新语句开始，不是断点
        if stripped.startswith(("def ", "class ", "import ", "from ",
                                 "@", "#", "if __name__")):
            return False
        # 情况2：行首是空格（缩进行）
        if line[0] == " ":
            # 列表/字典元素边界（{,},[,",]开头）不算断点
            # 因为 code_splitter 的括号感知会把切分点对齐到 },\n 后
            if stripped[0] in ('{', '}', '[', ']', '"'):
                return False
            # 其他缩进行（如 print/if/return 中间）→ 是断点
            return True
        # 情况3：顶层赋值语句（如 TOOL_DEFINITIONS = [...]）→ 不是断点
        return False
    # 分片全是空行（极端情况），不算断点
    return False


def get_end_line(start_line: int, chunk_text: str) -> int:
    """根据起始行号和分片文本里的换行符数量，算出终止行号。"""
    return start_line + chunk_text.count("\n")


def truncate(text: str, max_len: int = 80) -> str:
    """
    截取文本的前 max_len 个字符，并把换行符替换成 ↵，
    方便在一行里打印预览，不会被换行打乱排版。
    """
    snippet = text[:max_len]
    snippet = snippet.replace("\n", "↵")
    if len(text) > max_len:
        snippet += "..."
    return snippet


def truncate_tail(text: str, max_len: int = 80) -> str:
    """截取文本的末尾 max_len 个字符（和 truncate 相反，从尾部取）。"""
    if len(text) <= max_len:
        snippet = text
    else:
        snippet = text[-max_len:]
    snippet = snippet.replace("\n", "↵")
    return snippet


# ============ 主流程函数 ============

def view_chunks(file_path: str):
    """
    主流程：读取文件 → 调用切片 → 逐个输出分片信息 → 统计断点率。
    """
    # ---------- 第 1 步：读取源码 ----------
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    content = read_source_file(file_path)
    file_size = len(content)
    total_lines = content.count("\n") + 1
    print(f"\n{'=' * 70}")
    print(f"📄 文件: {file_path}")
    print(f"   总字符数: {file_size}  |  总行数: {total_lines}")
    print(f"{'=' * 70}")

    # ---------- 第 2 步：调用项目切片函数 ----------
    # 构造 doc 字典，格式和 file_loader.load_source_files 返回的一致
    doc = {
        "file_path": file_path,
        "content": content,
        "start_line": 1,
    }
    chunks = split_python_file(doc)
    print(f"🔪 切片完成，共得到 {len(chunks)} 个分片\n")

    # ---------- 第 3 步：逐个输出分片信息 ----------
    breakpoint_count = 0  # 断点计数器

    for i, chunk in enumerate(chunks):
        chunk_text = chunk["chunk_text"]
        start_line = chunk["start_line"]
        end_line = get_end_line(start_line, chunk_text)
        char_len = len(chunk_text)
        # 调用 is_breakpoint 判断是否切割在函数/类内部
        is_bp = is_breakpoint(chunk_text)
        if is_bp:
            breakpoint_count += 1

        # chunk_type 从 metadata 里取（code_splitter 会写入这个字段）
        chunk_type = chunk.get("metadata", {}).get("chunk_type", "?")

        # 标记：断点用 ❌，非断点用 ✅
        bp_tag = "❌ 断点" if is_bp else "✅ 正常"

        # 打印分片头部信息
        print(f"{'-' * 70}")
        print(f"  分片 [{i + 1}/{len(chunks)}]  {bp_tag}  类型={chunk_type}")
        print(f"  字符长度: {char_len}  |  起止行: {start_line} → {end_line}")

        # 打印开头 80 字符预览
        head_preview = truncate(chunk_text, 80)
        print(f"  开头80字: {head_preview}")

        # 打印末尾 80 字符预览
        tail_preview = truncate_tail(chunk_text, 80)
        print(f"  末尾80字: {tail_preview}")

    # ---------- 第 4 步：统计断点率 ----------
    total_chunks = len(chunks)
    print(f"\n{'=' * 70}")
    print(f"📊 切片质量统计")
    print(f"{'=' * 70}")
    print(f"  总分片数量:  {total_chunks}")
    print(f"  断点分片数:  {breakpoint_count}")
    if total_chunks > 0:
        bp_rate = breakpoint_count / total_chunks * 100
        print(f"  断点率:      {bp_rate:.1f}%")
    else:
        print(f"  断点率:      N/A（无分片）")

    # 给出一个简单的质量评价
    print()
    if total_chunks == 0:
        print("  评价: ⚠️ 文件为空，未产生分片")
    elif breakpoint_count == 0:
        print("  评价: ✅ 切片质量优秀，所有分片都从语句边界开始")
    elif breakpoint_count <= total_chunks * 0.2:
        print("  评价: 🟡 切片质量良好，少量断点可接受")
    else:
        print("  评价: ❌ 切片质量较差，断点较多，建议调整切片参数")


# ============ 程序入口 ============

if __name__ == "__main__":
    # 命令行参数检查：必须传入一个文件路径
    if len(sys.argv) < 2:
        print("用法: python chunk_viewer.py <源码文件路径>")
        print("示例: python chunk_viewer.py source_code/file_agent/react_scheduler.py")
        sys.exit(1)

    target_file = sys.argv[1]
    view_chunks(target_file)
