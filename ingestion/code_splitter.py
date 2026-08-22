"""
ingestion/code_splitter.py — Python 源码感知切片器。

职责：
  - 把整文件源码切成大小合适的 chunk（默认上限 2000 字符）
  - 优先在 def/class 逻辑块边界切，避免在函数/类中间断开
  - 单个 class 太长时按方法边界切；单个函数太长时兜底滑动窗口 + 括号感知
  - 为每个 chunk 注入 metadata.chunk_type 元标签（class_def/function/import/log_call/other）
"""

import re

# ================ 可调参数常量 ================
# 单个 chunk 最大字符数（含重叠部分）
# 1500 → 2000：让 _build_system_prompt(1935字符) 等长方法不被滑动窗口硬切
MAX_CHARS_PER_CHUNK = 2000
# 相邻分片之间重叠的字符数（滑动窗口步长 = MAX - OVERLAP）
OVERLAP_CHARS = 200
# ============================================


# 匹配一行开头是 "def " 或 "class "（允许前面有任意缩进空格）
_DEF_OR_CLASS_RE = re.compile(r"^\s*(def|class)\s+\w+")

# 匹配一行开头是 "def "（用于 class 内部方法识别）
_DEF_ONLY_RE = re.compile(r"^\s*(def)\s+\w+")

# 三引号字符串标记
_TRIPLE_QUOTES = ('"""', "'''")

# ================ 元标签识别相关正则 ================
# 顶层 class 定义行（行首无缩进）：class ClassName(Base):
_CLASS_DEF_RE = re.compile(r"^class\s+\w+")
# 顶层或缩进的 def 定义行：def func(...) /    def method(...)
_ANY_DEF_RE = re.compile(r"^\s*def\s+\w+")
# import 行：import xxx / from xxx import yyy
_IMPORT_LINE_RE = re.compile(r"^\s*(import\s+\S|from\s+\S+\s+import\b)")


def _classify_chunk_type(chunk_text: str) -> str:
    """
    根据分片文本识别其代码类型，用于 multi_retriever 后处理加权。

    识别优先级（依次判断，命中即返回）：
      1. 文本中存在 class 定义行 → "class_def"
      2. 文本中存在 def 函数/方法定义行 → "function"
      3. 非 def/class 行中，import 语句占比 >= 70% → "import"
      4. 无赋值、无 return，且日志/print/纯调用行占比 >= 50% → "log_call"
      5. 否则 → "other"

    参数:
        chunk_text: 分片的完整文本

    返回:
        str: "class_def" | "function" | "import" | "log_call" | "other"
    """
    if not chunk_text or not chunk_text.strip():
        return "other"

    # 提取有效代码行（去掉空行和纯注释行）
    code_lines = []
    for line in chunk_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        code_lines.append(s)

    if not code_lines:
        return "other"

    total = len(code_lines)

    # —— 1. 优先级最高：包含 class 定义 → class_def ——
    for line in code_lines:
        if _CLASS_DEF_RE.match(line):
            return "class_def"

    # —— 2. 包含 def 定义 → function ——
    for line in code_lines:
        if _ANY_DEF_RE.match(line):
            return "function"

    # —— 3. import 语句占比 >= 70% → import ——
    import_count = sum(1 for line in code_lines if _IMPORT_LINE_RE.match(line))
    if total > 0 and import_count / total >= 0.7:
        return "import"

    # —— 4. log_call：无赋值、无 return，且主要是日志/print/纯调用 ——
    # 判断是否含赋值（排除 == != <= >= 和 from...import）
    def _is_assignment(line: str) -> bool:
        if "import" in line:
            return False
        # 简化判定：包含 = 但不包含 ==、!=、<=、>=
        for op in ("==", "!=", "<=", ">="):
            if op in line:
                return False
        return "=" in line

    has_assignment = any(_is_assignment(line) for line in code_lines)
    has_return = any(line.startswith("return") for line in code_lines)

    if not has_assignment and not has_return:
        # 统计日志/print/纯函数调用行
        def _is_call_line(line: str) -> bool:
            # print(...)/logger.xxx/logging.xxx/log.xxx 调用
            if (line.startswith("print(") or line.startswith("logger.")
                    or line.startswith("logging.") or line.startswith("log.")):
                return True
            # 行尾以 ) 结尾且包含 ( （粗略认为是函数调用语句）
            if line.endswith(")") and "(" in line:
                return True
            return False

        call_count = sum(1 for line in code_lines if _is_call_line(line))
        if total > 0 and call_count / total >= 0.5:
            return "log_call"

    # —— 5. 其他 → other ——
    return "other"


def _make_chunk(file_path: str, chunk_text: str, start_line: int) -> dict:
    """
    统一构造 chunk 字典，自动注入 metadata.chunk_type 元标签。

    参数:
        file_path:  源文件路径
        chunk_text: 分片文本
        start_line: 起始行号

    返回:
        dict: {
            "file_path": str,
            "chunk_text": str,
            "start_line": int,
            "metadata": {"chunk_type": str}
        }
    """
    return {
        "file_path": file_path,
        "chunk_text": chunk_text,
        "start_line": start_line,
        "metadata": {"chunk_type": _classify_chunk_type(chunk_text)},
    }


def _count_unclosed_triples(line: str) -> str | None:
    """
    检查一行中是否有未闭合的三引号。
    返回未闭合的三引号字符串，如果没有则返回 None。
    """
    for tq in _TRIPLE_QUOTES:
        count = line.count(tq)
        if count % 2 == 1:
            return tq
    return None


def _parse_line_blocks(content: str):
    """
    把源码按行拆成"逻辑块"列表。

    什么是逻辑块？
      - 顶层连续的普通代码行（import、赋值、print 等）算一块
      - 一个完整的 def 函数（含其内部所有缩进行）算一块
      - 一个完整的 class 类（含其内部所有缩进行）算一块

    返回: list[tuple]，每个元素是 (start_line, end_line, text)
        start_line: 该块在原文件中的起始行号（从 1 开始）
        end_line:   该块在原文件中的结束行号（包含）
        text:       该块的完整文本（含末尾换行）
    """
    lines = content.splitlines(keepends=True)
    blocks = []

    i = 0
    total_lines = len(lines)

    while i < total_lines:
        line = lines[i]
        stripped = line.strip()

        # 跳过纯空行，把它们拼到前一个块或下一个块的边界上
        if stripped == "":
            if blocks:
                prev_start, prev_end, prev_text = blocks.pop()
                prev_text += line
                blocks.append((prev_start, prev_end + 1, prev_text))
            i += 1
            continue

        # 判断当前行是不是 def/class 的定义行
        match = _DEF_OR_CLASS_RE.match(line)

        if match:
            # ========== 情况 A：这是函数/类的开头 ==========
            block_start_line = i + 1
            block_start_idx = i
            base_indent = len(line) - len(line.lstrip(" "))
            i += 1
            triple_quote = None

            while i < total_lines:
                cur_line = lines[i]
                cur_stripped = cur_line.strip()

                # --- 三引号字符串状态追踪 ---
                if triple_quote:
                    if triple_quote in cur_line:
                        count = cur_line.count(triple_quote)
                        if count % 2 == 1:
                            triple_quote = None
                    i += 1
                    continue
                else:
                    unclosed = _count_unclosed_triples(cur_line)
                    if unclosed:
                        triple_quote = unclosed
                        i += 1
                        continue

                if cur_stripped == "":
                    i += 1
                    continue
                cur_indent = len(cur_line) - len(cur_line.lstrip(" "))
                if cur_indent > base_indent:
                    i += 1
                else:
                    break

            block_text = "".join(lines[block_start_idx:i])
            block_end_line = i
            blocks.append((block_start_line, block_end_line, block_text))

        else:
            # ========== 情况 B：这是顶层普通代码 ==========
            block_start_line = i + 1
            block_start_idx = i
            i += 1
            triple_quote = None

            while i < total_lines:
                cur_line = lines[i]
                cur_stripped = cur_line.strip()

                # --- 三引号字符串状态追踪 ---
                if triple_quote:
                    if triple_quote in cur_line:
                        count = cur_line.count(triple_quote)
                        if count % 2 == 1:
                            triple_quote = None
                    i += 1
                    continue
                else:
                    unclosed = _count_unclosed_triples(cur_line)
                    if unclosed:
                        triple_quote = unclosed
                        i += 1
                        continue

                if cur_stripped == "":
                    i += 1
                    continue
                cur_indent = len(cur_line) - len(cur_line.lstrip(" "))
                if cur_indent == 0 and _DEF_OR_CLASS_RE.match(cur_line):
                    break
                i += 1

            block_text = "".join(lines[block_start_idx:i])
            block_end_line = i
            blocks.append((block_start_line, block_end_line, block_text))

    return blocks


def _split_class_by_methods(class_text: str, class_start_line: int,
                            max_chars: int, overlap: int):
    """
    对一个过长的 class，按其内部的 def 方法边界切分。

    策略:
      1. 先把 class 头部（class 定义行 + 紧接的 docstring + 类属性）作为独立分片
      2. 然后按 def 方法逐个拆分，每个方法独立成块
      3. 如果某个方法本身还是太长，再对它做滑动窗口切分
      4. class 头部 + 第一个方法可以合并在一个分片里（如果空间够）

    返回: list[dict]，每个元素 {"file_path": str, "chunk_text": str, "start_line": int}
          注意：此处还没有 file_path，调用方需要补上。
    """
    lines = class_text.splitlines(keepends=True)
    total_lines = len(lines)

    # --- 第 1 步：找出 class 内部所有 def 方法的位置 ---
    methods = []
    i = 0
    header_end_idx = 0
    # 外层循环也维护三引号状态，与 _parse_line_blocks 保持一致
    # 避免类属性三引号字符串内部的 "def xxx():" 行被误识别为真实方法
    triple_quote = None

    # 找第一个 def（带三引号状态追踪）
    while i < total_lines:
        line = lines[i]

        # --- 三引号字符串状态追踪 ---
        if triple_quote:
            if triple_quote in line:
                count = line.count(triple_quote)
                if count % 2 == 1:
                    triple_quote = None
            i += 1
            continue
        else:
            unclosed = _count_unclosed_triples(line)
            if unclosed:
                triple_quote = unclosed
                i += 1
                continue

        if _DEF_ONLY_RE.match(line):
            break
        i += 1
    header_end_idx = i

    # 如果整个 class 里没有 def（纯数据类），回退到滑动窗口
    if i >= total_lines:
        return None

    # 逐个提取方法
    # 装饰器行（@xxx）并入紧随其后的方法块，避免装饰器丢失
    while i < total_lines:
        line = lines[i]
        stripped = line.strip()

        # --- 三引号字符串状态追踪（方法之间的非 def 行）---
        if triple_quote:
            if triple_quote in line:
                count = line.count(triple_quote)
                if count % 2 == 1:
                    triple_quote = None
            i += 1
            continue
        else:
            unclosed = _count_unclosed_triples(line)
            if unclosed:
                triple_quote = unclosed
                i += 1
                continue

        if stripped == "":
            i += 1
            continue

        # 收集装饰器行，并入紧随其后的方法
        decorator_start_idx = None
        while stripped.startswith("@"):
            if decorator_start_idx is None:
                decorator_start_idx = i
            i += 1
            if i >= total_lines:
                break
            line = lines[i]
            stripped = line.strip()

        match = _DEF_ONLY_RE.match(line) if i < total_lines else None
        if match:
            # 方法块起始行回退到第一个装饰器（如果有），保留装饰器上下文
            method_start_idx = decorator_start_idx if decorator_start_idx is not None else i
            method_start_line = class_start_line + method_start_idx
            base_indent = len(line) - len(line.lstrip(" "))
            i += 1
            triple_quote = None

            while i < total_lines:
                cur_line = lines[i]
                cur_stripped = cur_line.strip()

                if triple_quote:
                    if triple_quote in cur_line:
                        count = cur_line.count(triple_quote)
                        if count % 2 == 1:
                            triple_quote = None
                    i += 1
                    continue
                else:
                    unclosed = _count_unclosed_triples(cur_line)
                    if unclosed:
                        triple_quote = unclosed
                        i += 1
                        continue

                if cur_stripped == "":
                    i += 1
                    continue
                cur_indent = len(cur_line) - len(cur_line.lstrip(" "))
                if cur_indent > base_indent:
                    i += 1
                else:
                    break

            method_text = "".join(lines[method_start_idx:i])
            method_end_line = class_start_line + i
            methods.append((method_start_line, method_end_line, method_text))
        else:
            # 装饰器后不是 def（语法异常或装饰器后是 class），跳过
            i += 1

    if not methods:
        return None

    # --- 第 2 步：组装分片 ---
    chunks = []

    # 2a: class 头部
    header_text = "".join(lines[:header_end_idx])
    header_len = len(header_text)

    if header_len > max_chars:
        header_chunks = _split_with_overlap(header_text, "", class_start_line,
                                            max_chars, overlap)
        chunks.extend(header_chunks)
        header_text = ""
        header_len = 0

    # 2b: 把头部和第一个方法尽量合并
    current_text = header_text
    current_start_line = class_start_line
    current_len = header_len

    for (m_start, m_end, m_text) in methods:
        m_len = len(m_text)

        if m_len > max_chars:
            if current_text:
                chunks.append(_make_chunk("", current_text, current_start_line))
                current_text = ""
                current_start_line = None
                current_len = 0

            sub_chunks = _split_with_overlap(m_text, "", m_start,
                                              max_chars, overlap)
            chunks.extend(sub_chunks)
            continue

        if current_len + m_len > max_chars and current_text:
            chunks.append(_make_chunk("", current_text, current_start_line))
            current_text = ""
            current_start_line = None
            current_len = 0

        if not current_text:
            current_start_line = m_start
        current_text += m_text
        current_len += m_len

    if current_text:
        chunks.append(_make_chunk("", current_text, current_start_line))

    return chunks


def _find_safe_split_pos(full_text: str, target_pos: int) -> int:
    """
    在 target_pos 附近找一个"安全切分点"，避免切在括号内部。

    安全切分点的定义：
      - 括号深度为 0 的位置（不在任何 [...] {...} (...) 内部）
      - 或者在括号内时，往前找最近的 ",\\n"（逗号+换行，列表元素边界）

    这样长列表 TOOL_DEFINITIONS = [...] 不会被硬切在字典中间，
    而是在列表元素边界切分，保证每个分片至少包含完整的列表元素。

    返回：
      - 如果 target_pos 不在括号内，直接返回 target_pos
      - 如果在括号内，返回往前找的最近安全点（找不到则 fallback 到 target_pos）
    """
    depth = 0           # 当前括号深度（只计 [ 和 {，不计 ()）
    in_str = False      # 是否在字符串内
    str_char = None     # 当前字符串的引号类型
    last_safe_pos = 0   # 记录扫描过程中最近的安全切分点

    # 一次性扫描 full_text[:target_pos]，同时计算深度和记录安全点
    scan_end = min(target_pos, len(full_text))
    for i in range(scan_end):
        ch = full_text[i]
        if in_str:
            # 字符串内：处理转义和闭合
            if ch == '\\':
                # 转义符，跳过下一个字符
                continue
            if ch == str_char:
                in_str = False
        else:
            if ch in ('"', "'"):
                # 进入字符串
                in_str = True
                str_char = ch
            elif ch in '[{':
                # 只计列表 [] 和字典 {} 的深度，不计函数调用 ()
                # 因为 () 跨度小（如 State(...)），不需要调整切分点
                depth += 1
            elif ch in ']}':
                depth = max(0, depth - 1)
                # 括号闭合后是安全切分点
                if depth == 0:
                    last_safe_pos = i + 1
            # 在括号内遇到 ",\n"（逗号+换行），也是相对安全的切分点
            # （列表元素边界，切出来至少是完整的字典/列表项）
            elif i >= 1 and full_text[i - 1] == ',' and ch == '\n' and depth > 0:
                last_safe_pos = i + 1

    # target_pos 不在括号内，直接返回
    if depth <= 0:
        return target_pos

    # 在括号内，返回往前找的最近安全点；找不到就 fallback
    return last_safe_pos if last_safe_pos > 0 else target_pos


def _split_with_overlap(full_text: str, file_path: str, start_line: int,
                        max_chars: int, overlap: int):
    """
    对"太长的单个逻辑块"做普通滑动窗口切分（兜底方案）。

    带括号感知：切分点会自动避开括号内部，
    长列表 TOOL_DEFINITIONS=[...] / 长字典 TOOL_FUNC_MAPPING={...}
    不会被硬切在元素中间，而是在元素边界切分。
    """
    chunks = []
    step = max_chars - overlap
    if step <= 0:
        step = max_chars // 2

    pos = 0
    total_len = len(full_text)

    while pos < total_len:
        chunk_text = full_text[pos:pos + max_chars]
        chunk_start_line = start_line + full_text.count("\n", 0, pos)
        chunks.append(_make_chunk(file_path, chunk_text, chunk_start_line))

        # 计算下一个切分点
        next_pos = pos + step
        # 括号感知：如果 next_pos 在括号内，往前找安全点
        next_pos = _find_safe_split_pos(full_text, next_pos)
        # 防止死循环：如果安全点没动（极端情况），强制往前走一步
        if next_pos <= pos:
            next_pos = pos + step
        pos = next_pos
        if pos >= total_len:
            break

    return chunks


def split_python_file(doc: dict,
                      max_chars: int = MAX_CHARS_PER_CHUNK,
                      overlap: int = OVERLAP_CHARS) -> list:
    """
    对一个源码文件做"代码感知"切片。
    """
    if not isinstance(doc, dict):
        print(f"[警告] split_python_file 收到了非字典输入，已跳过: {type(doc)}")
        return []
    file_path = doc.get("file_path", "unknown")
    content = doc.get("content", "")
    base_start_line = doc.get("start_line", 1)

    if not content:
        return []

    if len(content) <= max_chars:
        # 短文件直接成单 chunk，注入 metadata.chunk_type
        return [_make_chunk(file_path, content, base_start_line)]

    try:
        line_blocks = _parse_line_blocks(content)
    except Exception as e:
        print(f"[警告] 解析逻辑块失败 ({file_path})，回退为普通切片: {e}")
        return _split_with_overlap(content, file_path, base_start_line,
                                   max_chars, overlap)

    chunks = []
    current_blocks = []
    current_length = 0
    current_start_line = None

    def _flush_current():
        """把当前累积的 blocks 合并成一个 chunk 并清空缓冲区。"""
        nonlocal current_blocks, current_length, current_start_line
        if not current_blocks:
            return
        chunk_text = "".join(blk[1] for blk in current_blocks)
        chunks.append(_make_chunk(
            file_path, chunk_text,
            current_blocks[0][0] + (base_start_line - 1),
        ))
        current_blocks = []
        current_length = 0
        current_start_line = None

    for (blk_start, blk_end, blk_text) in line_blocks:
        blk_len = len(blk_text)

        if blk_len > max_chars:
            _flush_current()

            # 判断是否是 class 块，是的话按方法切
            is_class_block = False
            first_line = blk_text.split("\n", 1)[0].strip() if blk_text else ""
            if first_line.startswith("class "):
                is_class_block = True

            if is_class_block:
                sub_chunks = _split_class_by_methods(
                    blk_text, blk_start + (base_start_line - 1),
                    max_chars, overlap,
                )
                if sub_chunks is not None:
                    for sc in sub_chunks:
                        sc["file_path"] = file_path
                    chunks.extend(sub_chunks)
                    continue

            # 非 class 块或 class 按方法切失败，回退滑动窗口
            sub_chunks = _split_with_overlap(
                blk_text, file_path,
                start_line=blk_start + (base_start_line - 1),
                max_chars=max_chars, overlap=overlap,
            )
            chunks.extend(sub_chunks)
            continue

        if current_length + blk_len > max_chars and current_blocks:
            # 当前 chunk 满了，保留尾部 overlap 个字符作为下一个 chunk 的开头
            overlap_texts = []
            overlap_len = 0
            for prev_blk in reversed(current_blocks):
                prev_blk_start, prev_blk_text = prev_blk[0], prev_blk[1]
                if overlap_len + len(prev_blk_text) <= overlap:
                    overlap_texts.insert(0, (prev_blk_start, prev_blk_text))
                    overlap_len += len(prev_blk_text)
                else:
                    break
            _flush_current()
            if overlap_texts:
                current_blocks = overlap_texts[:]
                current_length = overlap_len
                current_start_line = overlap_texts[0][0]

        if current_start_line is None:
            current_start_line = blk_start
        current_blocks.append((blk_start, blk_text))
        current_length += blk_len

    _flush_current()

    return chunks


def split_python_files(doc_list: list,
                       max_chars: int = MAX_CHARS_PER_CHUNK,
                       overlap: int = OVERLAP_CHARS) -> list:
    """
    批量切分多个文件。
    """
    all_chunks = []
    for doc in doc_list:
        try:
            chunks = split_python_file(doc, max_chars, overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            fp = doc.get("file_path", "unknown") if isinstance(doc, dict) else "unknown"
            print(f"[警告] 切分文件失败，已跳过: {fp} | 错误: {e}")
    return all_chunks


# ================= 自测入口 =================
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.file_loader import load_source_files

    docs = load_source_files()
    print(f"共读取 {len(docs)} 个源码文件")
    all_chunks = split_python_files(docs)
    print(f"共切出 {len(all_chunks)} 个分片")
    print("-" * 60)
    for idx, c in enumerate(all_chunks):
        lines = c["chunk_text"].count("\n") + 1
        char_len = len(c["chunk_text"])
        print(f"[{idx+1:>3}] {c['file_path']}:{c['start_line']}  "
              f"字符数={char_len:<5}  行数={lines}  "
              f"preview={c['chunk_text'][:60].replace(chr(10),' ')}")
