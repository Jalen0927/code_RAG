"""
ingestion/file_loader.py — 源码文件加载器。

职责：
  - 递归扫描 source_code 目录，加载所有 .py 源码文件
  - 跳过 __pycache__ / .git / .pytest_cache 等无关目录
  - 把每个文件打包成 {file_path, content, start_line} 字典，供切片器使用
"""

import os

# 需要跳过的目录名（进入这些目录也没意义，全是缓存或配置）
SKIP_DIRS = {
    "__pycache__",    # Python 编译缓存目录
    ".pytest_cache",  # pytest 测试缓存目录
    ".git",           # Git 版本控制目录
    ".idea",          # PyCharm 配置目录
    ".vscode",        # VS Code 配置目录
    "node_modules",   # 前端依赖（以防万一）
}


def load_source_files(source_dir: str = "source_code") -> list:
    """
    遍历指定目录，加载所有 .py 源码文件。

    参数:
        source_dir: 要遍历的源码根目录，默认为 "source_code"

    返回:
        list[dict]: 每一项格式为
            {
                "file_path": str,    # 文件相对路径（相对当前工作目录）
                "content": str,      # 文件全部文本内容
                "start_line": int,   # 起始行号，整个文件作为一个整体时为 1
            }
    """
    documents = []

    # 如果根目录不存在，打印提示并返回空列表，避免后续报错
    if not os.path.exists(source_dir):
        print(f"[警告] 源码目录不存在: {source_dir}，已返回空列表")
        return documents

    # os.walk 会递归地遍历目录树
    # 每一轮循环返回三元组 (root, dirs, files)：
    #   root  —— 当前正在遍历的目录路径
    #   dirs  —— 当前目录下的所有子目录名列表
    #   files —— 当前目录下的所有文件名列表
    for root, dirs, files in os.walk(source_dir):

        # 关键技巧：原地修改 dirs 列表，可以阻止 os.walk 继续深入被过滤的目录
        # 同时跳过以 "." 开头的隐藏目录（如 .git、.env 等）
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for file_name in files:
            # 只处理 .py 后缀的文件，其余一律跳过
            if not file_name.endswith(".py"):
                continue

            # 跳过以 "." 开头的隐藏文件
            if file_name.startswith("."):
                continue

            # 拼接完整文件路径，转成相对路径方便后续溯源展示
            full_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(full_path)

            # 用 try-except 包裹文件读取，单个文件出错不会影响整体流程
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"[警告] 读取文件失败，已跳过: {rel_path} | 错误信息: {e}")
                continue

            # 整个文件作为一个文档项，起始行号记为 1
            # 后续交给切片器（chunker）时，再根据切片位置计算真实行号
            documents.append({
                "file_path": rel_path,
                "content": content,
                "start_line": 1,
            })

    return documents


# 当直接运行本文件时，执行一个简单的自测
if __name__ == "__main__":
    docs = load_source_files()
    print(f"\n共加载 {len(docs)} 个源码文件：")
    for doc in docs:
        line_count = doc["content"].count("\n") + 1
        print(f"  - {doc['file_path']}  (起始行: {doc['start_line']}, 总行数: {line_count})")
