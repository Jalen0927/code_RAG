"""调用 main.py 的交互式链路，验证切片修复后的 RAG 效果"""
import sys
import os

# Bug 4 修复：确保 Windows 终端能正确显示 emoji（GBK 终端兜底）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 项目根目录是 test 目录的上一级，加入搜索路径以便 import 根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_index, answer_query

TEST_QUESTIONS = [
    "ReActScheduler 的初始化逻辑是什么？",
    "ReActScheduler 的 run 方法做了什么？",
    "_build_system_prompt 方法的作用是什么？",
    "ReActScheduler 有哪些方法？",
]

print("\n" + "=" * 60)
print("🚀 启动 CodeRAG 全链路验证（通过 main.py）")
print("=" * 60)

components = build_index()
if components is None:
    print("❌ 索引构建失败，退出")
    sys.exit(1)

for i, q in enumerate(TEST_QUESTIONS, 1):
    print(f"\n{'='*60}")
    print(f"📝 测试问题 {i}/{len(TEST_QUESTIONS)}: {q}")
    print("=" * 60)
    answer_query(q, components)

print("\n" + "=" * 60)
print("✅ 全链路验证完成！")
print("=" * 60)
