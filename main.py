"""
main.py — CodeRAG 交互式主入口。

职责：
  - 程序启动时一次性完成全部索引初始化（读取源码、切片、向量化、建索引、
    初始化多路召回和重排器），之后持续接收用户输入提问。
  - 每次提问完整跑通整套 RAG 链路：向量化查询 → 多路召回 → 邻域扩充 →
    重排 → 名额预留 → LLM 生成。

运行方式：
    python main.py
"""

import os
import sys

# 确保 Windows 终端能正确显示 emoji（GBK 终端兜底）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 确保项目根目录在搜索路径里（兼容不同启动位置）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 项目内部模块导入
from ingestion.file_loader import load_source_files
from ingestion.code_splitter import split_python_files
from embedding_client import EmbeddingClient, EmbeddingError
from retriever.vector_store import InMemoryVectorStore
from retriever.bm25_retriever import Bm25Retriever
from retriever.multi_retriever import MultiRetriever
from retriever.reranker import Reranker
from llm_client import MiMoClient, LLMRequestError
from config import EMBEDDING_DIMENSIONS


# ============================================================
# 一、索引初始化（程序启动时一次性完成）
# ============================================================

def build_index(source_dir: str = "source_code"):
    """
    一次性完成全部索引初始化：读源码 → 切片 → 向量化 → 建向量库 → 建 BM25 索引
    → 初始化多路召回 → 初始化重排器 → 初始化 LLM 客户端。

    参数:
        source_dir: 源码根目录，默认 "source_code"。可动态切换到任意本地路径。

    返回:
        dict: 包含所有已初始化组件的字典，供后续查询使用。
              如果初始化失败，返回 None。
    """
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          🤖 CodeRAG  代码智能问答助手 启动中...          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print(f"📁 源码目录: {source_dir}")

    # ------------------------------------------------------------
    # 1. 读取源码文件
    # ------------------------------------------------------------
    print("📂 [1/8] 扫描源码文件中...")
    try:
        docs = load_source_files(source_dir)
        print(f"       ✅ 已读取 {len(docs)} 个源码文件")
        if not docs:
            print("       ⚠️  没有读到任何源码文件，流程终止")
            return None
    except Exception as e:
        print(f"       ❌ 读取源码失败: {e}")
        return None

    # ------------------------------------------------------------
    # 2. 代码感知分片
    # ------------------------------------------------------------
    print(f"\n✂️  [2/8] 代码感知切片中...")
    try:
        chunks = split_python_files(docs)
        print(f"       ✅ 切分得到 {len(chunks)} 个代码分片")
        if not chunks:
            print("       ⚠️  没有得到任何分片，流程终止")
            return None
    except Exception as e:
        print(f"       ❌ 代码分片失败: {e}")
        return None

    # ------------------------------------------------------------
    # 3. 批量向量化 + 存入向量库
    # ------------------------------------------------------------
    print(f"\n🧠 [3/8] 向量化分片并存入向量库...")
    try:
        emb_client = EmbeddingClient()
        texts = [c["chunk_text"] for c in chunks]
        print(f"       📐 正在向量化 {len(texts)} 段文本（每批最多 20 条）...")
        vectors = emb_client.get_batch_embedding(texts)
        print(f"       ✅ 得到 {len(vectors)} 个向量，维度={len(vectors[0])}")

        vector_store = InMemoryVectorStore(embedding_dim=EMBEDDING_DIMENSIONS)
        added = vector_store.add(vectors, chunks)
        print(f"       💾 向量库已存储 {added} 条记录")
    except EmbeddingError as e:
        print(f"       ❌ 向量化失败: {e}")
        return None
    except Exception as e:
        print(f"       ❌ 向量库构建失败: {type(e).__name__}: {e}")
        return None

    # ------------------------------------------------------------
    # 4. 构建 BM25 检索索引
    # ------------------------------------------------------------
    print(f"\n🔎 [4/8] 构建 BM25 关键词索引...")
    try:
        bm25 = Bm25Retriever()
        n_indexed = bm25.build_index(chunks)
        print(f"       ✅ BM25 索引就绪，共索引 {n_indexed} 个分片")
    except Exception as e:
        print(f"       ❌ BM25 索引构建失败: {type(e).__name__}: {e}")
        return None

    # ------------------------------------------------------------
    # 5. 初始化多路召回融合器
    # ------------------------------------------------------------
    print(f"\n🔗 [5/8] 初始化多路召回融合器...")
    try:
        multi = MultiRetriever(vector_store, bm25, all_chunks=chunks)
        print(f"       ✅ 向量召回 + BM25 召回 双路融合就绪")
    except Exception as e:
        print(f"       ❌ MultiRetriever 初始化失败: {type(e).__name__}: {e}")
        return None

    # ------------------------------------------------------------
    # 6. 初始化 Reranker 重排器（懒加载，模型在首次 rerank 时才加载）
    # ------------------------------------------------------------
    print(f"\n🎯 [6/8] 初始化 Cross-Encoder 重排器...")
    try:
        reranker = Reranker()
        print(f"       ✅ 重排器就绪（精排模型首次使用时加载）")
    except Exception as e:
        print(f"       ❌ Reranker 初始化失败: {type(e).__name__}: {e}")
        return None

    # ------------------------------------------------------------
    # 7. 初始化 LLM 客户端
    # ------------------------------------------------------------
    print(f"\n💬 [7/8] 初始化大模型客户端...")
    try:
        llm = MiMoClient()
        print(f"       ✅ MiMoClient 就绪")
    except Exception as e:
        print(f"       ❌ LLM 客户端初始化失败: {type(e).__name__}: {e}")
        return None

    # ------------------------------------------------------------
    # 8. 收集所有组件
    # ------------------------------------------------------------
    components = {
        "emb_client": emb_client,
        "vector_store": vector_store,
        "bm25": bm25,
        "multi": multi,
        "reranker": reranker,
        "llm": llm,
        "chunks": chunks,
    }

    print(f"\n🎊 [8/8] 全部初始化完成！共索引 {len(chunks)} 个代码分片")
    print("=" * 60)
    print("  🚀 索引已就绪，随时可以开始提问！")
    print("  💡 输入你的问题，按回车提交")
    print("  👋 输入 exit / quit / q 退出")
    print("=" * 60)

    return components


# ============================================================
# 二、单次提问的完整 RAG 链路
# ============================================================

def answer_query(query: str, components: dict) -> bool:
    """
    对单次用户提问完整跑通 RAG 链路。

    流程:
        向量化查询 → 多路召回 → 邻域扩充 → 重排 → 名额预留 → 拼 Prompt → LLM 生成

    参数:
        query: 用户问题文本
        components: build_index() 返回的组件字典

    返回:
        True 表示正常处理完，False 表示出现不可恢复的错误
    """
    emb_client = components["emb_client"]
    multi = components["multi"]
    reranker = components["reranker"]
    llm = components["llm"]

    try:
        # ------------------------------------------------------------
        # 步骤 1：向量化用户问题
        # ------------------------------------------------------------
        print("\n  🔹 [1/4] 向量化你的问题...")
        query_vec = emb_client.get_embedding(query)

        # ------------------------------------------------------------
        # 步骤 2：多路召回（拉 20 个候选给重排留余量）
        # ------------------------------------------------------------
        print("  🔹 [2/4] 多路召回中（语义 + 关键词）...")
        candidates = multi.search(
            query_vector=query_vec,
            query_text=query,
            top_k=20,
        )
        print(f"       找到 {len(candidates)} 个候选代码片段")

        if not candidates:
            print("\n  🤔 抱歉，知识库中没有找到相关的代码片段。")
            print("  💡 尝试换一种说法，或者检查关键词是否正确。")
            return True

        # ------------------------------------------------------------
        # 步骤 2.5：同文件邻域扩充（多路召回之后、rerank 之前）
        # 把每个候选分片的前后 2 个邻居分片追加进候选池，
        # 让 reranker 看到完整上下文，由重排模型决定哪些分片有用
        # ------------------------------------------------------------
        candidates = multi.expand_with_neighbors(candidates, window=2)
        print(f"       扩充后候选池: {len(candidates)} 条")

        # ------------------------------------------------------------
        # 步骤 3：Cross-Encoder 重排，取 Top 8
        # ------------------------------------------------------------
        print("  🔹 [3/4] Cross-Encoder 精细重排中...")
        top_n = reranker.rerank(query, candidates, top_n=8)
        if not top_n:
            print("\n  🤔 重排后没有有效结果。")
            return True

        # 后处理：保证 Top N 中至少有 2 个"含关键术语的源文件分片"
        # 解决 rerank 模型偏向测试文件 docstring 的问题
        top_n = multi.ensure_source_file_in_top_n(candidates, top_n, query, quota=2)

        # 打印重排结果（带重排分数）
        print(f"       🎯 精选 Top {len(top_n)} 最相关片段:")
        for i, r in enumerate(top_n):
            boost_tag = " [源文件提升]" if r.get("boosted_by_source_filter") else ""
            print(f"          [{i+1}] 评分 {r['rerank_score']:+.4f}  "
                  f"📄 {r['file_path']}:{r['start_line']}{boost_tag}")

        # ------------------------------------------------------------
        # 步骤 4：组装 Prompt 调用大模型
        # ------------------------------------------------------------
        print("  🔹 [4/4] 正在思考并生成回答...")

        # 组装 context：每个片段带文件路径 + 起始行号
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

        messages = [{"role": "user", "content": prompt}]
        answer = llm.chat_completion(messages)

        # ------------------------------------------------------------
        # 输出结果：回答 + 引用来源
        # ------------------------------------------------------------
        print("\n" + "=" * 60)
        print("  ❓ 你的问题:")
        print(f"  {query}")
        print("\n" + "-" * 60)
        print("  🤖 CodeRAG 回答:")
        print(f"  {answer}")
        print("\n" + "-" * 60)
        print("  📎 参考来源:")
        for i, r in enumerate(top_n, start=1):
            print(f"  [{i}] 📄 {r['file_path']}  第 {r['start_line']} 行  "
                  f"相关度: {r['rerank_score']:+.4f}")
            # 把片段前 2 行打印出来，方便用户核对
            first_two_lines = r["chunk_text"].split("\n")[:2]
            for line in first_two_lines:
                print(f"      ┊ {line}")
        print("=" * 60)

        return True

    except EmbeddingError as e:
        print(f"\n  ❌ 查询向量化失败: {e}")
    except LLMRequestError as e:
        print(f"\n  ❌ 大模型调用失败: {e}")
    except KeyboardInterrupt:
        print("\n  ⏹ 已中断当前提问")
    except Exception as e:
        print(f"\n  ❌ 处理提问时发生错误: {type(e).__name__}: {e}")

    return True  # 单次提问出错不影响后续提问


# ============================================================
# 三、交互式主循环
# ============================================================

def interactive_loop(components: dict):
    """
    交互式主循环：持续接收用户输入提问，输入 exit 退出。

    参数:
        components: build_index() 返回的组件字典
    """
    print("\n" + "=" * 60)
    print("  💬 进入对话模式，随时向我提问吧！")
    print("  🌙 输入 exit / quit / q 退出")
    print("=" * 60)

    while True:
        try:
            # 接收用户输入
            query = input("\n👉 请输入你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D（Linux）或 Ctrl+Z（Windows）或 Ctrl+C
            print("\n\n  👋 再见！期待下次和你一起探索代码！")
            break

        # 空输入跳过
        if not query:
            continue

        # 退出命令
        if query.lower() in ("exit", "quit", "q", "退出"):
            print("\n  👋 再见！期待下次和你一起探索代码！")
            break

        # 处理提问
        answer_query(query, components)


# ============================================================
# 四、程序入口
# ============================================================

def main():
    # 1. 一次性初始化全部索引
    components = build_index()
    if components is None:
        print("\n  💥 索引初始化失败，程序退出。")
        print("  🔧 请检查 .env 配置、网络连接、source_code 目录等。")
        sys.exit(1)

    # 2. 进入交互循环
    interactive_loop(components)


if __name__ == "__main__":
    main()
