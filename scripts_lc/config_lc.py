import os
import torch
from pathlib import Path

# --- 基础路径配置 (推荐使用 pathlib 简化路径操作) ---
# 项目根目录 (langchain_demo)
# Path(__file__).resolve() -> /home/mw/project/langchain_demo/scripts_lc/config_lc.py
# Path(__file__).resolve().parent -> /home/mw/project/langchain_demo/scripts_lc
# Path(__file__).resolve().parent.parent -> /home/mw/project/langchain_demo
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- 数据路径配置 ---
# 知识库源文件路径 (根据你的 tree 输出修正了文件夹名称)
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / 'data' / 'knowledge_base_lc'

# LangChain FAISS 索引的存储路径
LC_FAISS_INDEX_PATH = PROJECT_ROOT / 'data' / 'index_storage_lc'


# --- 模型路径配置 (请根据您的服务器环境修改) ---
# 存放所有模型文件的根目录
MODEL_REPO_PATH = Path.home() / 'input' # 使用 Path.home() 代替 os.path.expanduser('~')

# 模型文件夹名称 (与 MODEL_REPO_PATH 中的文件夹对应)
EMBEDDING_MODEL_NAME = 'data7053'       # 对应 Qwen3-Embedding-4B
RERANKER_MODEL_NAME = 'data2824'         # 对应 Qwen3-Reranker-4B
LLM_MODEL_NAME = 'models2179'     # 对应 Qwen2-7B-Instruct

# 拼接完整的模型路径
EMBEDDING_MODEL_PATH = MODEL_REPO_PATH / EMBEDDING_MODEL_NAME
RERANKER_MODEL_PATH = MODEL_REPO_PATH / RERANKER_MODEL_NAME
LLM_MODEL_PATH = MODEL_REPO_PATH / LLM_MODEL_NAME


# --- RAG 参数配置 ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 20  # 检索器初步检索的文档数量
RERANK_TOP_N = 5      # Reranker 模型重排后保留的文档数量


# --- 模型超参数 ---
# 注意：这些值应与你使用的模型能力相匹配
EMBEDDING_MAX_LENGTH = 8192
RERANKER_MAX_LENGTH = 8192  # 根据官方示例, Reranker 的 max_length 也是 8192
LLM_MAX_NEW_TOKENS = 512


# --- 设备配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 检查路径是否存在 (可选，但推荐) ---
def check_model_paths():
    """启动时检查必要的模型路径是否存在"""
    print("--- 正在检查模型路径 ---")
    paths_to_check = {
        "Embedding Model": EMBEDDING_MODEL_PATH,
        "Reranker Model": RERANKER_MODEL_PATH,
        "LLM Model": LLM_MODEL_PATH,
    }
    all_paths_exist = True
    for name, path in paths_to_check.items():
        if not path.exists():
            print(f"❌ 警告: {name} 路径不存在: {path}")
            all_paths_exist = False
        else:
            print(f"✅ {name} 路径正常: {path}")

    if not all_paths_exist:
        print("\n[重要提示] 部分模型路径未找到。")
        print(f"请确保所有模型已正确下载并放置在 '{MODEL_REPO_PATH}' 目录下。")
    print("--- 路径检查完毕 ---\n")

# 如果此文件作为主程序运行 (例如: python -m scripts_lc.config_lc)，则执行检查
if __name__ == '__main__':
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"知识库目录: {KNOWLEDGE_BASE_DIR}")
    print(f"FAISS 索引目录: {LC_FAISS_INDEX_PATH}")
    check_model_paths()