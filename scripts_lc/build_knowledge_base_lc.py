import os
import sys
# Add the script's directory to sys.path to allow for local imports
# This must be at the very top before any other local imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from config_lc import (
    KNOWLEDGE_BASE_DIR,
    LC_FAISS_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from langchain_components import QwenEmbeddings


def build_knowledge_base():
    """
    使用 LangChain 构建和存储知识库
    - 优化: 采用分批处理方式防止 GPU 显存溢出
    - 修复: 增加本地模块导入路径，避免 ModuleNotFoundError
    """
    if not KNOWLEDGE_BASE_DIR.exists() or not any(KNOWLEDGE_BASE_DIR.iterdir()):
        print(f"知识库目录 '{KNOWLEDGE_BASE_DIR}' 为空或不存在，请先将 .txt 文件放入其中。")
        return

    # 1. 加载文档
    print(f"正在从 '{KNOWLEDGE_BASE_DIR}' 加载文档...")
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR), # DirectoryLoader 需要字符串路径
        glob="**/*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    print(f"文档加载完毕，共 {len(docs)} 个文件。")

    # 2. 文本分块
    print("正在进行文本分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"文本分块完成，共生成 {len(all_splits)} 个文本块。")

    # 3. 初始化 Embedding 模型
    print("初始化自定义 Embedding 模型...")
    embeddings = QwenEmbeddings()
    
    # 4. 分批构建 FAISS 索引以节省显存
    print("正在分批构建 FAISS 索引并存储...")
    
    # 根据你的显存大小调整此值，32 是一个比较安全的大小
    BATCH_SIZE = 32 
    
    # 使用第一个批次来初始化 FAISS 数据库
    first_batch = all_splits[:BATCH_SIZE]
    if not first_batch:
        print("没有可处理的文本块，程序退出。")
        return
        
    print(f"正在处理第 1 批 (共 {len(all_splits) // BATCH_SIZE + 1} 批)...")
    db = FAISS.from_documents(first_batch, embeddings)
    
    # 循环处理剩余的批次，并添加到现有数据库中
    for i in tqdm(range(BATCH_SIZE, len(all_splits), BATCH_SIZE), desc="Adding documents to FAISS"):
        batch = all_splits[i:i + BATCH_SIZE]
        if batch:
            db.add_documents(batch)

    # 5. 保存最终的索引文件
    # 确保目标目录存在
    LC_FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    db.save_local(str(LC_FAISS_INDEX_PATH)) # save_local 需要字符串路径
    
    print("\n--- LangChain 知识库构建完成！ ---")
    print(f"FAISS 索引已保存至: {LC_FAISS_INDEX_PATH}")

if __name__ == '__main__':
    build_knowledge_base()