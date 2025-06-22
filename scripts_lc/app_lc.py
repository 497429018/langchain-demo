import os
import sys
from operator import itemgetter
from typing import List, Dict

# Add the script's directory to sys.path to allow for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from config_lc import LC_FAISS_INDEX_PATH, RETRIEVAL_TOP_K, RERANK_TOP_N
from langchain_components import QwenEmbeddings, QwenReranker, QwenLLM

# --- Pydantic Models for Structured Output ---

class LLMResponse(BaseModel):
    thinking: str = Field(description="详细说明您是如何根据上下文得出答案的思考过程。")
    final_answer: str = Field(description="根据思考过程得出的最终简洁答案。")

class ChatResponse(BaseModel):
    answer: str
    thinking: str
    sources: str

# --- 全局变量 ---
rag_chain = None

# --- FastAPI 生命周期 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("--- 正在初始化 LangChain RAG 系统... ---")

    # 1. 初始化所有自定义组件
    embeddings = QwenEmbeddings()
    reranker = QwenReranker(top_n=RERANK_TOP_N)
    llm = QwenLLM()
    
    # 2. 加载 FAISS 索引
    faiss_path_str = str(LC_FAISS_INDEX_PATH)
    print(f"正在从 '{faiss_path_str}' 加载 FAISS 索引...")
    if not LC_FAISS_INDEX_PATH.exists():
        raise RuntimeError(f"FAISS 索引目录不存在: {faiss_path_str}。请先运行 build_knowledge_base_lc.py")
    vector_store = FAISS.load_local(faiss_path_str, embeddings, allow_dangerous_deserialization=True)

    # 3. 创建检索器
    base_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
    
    # 4. 创建用于结构化输出的解析器和提示模板 - [已优化]
    parser = JsonOutputParser(pydantic_object=LLMResponse)

    # 优化了 System Prompt，指令更明确、更强制，以提高回答的准确性
    system_prompt = """你是一个专门从文本中提取信息的专家级AI助手。你的唯一任务是严格根据下面提供的“上下文”来回答用户的“问题”。

**核心指令:**
1.  **提取答案**: 你的首要目标是在“上下文”中定位并提取出可以直接回答问题的关键信息。
2.  **分步思考**: 在 `thinking` 字段中，清晰地列出你是如何一步步找到答案的。说明你分析了哪些片段，以及为什么某些片段是相关的，而其他片段是无关的。
3.  **给出最终答案**: 在 `final_answer` 字段中，只给出最直接、最简洁的答案，不要添加任何多余的解释。
4.  **处理无答案情况**: 只有在通读所有“上下文”后，100%确定没有任何信息能回答问题时，才能将 `final_answer` 设为“根据提供的资料，我无法回答这个问题。”
5.  **严格遵守格式**: 你的输出必须是，也只能是一个符合以下格式的JSON对象。绝不要输出任何JSON之外的文本。

{format_instructions}

---
**上下文:**
{context}
---
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    # 5. 构建 RAG 链 (LCEL)
    def format_docs(docs: List[Dict]) -> str:
        return "\n\n".join(f"来源: {doc.metadata.get('source', '未知')}\n内容: {doc.page_content}" for doc in docs)

    def parse_with_fallback(output_str: str) -> dict:
        try:
            return parser.parse(output_str)
        except OutputParserException as e:
            print(f"JSON OutputParser failed. Raw output: '{output_str}'. Error: {e}")
            return {
                "thinking": "模型未能按要求生成JSON格式的回答。以下是模型的原始输出。",
                "final_answer": str(output_str)
            }

    structured_generation_chain = prompt | llm | RunnableLambda(parse_with_fallback)

    rag_chain = (
        {
            "context": itemgetter("question") | compression_retriever | RunnableLambda(format_docs),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | RunnablePassthrough.assign(
            llm_output=structured_generation_chain
          )
    )
    
    print("--- LangChain RAG 系统初始化完成。 ---")
    yield
    print("--- 服务关闭。 ---")

# --- API 定义 ---
app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    chat_history_messages = []
    for msg in request.history:
        if msg.get("role") == "user":
            chat_history_messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            chat_history_messages.append(AIMessage(content=msg.get("content")))
            
    input_data = {
        "question": request.query,
        "chat_history": chat_history_messages
    }
    
    try:
        result = await rag_chain.ainvoke(input_data)
        llm_output = result["llm_output"]
        
        return ChatResponse(
            answer=llm_output["final_answer"],
            thinking=llm_output["thinking"],
            sources=result["context"]
        )
    except Exception as e:
        print(f"处理请求时发生未预料的错误: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error - Check logs for details.")