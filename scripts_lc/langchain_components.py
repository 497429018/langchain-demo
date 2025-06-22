import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import List, Any, Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document

from config_lc import (
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, LLM_MODEL_PATH, DEVICE,
    EMBEDDING_MAX_LENGTH, RERANKER_MAX_LENGTH, LLM_MAX_NEW_TOKENS
)

# --- 辅助函数 ---
def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Helper function to perform last token pooling.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# 1. 自定义 Embedding 组件
class QwenEmbeddings(Embeddings):
    """
    封装 Qwen Embedding 模型.
    """
    model: Any = None
    tokenizer: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("正在加载 Embedding 模型...")
        # 优化: 根据官方示例, 增加 padding_side='left'
        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL_PATH, trust_remote_code=True, padding_side='left'
        )
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16
        ).to(DEVICE).eval()
        print("Embedding 模型加载成功。")

    def _get_instruct(self, task_description: str, query: str) -> str:
        # 与官方示例 `get_detailed_instruct` 函数对齐
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            batch_dict = self.tokenizer(
                texts, padding=True, truncation=True, 
                max_length=EMBEDDING_MAX_LENGTH, return_tensors="pt"
            )
            batch_dict = {k: v.to(DEVICE) for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = _last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            return normalized_embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """计算文档的 embedding (用于知识库构建)"""
        # 对于文档，官方建议不加 instruction
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """计算查询的 embedding (用于检索)"""
        # 对于查询，官方建议添加 instruction
        task = "Given a web search query, retrieve relevant passages that answer the query"
        instructed_text = self._get_instruct(task, text)
        embedding = self._embed([instructed_text])
        return embedding[0]

# 2. 自定义 Reranker 组件 (根据官方示例优化)
class QwenReranker(BaseDocumentCompressor):
    """
    封装 Qwen Reranker 模型.
    - 优化: 完全对齐官方示例的 prompt 模板和 tokenization 逻辑.
    """
    model: Any = None
    tokenizer: Any = None
    top_n: int = 5
    token_false_id: int = 0
    token_true_id: int = 0
    prefix_tokens: List[int] = None
    suffix_tokens: List[int] = None

    def __init__(self, top_n: int, **kwargs):
        super().__init__(**kwargs)
        self.top_n = top_n
        print("正在加载 Reranker 模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_PATH, trust_remote_code=True, padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        
        # 优化: 对齐官方示例的 prompt 模板和 token id
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        print("Reranker 模型加载成功。")

    def _format_instruction(self, instruction, query, doc):
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        """使用 Reranker 模型对检索到的文档进行排序和筛选"""
        if not documents:
            return []
        
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        pairs = [self._format_instruction(instruction, query, doc.page_content) for doc in documents]
        
        with torch.no_grad():
            # 优化: 对齐官方的 `process_inputs` 逻辑
            # 1. 先对 pair 进行 token 化，不填充
            inputs = self.tokenizer(
                pairs, padding=False, truncation='longest_first',
                return_attention_mask=False, max_length=RERANKER_MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens)
            )
            # 2. 手动拼接 prefix 和 suffix 的 token
            for i in range(len(inputs['input_ids'])):
                inputs['input_ids'][i] = self.prefix_tokens + inputs['input_ids'][i] + self.suffix_tokens
            # 3. 使用 tokenizer.pad 进行统一填充
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=RERANKER_MAX_LENGTH)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # 优化: 对齐官方的 `compute_logits` 逻辑
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            # 使用 log_softmax 和 exp 计算概率
            stacked_scores = torch.stack([false_vector, true_vector], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked_scores, dim=1)
            scores = log_probs[:, 1].exp().cpu().tolist()
        
        # 将分数添加到文档元数据中，并排序
        for doc, score in zip(documents, scores):
            doc.metadata['rerank_score'] = score
        
        sorted_docs = sorted(documents, key=lambda x: x.metadata['rerank_score'], reverse=True)
        return sorted_docs[:self.top_n]


# 3. 自定义 LLM 组件
class QwenLLM(LLM):
    """
    封装 Qwen2-7B-Instruct LLM
    """
    model: Any = None
    tokenizer: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        print("正在加载 LLM 生成模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        ).eval()
        print("LLM 生成模型加载成功。")

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(DEVICE)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            attention_mask=model_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        input_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_len:]
        
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response