# 基础环境配置
pip install -r langchain_demo/requirements_lc.txt

# 执行命令
## 将小说原本生成索引
python -m scripts_lc.build_knowledge_base_lc

## 启动服务
nohup bash -c '
uvicorn scripts_lc.app_lc:app --host 0.0.0.0 --port 8000
' > logs/app_lc.log 2>&1 &

## 启动可视化
python classic_novels_rag/scripts/frontend.py

netstat -tulnp | grep :8000
kill $(lsof -t -i:8000)

# 测试实例
## 测试《西游记》相关问题
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "孙悟空的师父是谁？",
    "history": []
  }' | jq

## 测试《三国演义》相关问题
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "刘备、关羽、张飞在何处结义？",
    "history": []
  }' | jq

## 测试《水浒传》相关问题
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "武松在景阳冈打死了什么？",
    "history": []
  }' | jq

## 测试《红楼梦》相关问题
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "林黛玉的性格怎么样？",
    "history": []
  }' | jq
