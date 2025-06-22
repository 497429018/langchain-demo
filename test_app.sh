#!/bin/bash

# --- 配置 ---
# API 的地址和端口
API_URL="http://127.0.0.1:8000/chat"

# --- 辅助函数 ---
# 定义一个函数来发送请求，减少代码重复
# 参数1: 问题 (Query)
# 参数2: 聊天历史 (History JSON string, optional)
function ask_question() {
    local query="$1"
    local history="${2:-[]}" # 如果没有提供历史记录，则默认为空数组 '[]'

    echo "------------------------------------------------------"
    echo "🤔 正在提问: $query"
    echo "------------------------------------------------------"

    # 优化: 使用 jq 来安全地构建 JSON payload，防止特殊字符导致的问题
    local payload
    payload=$(jq -n --arg q "$query" --argjson h "$history" '{query: $q, history: $h}')

    # 使用 curl 发送 POST 请求
    # -s: 静默模式，不显示进度条
    # -X POST: 指定请求方法
    # --header: 指定请求头
    # --data: 发送 JSON 数据
    # 将 $payload 直接传递给 --data
    curl -s -X POST "$API_URL" \
         --header 'Content-Type: application/json' \
         --header 'accept: application/json' \
         --data "$payload" | jq .
    
    echo -e "\n" # 输出一个换行符，让格式更清晰
}

# --- 测试开始 ---
echo "🚀 开始测试 RAG 应用..."
echo "======================================================"


# --- 场景1: 单轮问答测试 ---
echo "场景1: 单轮问答测试 - 测试对四本小说的基础知识"

ask_question "孙悟空的师父是谁？"
ask_question "刘备、关羽、张飞在何处结义？"
ask_question "武松在景阳冈打死了什么？"
ask_question "林黛玉的性格怎么样？"


# --- 场景2: 多轮对话测试 ---
echo "场景2: 多轮对话测试 - 基于上下文进行追问"

# 首先，定义好多轮对话的历史记录
# 注意: Shell 脚本中处理复杂的 JSON 字符串需要小心引号
HISTORY_JSON='[
    {
        "role": "user",
        "content": "林黛玉是谁？"
    },
    {
        "role": "assistant",
        "content": "林黛玉是贾母的外甥女，系林如海之女。林如海是前科探花，现任巡盐御史，其子早逝，黛玉为他与妻子贾氏所生的女儿，年方五岁，因父母双亡而被贾家收养。"
    }
]'

ask_question "她和贾宝玉是什么关系？" "$HISTORY_JSON"


echo "✅ 所有测试已完成。"
echo "======================================================"