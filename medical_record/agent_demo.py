import requests
import json


# 1. 读取本地医疗文本
def read_medical_record(file_path):
    print(f"--- 正在读取文件: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("文件读取成功！")
        return content
    except FileNotFoundError:
        print("错误：找不到文件，请检查 medical_record.txt 是否在同一目录下。")
        return None


# 2. 调用本地大模型 API (Ollama 默认提供本地接口)
def call_local_llm(prompt_text):
    print("--- 正在调用本地大模型分析... ---")

    # Ollama 的本地 API 地址
    url = "http://localhost:11434/api/generate"

    # 构造请求数据
    payload = {
        "model": "qwen2.5:0.5b",  # 这里填你刚才下载的模型名
        "prompt": prompt_text,
        "stream": False  # 设为 False 表示一次性返回结果，不流式输出
    }

    try:
        # 发送 POST 请求
        response = requests.post(url, json=payload)
        # 检查请求是否成功
        response.raise_for_status()
        # 解析返回的 JSON 数据
        result = response.json()
        return result.get('response', '模型未返回内容')
    except Exception as e:
        return f"调用模型失败: {str(e)}"


# 3. 主程序入口
if __name__ == "__main__":
    # 第一步：读取病历
    record_text = read_medical_record("medical_record.txt")

    if record_text:
        # 第二步：构造提示词（Prompt）
        # 这里我们先让模型做一个最简单的任务：提取患者的主诉
        prompt = f"""
        你是一个医疗合规助手。请阅读以下病历内容，并提取出患者的【主诉】。
        要求：只输出主诉内容，不要输出其他废话。

        病历内容：
        {record_text}
        """

        # 第三步：调用大模型
        llm_result = call_local_llm(prompt)

        # 第四步：打印结果
        print("\n" + "=" * 30)
        print("【大模型分析结果】")
        print("=" * 30)
        print(llm_result)
        print("=" * 30)