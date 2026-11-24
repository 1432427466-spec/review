# 安装命令: pip install elasticsearch
from elasticsearch import Elasticsearch

# 建立连接（如果你的 ES 开启了安全认证，需传入 basic_auth）
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "your_password") # 若无密码可省略此参数
)

# 测试连接
if es.ping():
    print("ES 连接成功！")

# 核心代码检索
query = {
    "query": {
        "match": {
            "title": "Elasticsearch 教程"  # 会被分词为 "elasticsearch" 和 "教程" 进行匹配
        }
    }
}
result = es.search(index="my_index", body=query)

# 解析并打印结果
for hit in result['hits']['hits']:
    print(f"文档ID: {hit['_id']}, 得分: {hit['_score']}, 内容: {hit['_source']['title']}")

query = {
    "query": {
        "bool": {
            "must": [
                {"term": {"status": "published"}}  # 精确匹配状态
            ],
            "filter": [
                {"range": {"timestamp": {"gte": "2024-01-01"}}} # 过滤大于等于某日期的数据
            ]
        }
    }
}
result = es.search(index="my_index", body=query)


query = {
    "query": {
        "bool": {
            "must": [
                {"match": {"content": "Python"}}
            ],
            "should": [
                {"match": {"tags": "AI"}},
                {"match": {"tags": "RAG"}}
            ],
            "must_not": [
                {"term": {"category": "draft"}}
            ],
            "filter": [
                {"range": {"views": {"gt": 100}}}
            ]
        }
    },
    "from": 0,   # 分页：起始位置
    "size": 10   # 分页：返回条数
}
result = es.search(index="my_index", body=query)

query = {
    "query": {
        "multi_match": {
            "query": "大模型",
            "fields": ["title^3", "content"],  # ^3 表示 title 字段的权重是 content 的 3 倍
            "type": "best_fields"
        }
    }
}
result = es.search(index="my_index", body=query)

