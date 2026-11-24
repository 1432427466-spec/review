from mineru import MinerU

# 初始化客户端（不传 token 默认进入免登录的 flash 模式）
client = MinerU()

# 解析 PDF
result = client.flash_extract("input.pdf")

# 获取 Markdown 结果
print(result.markdown)