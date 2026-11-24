from collections import deque, defaultdict


def kahn_topological_sort_and_layering(n, edges):
    """
    使用Kahn算法进行拓扑排序与拓扑分层
    :param n: 节点总数 (节点编号 0 到 n-1)
    :param edges: 边的列表，例如 [(0, 1), (1, 2)] 表示 0 -> 1, 1 -> 2
    """
    # 1. 构建邻接表并计算初始入度
    graph = defaultdict(list)
    indegree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    # 2. 初始化队列，将所有入度为 0 的节点入队
    queue = deque()
    for i in range(n):
        if indegree[i] == 0:
            queue.append(i)

    topo_order = []  # 一维拓扑序列
    layers = []  # 拓扑分层结果

    # 3. BFS 主循环
    while queue:
        # 【分层核心】记录当前层的节点数量
        layer_size = len(queue)
        current_layer = []

        # 处理当前层的所有节点
        for _ in range(layer_size):
            u = queue.popleft()
            topo_order.append(u)
            current_layer.append(u)

            # 拆除依赖：更新后继节点的入度
            for v in graph[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        # 将处理完的当前层加入分层结果
        layers.append(current_layer)

    # 4. 环检测
    if len(topo_order) != n:
        raise ValueError("图中存在环，无法完成拓扑排序！")

    return topo_order, layers


# ================= 测试代码 =================
if __name__ == "__main__":
    # 假设有6个任务，依赖关系如下：
    # 0 -> 1, 0 -> 2
    # 1 -> 3, 2 -> 3
    # 3 -> 4
    # 5 (孤立节点)
    num_nodes = 6
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]

    try:
        order, layer_result = kahn_topological_sort_and_layering(num_nodes, edges)
        print("一维拓扑序列:", order)
        print("拓扑分层结果:")
        for i, layer in enumerate(layer_result):
            print(f"  第 {i} 层 (可并行执行): {layer}")
    except ValueError as e:
        print(e)