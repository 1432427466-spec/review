# -*- coding: utf-8 -*-
"""
医疗多模态 AI 平台 - LangGraph 状态机编排
链路：Supervisor -> 影像分析(YOLO+扩散) -> Milvus检索 -> RAG循证 -> 报告生成 -> 人工复核
"""
import operator
from typing import TypedDict, Annotated, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# 一、State 定义（全局状态 Schema）
# 强类型 + Reducer，是整个状态机的"短期记忆"
# ============================================================
class MedicalState(TypedDict):
    # ---- 输入 ----
    patient_id: str
    image_path: str                 # DICOM 影像路径
    chief_complaint: str            # 主诉，如"胸痛，疑似肺结节"

    # ---- 视觉感知中间态 ----
    yolo_boxes: List[dict]          # YOLO 检测框 [{cls, xyxy, conf}]
    yolo_conf: float                # 最高置信度
    diffusion_applied: bool         # 是否已做过扩散增强
    enhance_retry: int              # 增强重试次数（防死循环）

    # ---- 检索中间态 ----
    similar_cases: List[dict]       # Milvus 召回的相似病例
    rag_evidence: List[dict]        # RAG 召回的指南证据（带引用溯源）

    # ---- 输出与人工介入 ----
    draft_report: str               # AI 草稿报告
    doctor_approved: bool           # 医生是否批准
    final_report: Optional[str]

    # ---- 审计日志（reducer：并行/多次写入自动追加合并）----
    audit_log: Annotated[List[str], operator.add]


# ============================================================
# 二、工具桩函数（生产环境替换为真实 YOLO/扩散/Milvus/RAG 调用）
# ============================================================
def yolo_detect(image_path: str):
    """模拟 YOLO 病灶检测，返回检测框与置信度"""
    # 真实实现：model.predict(image_path) -> boxes, conf
    return [{"cls": "nodule", "xyxy": [120, 80, 180, 140], "conf": 0.62}], 0.62


def diffusion_enhance(image_path: str):
    """模拟扩散模型去噪/特征增强，返回增强后的影像路径"""
    return image_path + "_enhanced.dcm"


def milvus_search(image_path: str, complaint: str, top_k: int = 5):
    """模拟 Milvus HNSW 混合检索（图像向量+文本向量）"""
    return [
        {"case_id": "P001", "score": 0.91, "diag": "左上肺腺癌"},
        {"case_id": "P007", "score": 0.87, "diag": "炎性结节"},
    ]


def rag_retrieve(complaint: str, top_k: int = 3):
    """模拟 RAG 循证检索（ES+向量混合召回 -> Rerank 精排）"""
    return [
        {"doc": "肺结节诊治中国专家共识2024", "page": 12,
         "text": "直径>8mm实性结节建议3个月随访CT"},
    ]


# ============================================================
# 三、节点定义（Nodes）—— 每个节点只干一件事，职责单一
# ============================================================
def supervisor_node(state: MedicalState) -> dict:
    """主管节点：意图识别与入口校验"""
    log = f"[Supervisor] 接诊 patient={state['patient_id']} 主诉={state['chief_complaint']}"
    print(log)
    return {"audit_log": [log]}


def image_analysis_node(state: MedicalState) -> dict:
    """影像分析 Agent：调用 YOLO 检测病灶"""
    boxes, conf = yolo_detect(state["image_path"])
    log = f"[ImageAgent] YOLO 检测完成，最高置信度={conf:.2f}"
    print(log)
    return {"yolo_boxes": boxes, "yolo_conf": conf, "audit_log": [log]}


def diffusion_enhance_node(state: MedicalState) -> dict:
    """扩散模型增强节点：低置信度/低质量影像触发去噪与特征增强"""
    new_path = diffusion_enhance(state["image_path"])
    log = f"[Diffusion] 影像增强完成 -> {new_path}（第{state['enhance_retry']+1}次）"
    print(log)
    return {
        "image_path": new_path,
        "diffusion_applied": True,
        "enhance_retry": state["enhance_retry"] + 1,
        "audit_log": [log],
    }


def milvus_search_node(state: MedicalState) -> dict:
    """Milvus 检索节点：以图搜图 + 语义匹配历史相似病例"""
    cases = milvus_search(state["image_path"], state["chief_complaint"])
    log = f"[Milvus] 召回相似病例 {len(cases)} 例"
    print(log)
    return {"similar_cases": cases, "audit_log": [log]}


def rag_node(state: MedicalState) -> dict:
    """RAG 循证节点：检索临床指南并做证据校验"""
    evidence = rag_retrieve(state["chief_complaint"])
    log = f"[RAG] 召回指南证据 {len(evidence)} 条（已Rerank精排）"
    print(log)
    return {"rag_evidence": evidence, "audit_log": [log]}


def report_generation_node(state: MedicalState) -> dict:
    """报告生成 Agent：整合视觉+检索+循证证据，生成带溯源的结构化报告"""
    box = state["yolo_boxes"][0] if state["yolo_boxes"] else {}
    evidence_txt = "; ".join(
        f"{e['doc']}P{e['page']}: {e['text']}" for e in state["rag_evidence"]
    )
    case_txt = ", ".join(c["case_id"] for c in state["similar_cases"])
    report = (
        f"【影像描述】检出{box.get('cls','未知')}病灶，"
        f"坐标{box.get('xyxy')}，置信度{box.get('conf',0):.2f}。\n"
        f"【相似病例】{case_txt}\n"
        f"【诊断依据】{evidence_txt}\n"
        f"【随访建议】建议3个月复查CT。"
    )
    log = "[ReportAgent] 草稿报告生成完毕，等待医生复核"
    print(log)
    return {"draft_report": report, "audit_log": [log]}


def human_review_node(state: MedicalState) -> dict:
    """人工复核节点（HIL）：医生确认/修改后归档，并将高质量数据回流"""
    status = "批准" if state["doctor_approved"] else "驳回修改"
    log = f"[Doctor] 人工复核：{status}"
    print(log)
    final = state["draft_report"] if state["doctor_approved"] else None
    return {"final_report": final, "audit_log": [log]}


# ============================================================
# 四、条件路由（Conditional Edges）—— 用代码写死关键跳转，不让LLM自由发挥
# ============================================================
def route_after_yolo(state: MedicalState) -> Literal["diffuse", "milvus", "__end__"]:
    """YOLO 后的三路分支：
       置信度够高 -> 直接检索
       置信度低且未增强过 -> 触发扩散增强后重检（循环）
       增强2次仍不达标 -> 熔断，转人工
    """
    if state["yolo_conf"] >= 0.8:
        return "milvus"
    if state["enhance_retry"] < 2:
        return "diffuse"
    return "__end__"   # 安全熔断：宁可拒答，不可误诊


def route_after_review(state: MedicalState) -> Literal["milvus", "__end__"]:
    """医生驳回 -> 回到检索重新生成；批准 -> 结束归档"""
    return "__end__" if state["doctor_approved"] else "milvus"


# ============================================================
# 五、构图（Build Graph）
# ============================================================
def build_graph():
    g = StateGraph(MedicalState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("image_analysis", image_analysis_node)
    g.add_node("diffusion_enhance", diffusion_enhance_node)
    g.add_node("milvus_search", milvus_search_node)
    g.add_node("rag", rag_node)
    g.add_node("report_generation", report_generation_node)
    g.add_node("human_review", human_review_node)

    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "image_analysis")

    # 关键条件边1：YOLO 后决定 增强重检 / 继续检索 / 熔断
    g.add_conditional_edges("image_analysis", route_after_yolo, {
        "diffuse": "diffusion_enhance",
        "milvus": "milvus_search",
        "__end__": END,
    })
    # 增强后回到 YOLO 重检，形成自愈循环
    g.add_edge("diffusion_enhance", "image_analysis")

    g.add_edge("milvus_search", "rag")
    g.add_edge("rag", "report_generation")
    g.add_edge("report_generation", "human_review")

    # 关键条件边2：人工复核后决定 重新生成 / 归档结束
    g.add_conditional_edges("human_review", route_after_review, {
        "milvus": "milvus_search",
        "__end__": END,
    })

    # 编译：注入 Checkpointer（状态持久化）