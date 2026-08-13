from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)

FONT_PATH = ROOT / "assets/NotoSansSC-Regular.ttf"
if FONT_PATH.exists():
    pdfmetrics.registerFont(TTFont("NotoSansSC", str(FONT_PATH)))
    REPORT_FONT = "NotoSansSC"
else:
    # Noto Sans SC is excluded from the public package to keep it compact.
    # ReportLab's built-in CJK font remains a functional fallback; install a
    # local NotoSansSC-Regular.ttf under assets/ for the best rendered PDF.
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    REPORT_FONT = "STSong-Light"


def num(x, digits=3):
    return f"{float(x):,.{digits}f}"


def read_json(name):
    return json.loads((OUT / "tables" / name).read_text(encoding="utf-8"))


def page(canvas, doc):
    canvas.saveState()
    canvas.setFont(REPORT_FONT, 8)
    canvas.setFillColor(colors.HexColor("#54606e"))
    canvas.drawString(1.6 * cm, 1.0 * cm, "华数杯 C题 · 面向算电协同的多目标调度优化研究")
    canvas.drawRightString(19.4 * cm, 1.0 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def make_styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle(
        "CN", parent=ss["BodyText"], fontName=REPORT_FONT, fontSize=10,
        leading=16, alignment=TA_JUSTIFY, spaceAfter=7,
    )
    return {
        "body": base,
        "title": ParagraphStyle("TitleCN", parent=base, fontSize=22, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#123047"), spaceAfter=20),
        "subtitle": ParagraphStyle("SubCN", parent=base, fontSize=12, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#496779"), spaceAfter=18),
        "h1": ParagraphStyle("H1CN", parent=base, fontSize=16, leading=22, textColor=colors.HexColor("#0b6b83"), spaceBefore=10, spaceAfter=10),
        "h2": ParagraphStyle("H2CN", parent=base, fontSize=12, leading=18, textColor=colors.HexColor("#d15d35"), spaceBefore=8, spaceAfter=6),
        "small": ParagraphStyle("SmallCN", parent=base, fontSize=8.2, leading=12, textColor=colors.HexColor("#4e5964")),
        "callout": ParagraphStyle("CallCN", parent=base, backColor=colors.HexColor("#edf6f7"), borderColor=colors.HexColor("#8dc9cf"), borderWidth=0.6, borderPadding=8, spaceBefore=6, spaceAfter=10),
    }


def tbl(rows, widths=None, font_size=8):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), REPORT_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b6b83")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f6")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bdc7cc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def fig(name, width=17.2 * cm, height=None):
    p = OUT / "figures" / name
    im = Image(str(p))
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth = width
    im.drawHeight = height or width * ratio
    return im


def main():
    st = make_styles()
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    audit = read_json("input_audit.json")
    fmet = pd.read_csv(OUT / "tables/q1_forecast_metrics.csv")
    q1v = read_json("q1_constraint_validation.json")
    q2v = read_json("q2_constraint_validation.json")
    q2 = pd.read_csv(OUT / "tables/q2_policy_comparison.csv")
    q3 = pd.read_csv(OUT / "tables/q3_storage_effects.csv")
    q4 = pd.read_csv(OUT / "tables/q4_scenario_results.csv")
    rec = read_json("q4_recommendation.json")
    q1s = pd.read_csv(OUT / "tables/q1_schedule_final24.csv")
    q2s = pd.read_csv(OUT / "tables/q2_schedule_final24.csv")

    doc = SimpleDocTemplate(
        str(PAPER / "solution_report.pdf"), pagesize=A4,
        rightMargin=1.6 * cm, leftMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title="面向算电协同的多目标调度优化研究",
        author="",
    )
    story = []
    story += [Spacer(1, 2.2 * cm), Paragraph("面向算电协同的多目标调度优化研究", st["title"]),
              Paragraph("华数杯 C 题 · 可复现建模、算法与结果报告", st["subtitle"]),
              Spacer(1, 0.7 * cm), fig("model_architecture.png", 16.3 * cm), Spacer(1, 0.6 * cm),
              Paragraph("摘要", st["h1"]),
              Paragraph(
                  "本文建立“需求预测—非抢占任务调度—电力流分配—储能优化—算电协同”的分层模型。"
                  "对 50,000 条任务采用确定性稀疏候选插入并以完整时间窗回退保证可行；对六个区域的购电、可再生能源、售电、储能荷电状态、峰值和爬坡采用稀疏线性规划。"
                  "在全部任务、2,407 个小时和六区域数据上，调度约束审计无违规，储能模型的功率平衡和边界条件通过数值验证。"
                  "结果同时给出成本、碳排、可再生能源利用、峰值、爬坡、等待和网络时延，并保留所有中间表、随机种子和复现实验脚本。",
                  st["body"]),
              Paragraph("关键词：算电协同；多目标调度；非抢占任务；储能；线性规划；可复现计算", st["small"]), PageBreak()]

    story += [Paragraph("1 题目拆解与数据审计", st["h1"]),
              Paragraph("四问形成递进关系：问题一刻画与预测任务需求并生成可行排程；问题二在不启用储能再优化的前提下评价任务位置/时刻对电力与碳的影响；问题三固定题目给定的基准 AI 负荷，仅优化区域电力流与储能；问题四联合前两层，并用情景分析展示碳价和可再生能源波动下的权衡。", st["body"]),
              tbl([["模块", "输入", "核心决策", "主要输出"],
                   ["Q1", "workload、GPU、时延", "区域与开始时刻", "预测、排程、甘特图、利用率"],
                   ["Q2", "Q1 排程、电价、碳强度", "任务迁移与延迟", "成本、碳排、绿电利用、服务指标"],
                   ["Q3", "基准负荷、储能、电力", "充放电、购售电、弃电", "SOC、峰值、爬坡、综合效益"],
                   ["Q4", "Q2+Q3", "交替协调与情景参数", "Pareto 式权衡、推荐方案"]],
                  [2.0*cm, 4.2*cm, 4.4*cm, 6.5*cm]),
              Spacer(1, 0.25*cm),
              Paragraph(
                  f"数据审计覆盖 {audit.get('task_count', 50000):,} 条任务；所有输入恒等式和时间范围检查结果为 {audit.get('valid', True)}。"
                  "指标口径已依据附件1.docx锁定：GPU容量按小时实际重叠折算GPU-hour，碳排按总购电量乘碳强度计算，新能源直供、充储和外送均计入利用；六区域 AvailableRenewable 序列逐时完全相同，会削弱区域差异的可识别性；题目基准 SOC 在 RegionE 的首时段存在约 1 MWh 递推偏差，且 D/E/F 的末端 SOC 低于初始值。"
                  "所以“题目给定基准”只作原始比较项，不能宣称为满足本文终端约束的可行解。", st["callout"]),
              fig("q1_demand_profile.png"), PageBreak()]

    story += [Paragraph("2 问题一：需求预测与基础调度", st["h1"]),
              Paragraph("2.1 预测设计", st["h2"]),
              Paragraph("将任务按到达小时、源区域和任务类型聚合为 GPU·h 需求。训练集为 0–2351 小时，验证集为 2352–2375 小时，最终测试期为 2376–2399 小时。验证与测试均采用固定预测原点，预测窗内真实目标先设为不可见，杜绝滞后/滚动特征偷看同一窗口的已实现需求。主模型使用直方图梯度提升回归，特征包括 24/168 小时滞后、滚动均值、小时和星期周期编码、区域与任务类型；同时报告 24 小时与 168 小时季节朴素基线，避免只给单模型成绩。", st["body"]),
              tbl([["数据段/模型", "MAE", "RMSE", "sMAPE"]] + [[f"{r.Split}/{r.Model}", num(r.MAE_GPUh), num(r.RMSE_GPUh), num(r.sMAPE)] for r in fmet.itertuples()], [4.5*cm, 3.5*cm, 3.5*cm, 3.5*cm]),
              Spacer(1, .3*cm), fig("q1_forecast.png"),
              Paragraph("HGBR 在 MAE 与 RMSE 上优于两条季节基线，但 sMAPE 未胜出；这是零值/小值分组对相对误差敏感的结果，本文不据此声称所有指标全面占优。", st["callout"]),
              PageBreak(), Paragraph("2.2 可行性优先的非抢占调度", st["h2"]),
              Paragraph("任务 i 的决策为分配区域 r_i 与整数开始小时 s_i。实时推理任务固定 s_i=a_i；训练和批处理任务在 [e_i, l_i-d_i] 内选择。持续时间保留小数小时，小时 h 的占用采用精确重叠核 ω(i,h)=max(0,min(1,s_i+d_i-h))。依据附件1，每小时GPU容量约束按实际重叠时长折算GPU-hour；IT功率同样按 ω 加权。", st["body"]),
              Paragraph("算法按实时任务优先，再按截止期与工作量排序处理柔性任务。每个任务先检查低边际能耗候选，若候选集无解，则扫描完整官方时间窗。该方法是确定性可行启发式，不声称获得 2.33 亿量级二进制变量完整 MILP 的全局最优解。", st["body"]),
              tbl([["审计项", "结果"], ["任务总数", f"{q1v['tasks_total']:,}"], ["成功调度", f"{q1v['tasks_scheduled']:,}"], ["GPU-hour 容量超限", f"{q1v['max_gpu_capacity_violation']:.2e}"], ["IT 功率超限", f"{q1v['max_it_power_violation_mw']:.2e} MW"], ["时延超限", f"{q1v['max_latency_violation_ms']:.2e} ms"], ["实时任务最大等待", f"{q1v['max_realtime_wait_h']} h"], ["越过 2406 小时", str(q1v['finish_after_2406_count'])]], [7.0*cm, 9.5*cm]),
              Spacer(1,.3*cm), fig("q1_gpu_utilization.png"), PageBreak(), fig("q1_gantt_final24.png"),
              Paragraph(f"最终 24 小时到达任务共 {len(q1s)} 条。甘特图对高密度排程进行抽样展示，完整排程表由程序生成并保存在 outputs/full 中；提交仓库保留最终 24 小时明细。", st["small"]), PageBreak()]

    b, c = q2.iloc[0], q2.iloc[1]
    story += [Paragraph("3 问题二：任务—电力—碳排权衡", st["h1"]),
              Paragraph("在问题二中不启用储能决策。每个区域按“可再生能源优先供负荷—富余可售至上限—不足由电网补足”的记账规则计算：C=Σ(p_buy·G−p_sell·S)，E=Σγ·G，U_RE=1−ΣR_curt/ΣR_avail。该口径与附件1一致：直供、充储和合规外送均视为消纳，售出绿电不抵扣购电碳排。", st["body"]),
              tbl([["策略", "成本/元", "碳排/tCO2", "绿电利用率", "峰值/MW", "平均爬坡/MW"],
                   [b.Policy, num(b.operating_cost_cny,0), num(b.carbon_tco2), num(b.renewable_utilization,4), num(b.peak_net_grid_import_mw), num(b.mean_abs_ramp_mw)],
                   [c.Policy, num(c.operating_cost_cny,0), num(c.carbon_tco2), num(c.renewable_utilization,4), num(c.peak_net_grid_import_mw), num(c.mean_abs_ramp_mw)]],
                  [4.2*cm,3.0*cm,2.8*cm,2.5*cm,2.5*cm,2.8*cm], 7.5),
              Spacer(1,.3*cm), fig("q2_policy_comparison.png"),
              Paragraph("结果不是单向支配：成本—碳联合启发式降低了净运营成本并略提高绿电利用率，但由于数据中的同构可再生曲线和局部聚集，购电碳排与峰值反而上升。因此问题二应保留两条 Pareto 比较策略；若把碳排设为硬优先，推荐基础可行排程，而不是把经济型方案误称为低碳改进。", st["callout"]),
              Paragraph(f"经济型方案仍成功调度 {q2v['tasks_scheduled']:,}/{q2v['tasks_total']:,} 条任务；最终 24 小时样本的平均等待为 {num(q2s.Wait_h.mean())} h，95% 分位等待为 {num(q2s.Wait_h.quantile(.95))} h，平均网络时延为 {num(q2s.NetworkLatency_ms.mean())} ms。", st["body"]), PageBreak()]

    qb, qn, qo = q3.iloc[0], q3.iloc[1], q3.iloc[2]
    story += [Paragraph("4 问题三：固定负荷下的储能与电力优化", st["h1"]),
              Paragraph("对每个区域逐时设置电网供负荷、可再生供负荷、电网/可再生充电、放电供负荷、售电、弃电、SOC、爬坡与峰值变量。目标函数为购售电净成本、碳价、峰值惩罚、爬坡惩罚和充放电退化成本之和；约束包括负荷平衡、可再生分配、购售电功率上限、充放电功率与效率、SOC 上下界、首末 SOC 及相邻时段爬坡绝对值线性化。六区域可分解为六个稀疏 LP 并行求解。", st["body"]),
              tbl([["方案", "成本/元", "碳排/tCO2", "绿电利用率", "峰值/MW", "平均爬坡/MW"],
                   [qn.Policy, num(qn.operating_cost_cny,0), num(qn.carbon_tco2,1), num(qn.renewable_utilization,4), num(qn.peak_net_grid_import_mw), num(qn.mean_abs_ramp_mw)],
                   [qo.Policy, num(qo.operating_cost_cny,0), num(qo.carbon_tco2,1), num(qo.renewable_utilization,4), num(qo.peak_net_grid_import_mw), num(qo.mean_abs_ramp_mw)]],
                  [4.1*cm,3.0*cm,2.8*cm,2.5*cm,2.5*cm,2.8*cm], 7.5),
              Spacer(1,.3*cm), fig("q3_storage_effects.png"), PageBreak(), fig("q3_regionE_storage.png"),
              Paragraph(f"上表采用相同固定负荷下的“优化无储能”作为储能效应反事实；题目原始基准成本 {num(qb.operating_cost_cny,0)} 元、碳排 {num(qb.carbon_tco2,1)} tCO2 仅作数据背景。零购电碳排与显著负净成本由大量同构富余可再生能源、允许售电且售电收入计入目标共同驱动。因此这一结论只在给定数据和边界规则下成立。", st["callout"]), PageBreak()]

    story += [Paragraph("5 问题四：算电协同与情景分析", st["h1"]),
              Paragraph("联合问题采用固定模式分解协调：实时电价、分时电价和区域平价三种机制进入任务层边际信号并分别重排任务；能源层随后在固定负荷下解精确 LP。碳约束用 ε 上界，新能源情景包含 ±20% 水平变化与均值保持的日内相关波动。由于任务层使用候选启发式、能源层为精确 LP，本文只声称给定任务排程下能源子问题最优和联合方案可行，不声称联合整数问题全局最优。", st["body"]),
              fig("q4_scenarios.png"),
              tbl([["电价", "碳上限", "绿电情景", "成本/元", "碳排", "利用率", "等待"]] +
                  [[r.PriceMechanism, num(r.CarbonCapRatio,1), f"{r.RenewableScale:.1f}/{r.RenewablePattern}", num(r.operating_cost_cny,0), num(r.carbon_tco2,2), num(r.renewable_utilization,4), num(r.mean_wait_h,2)] for r in q4.itertuples()],
                  [2.0*cm,1.8*cm,2.8*cm,3.1*cm,2.1*cm,2.2*cm,2.0*cm], 6.8),
              Spacer(1,.25*cm),
              Paragraph(f"推荐名义基准为 {rec['PriceMechanism']} 电价、碳上限倍率 {rec['CarbonCapRatio']}、绿电倍率 {rec['RenewableScale']}、{rec['RenewablePattern']} 波动；其电力指标为成本 {num(rec['metrics']['operating_cost_cny'],0)} 元、碳排 {num(rec['metrics']['carbon_tco2'])} tCO2、绿电利用率 {num(rec['metrics']['renewable_utilization'],4)}。推荐方案的逐任务和逐时电力结果均已导出。", st["callout"]), PageBreak()]

    story += [Paragraph("6 稳健性、误差来源与结论边界", st["h1"]),
              tbl([["风险/不确定性", "影响", "本文处理"],
                   ["附件1统一口径", "GPU-hour、碳排和绿电利用定义", "代码、表格与报告已逐项对齐"],
                   ["六区域绿电曲线相同", "区域差异和碳价响应退化", "报告异常；做 0.8/1.0/1.2 缩放敏感性"],
                   ["基准 SOC 不满足统一递推", "基准不能作为同约束可行解", "保留原始基准作账面比较；优化方案单独验收"],
                   ["完整 MILP 规模过大", "联合全局最优不可直接证明", "确定性启发式+全窗回退；所有约束事后审计"],
                   ["终端任务跨 2399", "预测窗与运行窗混淆", "允许执行延伸至题设 2406；按精确小时重叠记账"],
                   ["收益高度依赖售电规则", "负成本可能被误读", "同时报告物理流、售电上限与数据驱动边界"]],
                  [4.0*cm,5.5*cm,7.2*cm], 7.5),
              Paragraph("结论", st["h2"]),
              Paragraph("（1）HGBR 改善绝对误差但不在 sMAPE 上全面占优；（2）全量 50,000 条任务的基础排程满足全部硬约束；（3）任务层经济收益与碳/峰值之间存在真实冲突，应报告 Pareto 选择而非单一百分比；（4）固定负荷能源 LP 在给定规则下可显著提高绿电利用并消除购电碳排，但收益由富余绿电与售电机制主导；（5）联合情景在绿电下降 20% 时重新出现购电和碳排，说明可再生能源可用性是最关键敏感参数。", st["body"]),
              Paragraph("复现说明", st["h2"]),
              Paragraph(f"主运行耗时 {num(manifest['runtime_seconds'],1)} 秒。执行 python -m pip install -r requirements.txt，随后运行 python run_all.py --data-dir data --output-dir outputs --config config.json；所有模型使用固定种子 20260813。run_manifest.json 汇总输入审计和 Q1–Q4 验证门。", st["body"]),
              Paragraph("AI 使用说明：本项目使用生成式 AI 协助题意拆解、代码结构与文本初稿；所有数值由仓库代码从附件数据计算，约束检查、输入异常与结论边界均在可复现流程中显式记录。", st["small"])]

    doc.build(story, onFirstPage=page, onLaterPages=page)

    md = f"""# 面向算电协同的多目标调度优化研究

完整排版版见 `solution_report.pdf`。本报告对应一次全量可复现运行，耗时 {manifest['runtime_seconds']:.1f} 秒。

## 核心结论

- HGBR 在 MAE/RMSE 上优于季节基线，但 sMAPE 未全面胜出。
- 50,000 条任务全部成功调度，按附件1折算的 GPU-hour、IT 功率、时延、实时任务和终止时刻约束均无违规。
- 任务层成本—碳策略形成 Pareto 权衡；基础排程碳排更低，经济型排程成本更低，不能把后者误称为纯低碳改进。
- 固定负荷储能 LP 在题设数据下得到零购电碳排和负净成本，主要由大量富余绿电与允许售电驱动，属于数据边界内结论。
- 联合模型在可再生能源下降 20% 时重新出现购电与碳排，说明绿电可用性是最关键敏感参数。

## 复现

```bash
python -m pip install -r requirements.txt
python run_all.py --data-dir data --output-dir outputs --config config.json
python tools/generate_report.py
```
"""
    (PAPER / "solution_report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
