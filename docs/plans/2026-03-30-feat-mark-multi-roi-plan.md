---
title: Mark 检测多 ROI（方案 A）
type: feat
status: completed
date: 2026-03-30
brainstorm: docs/brainstorms/2026-03-30-mark-multi-roi-brainstorm.md
---

# Mark 检测多 ROI（方案 A）

## Overview

在 [`MarkDetector`](src/detectors/mark_detector.py) 内对多个 ROI 复用现有单区域二值化/轮廓逻辑，返回**按工位语义聚合后的 `is_valid`** 与 **`per_roi` 明细**；[`execute_product_detection`](src/support/support_funs.py) 保持现有 `allow_mark` 分支不变。配置**仅使用** **`mark_rois`（1～4 个矩形列表）**，**不再使用、不再写入**旧键 `mark_roi`（代码与 ini 中均删除）；干燥/移栽**共用**同一几何配置；**参数确认**时若启用 Mark 检测但 ROI 数量不合法则**弹窗警告并阻止保存**。

**依据头脑风暴**：[docs/brainstorms/2026-03-30-mark-multi-roi-brainstorm.md](../brainstorms/2026-03-30-mark-multi-roi-brainstorm.md)（方案 A、最少 1 / 最多 4 ROI、共用 `mark_rois`）。

## Problem Statement / Motivation

- 旧版单框 `mark_roi` 无法覆盖多位置 Mark 规则；干燥需「任一见 Mark 即 NG」，移栽需「任一无 Mark 即 NG」。
- UI 仅支持框选一次，与工艺不符。

## Proposed Solution（方案 A）

1. **`MarkDetector`**  
   - 仅读取 **`mark_rois`**：`list[list[int]]`，每项 `[x, y, w, h]`；默认 `[]`。**不读取 `mark_roi`**。  
   - 对每个 ROI 执行与当前 `detect` 相同的裁剪与轮廓逻辑，得到 `is_valid_i`。  
   - **`params["allow_mark"]`**（布尔）参与聚合：  
     - `allow_mark == False`（干燥）：`is_valid = any(is_valid_i)`（任一检测到 Mark → 与单 ROI「检测到」语义一致，供现有 NG 分支使用）。  
     - `allow_mark == True`（移栽）：`is_valid = all(is_valid_i)`（全部检测到才算「有 Mark」意义上的 OK）。  
   - `result_dict`：`is_valid`（聚合）、`per_roi`（每项含 `is_valid`、`mark_contour` 全图坐标、`mark_area` 等）、可选 `mark_contour` 为**全部 ROI 轮廓列表**供绘制遍历。  
   - 抽取 `_detect_single_roi_mark(image_gray, x, y, w, h)` 减少重复。

2. **`execute_product_detection`**  
   - 无需改 `allow_mark` 分支逻辑，**前提**是 `mark_detect_result[2]["is_valid"]` 已为上述聚合值。

3. **线程**  
   - [`dry_thread.py`](src/threads/dry_thread.py) / [`transfer_thread.py`](src/threads/transfer_thread.py)：`mark_detect_params` 仅含 **`mark_rois`**、`allow_mark` 及原有阈值等；**移除 `mark_roi`**。

4. **配置持久化**  
   - **仅键名 `mark_rois`**；`work_dry_params` / `work_transfer_params`（及 `config_manager` 读写路径）**删除对 `mark_roi` 的读写**。  
   - **共用策略**：保存时将**同一份** `mark_rois` 双写到 **`work_dry_params` 与 `work_transfer_params`**。  
   - **配置**：旧键 `mark_roi` 由人工编辑 JSON（如 `config/config_2.0.json`）改为 `mark_rois`，**不在代码中做自动迁移**。

5. **UI（Dry / Transfer 参数对话框）**  
   - [`DryPramasSetDialog.py`](DryPramasSetDialog.py)、[`TransferPramasSetDialog.py`](TransferPramasSetDialog.py)：`local_params["mark_rois"]` 为列表；**追加框选**向列表 append（≤4）；**删除**支持移除指定项或清空；预览绘制**多矩形**（颜色区分或序号）。  
   - **确认/保存**：若 `mark_check_enable` 为真（或等价开关）且 `len(mark_rois) < 1`，`QMessageBox.warning`，**return 不关闭**。若 `len > 4`，警告并阻止或截断（建议阻止并提示上限 4）。  
   - **`.ui` / `*_ui.py`**：按需增加列表控件、删除选中、最多 4 个提示文案（可仅代码动态启用按钮）。

6. **绘制与主界面**  
   - [`draw_detection_results`](src/support/support_funs.py)：若 `mark_result[2]` 含 `per_roi` 或 `mark_contour` 为列表，则遍历绘制多轮廓。

7. **测试**  
   - 单元测试：`MarkDetector` 对合成图多 ROI、`allow_mark` True/False 聚合；仅使用 `mark_rois` 参数。

## Technical Considerations

| 项目 | 说明 |
|------|------|
| 聚合与 `allow_mark` | 必须在检测器内完成，因单布尔 `is_valid` 在 dry（OR）与 transfer（AND）下语义不同。 |
| 性能 | 最多 4 ROI，开销可接受。 |
| 配置重复 | 两 `work_*_section` 写入相同 `mark_rois`，避免干燥/移栽几何漂移。 |
| 旧键残留 | 配方 JSON 仅保留 `mark_rois`，编辑配置时删除 `mark_roi`。 |

## Acceptance Criteria

- [x] `mark_rois` 支持 1～4 个矩形；配置与代码路径中**无 `mark_roi`**（迁移后或新工程仅 `mark_rois`）。  
- [x] 干燥：`allow_mark=False` 时任一 ROI 检出 Mark → `defect_type` 含 Mark（与现有多检逻辑一致）。  
- [x] 移栽：`allow_mark=True` 时任一 ROI 未检出 Mark → 含 Mark（NG）。  
- [x] 参数对话框：Mark 开启且 ROI 少于 1 时确认弹窗且**不保存**；超过 4 个不允许追加或保存失败并提示。  
- [x] 干燥/移栽读取到**相同** `mark_rois`（按选定持久化策略验证）。  
- [x] 预览/结果图可显示多个 Mark ROI 轮廓（或等价可视化）。

## Dependencies & Risks

| 风险 | 缓解 |
|------|------|
| 仅一侧对话框保存导致不同步 | 保存时双写 `mark_rois` 到两个 section。 |
| 绘制 API 假设单轮廓 | 统一遍历 `per_roi` 或轮廓列表。 |

## Implementation Phases

### Phase 1：MarkDetector + 线程参数

- [x] 规范化 `mark_rois`、`allow_mark` 聚合、`per_roi`、单 ROI 抽取函数。  
- [x] `dry_thread` / `transfer_thread` / `main_window` 模板测试路径传入 `mark_rois` 与 `allow_mark`。

### Phase 2：execute_product_detection 与绘制

- [x] 验证无需改分支；扩展 `draw_detection_results` 多轮廓。

### Phase 3：对话框与 config

- [x] Dry/Transfer 多 ROI 编辑、确认校验、双写 **`mark_rois`**；移除所有 `mark_roi` 引用与 ini 键；必要时 `.ui` 增量。

### Phase 4：测试与手工验收

- [x] 单元测试 + 主流程点选验证。

## References & Research

### Internal

- [`src/detectors/mark_detector.py`](src/detectors/mark_detector.py)  
- [`src/support/support_funs.py`](src/support/support_funs.py) — `execute_product_detection`、`draw_detection_results`  
- [`src/threads/dry_thread.py`](src/threads/dry_thread.py)、[`src/threads/transfer_thread.py`](src/threads/transfer_thread.py)  
- [`DryPramasSetDialog.py`](DryPramasSetDialog.py)、[`TransferPramasSetDialog.py`](TransferPramasSetDialog.py)

### Brainstorm

- [docs/brainstorms/2026-03-30-mark-multi-roi-brainstorm.md](../brainstorms/2026-03-30-mark-multi-roi-brainstorm.md)

## Post-Deploy Monitoring & Validation

- **日志/界面**：加载的配方 JSON 中 `work_*_params` 仅含 `mark_rois`。  
- **产线**：干燥任一侧假 Mark、移栽缺一侧 Mark 是否按预期 NG。  
- 无云端发布时：`No additional operational monitoring required: 本地工位软件与 ini 配置变更。`
