# 检测逻辑差异分析报告

## 一、概述

本文档对比了 `dry_thread.py` 中的 `_detect_product` 方法和 `DryPramasSetDialog.py` 中的 `_execute_detections` 方法的逻辑差异。

## 二、主要差异

### 2.1 执行流程差异

#### `dry_thread._detect_product`
- **提前返回策略**：检测到NG缺陷后立即返回，不继续后续检测
- **检测顺序**：Mark → Size → Ball → Shift → Scratch
- **优点**：性能优化，减少不必要的检测
- **缺点**：无法获取完整的检测结果

#### `DryPramasSetDialog._execute_detections`
- **完整检测策略**：执行所有启用的检测，不提前返回
- **检测顺序**：Mark → Size → Ball → Shift → Scratch
- **优点**：获取完整的检测结果，便于调试和分析
- **缺点**：即使检测到NG也会继续执行后续检测

### 2.2 检测逻辑对比

#### Mark检测

**dry_thread.py:**
```python
if not mark_detect_result[2]["is_valid"]:
    # 未检测到Mark → OK
    product_info["mark_result"] = mark_detect_result
else:
    # 检测到Mark → NG
    product_info["defect_type"].remove("OK")
    product_info["defect_type"].append("Mark")
    return product_info  # 提前返回
```

**DryPramasSetDialog.py:**
```python
if mark_detect_result[2]["is_valid"]:
    # 检测到Mark → NG
    product_info["defect_type"].remove("OK")
    product_info["defect_type"].append("Mark")
# 不提前返回，继续后续检测
```

**结论**：逻辑一致，但执行策略不同

#### Size检测

**dry_thread.py:**
```python
if size_detect_result[2]["is_valid"]:
    # 尺寸合格 → OK，继续检测
    product_info["size_result"] = size_detect_result
else:
    # 尺寸不合格 → NG
    product_info["defect_type"].remove("OK")
    product_info["defect_type"].append("Size")
    return product_info  # 提前返回
```

**DryPramasSetDialog.py:**
```python
if not size_detect_result[2]["is_valid"]:
    # 尺寸不合格 → NG
    product_info["defect_type"].remove("OK")
    product_info["defect_type"].append("Size")
# 不提前返回，继续后续检测
```

**结论**：逻辑一致，但执行策略不同

#### Ball检测

**dry_thread.py:**
```python
if ball_detect_result[2]["is_valid"]:
    # 球检测合格 → OK，继续检测
    product_info["ball_result"] = ball_detect_result
else:
    # 球检测不合格 → NG
    product_info["defect_type"].remove("OK")
    # 判断是数量问题还是质量问题
    if ball_detect_result[2]["ball_count"] != self.params["ball_count"]:
        product_info["defect_type"].append("Ball Count")
    else:
        product_info["defect_type"].append("Ball")
    return product_info  # 提前返回
```

**DryPramasSetDialog.py:**
```python
if not ball_detect_result[2]["is_valid"]:
    # 球检测不合格 → NG
    product_info["defect_type"].remove("OK")
    # 判断是数量问题还是质量问题
    if ball_detect_result[2].get("ball_count", 0) != expected_count:
        product_info["defect_type"].append("Ball Count")
    else:
        product_info["defect_type"].append("Ball")
# 不提前返回，继续后续检测
```

**结论**：逻辑一致，但执行策略不同

#### Shift检测

**dry_thread.py:**
```python
if self.params.get("shift_check_enable", False):
    shift_detect_result = self.shift_detector.detect(ball_detect_result, size_detect_result)
    if shift_detect_result[2]["is_valid"]:
        # 偏移合格 → OK，继续检测
        product_info["shift_result"] = shift_detect_result
    else:
        # 偏移不合格 → NG
        product_info["defect_type"].remove("OK")
        product_info["defect_type"].append("Shift")
        return product_info  # 提前返回
```

**DryPramasSetDialog.py:**
```python
if shift_check_enable and product_info["ball_result"] is not None and product_info["size_result"] is not None:
    ball_result_dict = product_info["ball_result"][2]
    size_result_dict = product_info["size_result"][2]
    shift_detect_result = self.shift_detector.detect(ball_result_dict, size_result_dict)
    
    if not shift_detect_result[2]["is_valid"]:
        # 偏移不合格 → NG
        product_info["defect_type"].remove("OK")
        product_info["defect_type"].append("Shift")
# 不提前返回，继续后续检测
```

**差异点**：
1. `dry_thread` 直接传递 `ball_detect_result` 和 `size_detect_result`（tuple格式）
2. `DryPramasSetDialog` 从tuple中提取 `result_dict` 再传递（dict格式）
3. `dry_thread` 没有检查 `ball_result` 和 `size_result` 是否为None

**结论**：逻辑基本一致，但数据格式处理不同

#### Scratch检测（新增）

**dry_thread.py（已更新）:**
```python
if self.params.get("scratch_check_enable", False):
    scratch_detect_result = self.scratch_detector.detect(product_image)
    if scratch_detect_result[2]["is_valid"]:
        # 划痕检测合格 → OK，继续检测
        product_info["scratch_result"] = scratch_detect_result
    else:
        # 划痕检测不合格 → NG
        product_info["defect_type"].remove("OK")
        product_info["defect_type"].append("Scratch")
        return product_info  # 提前返回
```

**DryPramasSetDialog.py:**
```python
if scratch_check_enable:
    scratch_detect_result = self.scratch_detector.detect(image)
    if scratch_detect_result[0]:  # success
        product_info["scratch_result"] = scratch_detect_result
        if not scratch_detect_result[2]["is_valid"]:
            # 划痕检测不合格 → NG
            product_info["defect_type"].remove("OK")
            product_info["defect_type"].append("Scratch")
# 不提前返回，继续后续检测
```

**差异点**：
1. `DryPramasSetDialog` 检查了 `scratch_detect_result[0]`（success标志）
2. `dry_thread` 没有检查success标志，直接使用结果

**结论**：逻辑基本一致，但错误处理不同

### 2.3 错误处理差异

#### `dry_thread._detect_product`
- **无错误检查**：直接使用检测结果，假设检测总是成功
- **风险**：如果检测失败，可能导致异常

#### `DryPramasSetDialog._execute_detections`
- **有错误检查**：检查 `result[0]`（success标志）
- **错误处理**：使用 `QMessageBox.warning` 显示错误
- **优点**：更健壮的错误处理

### 2.4 参数访问差异

#### `dry_thread._detect_product`
- 使用 `self.params.get("key", default)` 安全访问参数
- 提供默认值，避免KeyError

#### `DryPramasSetDialog._execute_detections`
- 使用 `self.local_params.get("key", default)` 安全访问参数
- 提供默认值，避免KeyError

**结论**：参数访问方式一致

## 三、建议改进

### 3.1 统一错误处理

建议在 `dry_thread._detect_product` 中添加错误检查：

```python
mark_detect_result = self.mark_detector.detect(product_image)
if not mark_detect_result[0]:  # 检查success标志
    self._update_message_signal.emit(f"Mark检测失败: {mark_detect_result[1]}")
    # 可以选择跳过或使用默认值
    continue  # 或 return product_info
```

### 3.2 统一数据格式处理

建议在 `dry_thread._detect_product` 中统一使用dict格式：

```python
# Shift检测
if self.params.get("shift_check_enable", False) and product_info["ball_result"] and product_info["size_result"]:
    ball_result_dict = product_info["ball_result"][2]
    size_result_dict = product_info["size_result"][2]
    shift_detect_result = self.shift_detector.detect(ball_result_dict, size_result_dict)
```

### 3.3 考虑是否需要完整检测结果

根据业务需求决定：
- **生产环境**：使用提前返回策略（性能优先）
- **调试/测试环境**：使用完整检测策略（信息完整）

## 四、总结

### 4.1 已完成的更新

1. ✅ 在 `dry_thread.py` 中添加了 `scratch_detector`
2. ✅ 在 `update_params` 中添加了 `scratch_detect_params` 配置
3. ✅ 在 `_detect_product` 中添加了 Scratch 检测逻辑
4. ✅ 添加了 `scratch_result` 到 `product_info`

### 4.2 主要差异总结

| 项目 | dry_thread | DryPramasSetDialog |
|------|-----------|-------------------|
| 执行策略 | 提前返回 | 完整检测 |
| 错误检查 | 无 | 有 |
| Shift数据格式 | tuple | dict |
| Scratch检测 | ✅ 已添加 | ✅ 已有 |
| 参数访问 | 安全访问 | 安全访问 |

### 4.3 逻辑一致性

- ✅ Mark检测逻辑一致
- ✅ Size检测逻辑一致
- ✅ Ball检测逻辑一致
- ✅ Shift检测逻辑基本一致（数据格式不同）
- ✅ Scratch检测逻辑基本一致（错误处理不同）

总体而言，两个方法的检测逻辑是一致的，主要差异在于执行策略和错误处理方式。
