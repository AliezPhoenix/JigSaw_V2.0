# draw_detection_results 方法分析报告

## 一、现状分析

### 1.1 方法分布情况

在代码库中发现了4个 `draw_detection_results` 方法的实现：

1. **`src/threads/dry_thread.py`** (第404-474行)
   - 方法名: `draw_detection_results`
   - 绘制类型: size, ball, mark
   - 错误处理: 使用信号 `self._update_message_signal.emit`
   - Mark颜色: COLOR_RED

2. **`src/threads/transfer_thread.py`** (第403-473行)
   - 方法名: `draw_detection_results`
   - 绘制类型: size, ball, mark
   - 错误处理: 使用信号 `self._update_message_signal.emit`
   - Mark颜色: COLOR_GREEN（与dry_thread不同）

3. **`DryPramasSetDialog.py`** (第393-524行)
   - 方法名: `_draw_detection_results`
   - 绘制类型: size, ball, mark, scratch, shift（最完整）
   - 错误处理: 使用 `print`
   - Mark颜色: COLOR_GREEN

4. **`TransferPramasSetDialog.py`** (第414-545行)
   - 方法名: `_draw_detection_results`
   - 绘制类型: size, ball, mark, scratch, shift
   - 错误处理: 使用 `print`
   - Mark颜色: COLOR_RED

### 1.2 代码重复问题

- **代码重复率**: 约70-80%的代码逻辑相同
- **主要差异**:
  1. 错误处理方式（信号 vs print）
  2. Mark绘制颜色（红色 vs 绿色）
  3. 是否支持scratch和shift绘制
  4. 数据格式处理（tuple vs dict）

## 二、通用方法设计

### 2.1 方法签名

```python
def draw_detection_results(image_result: np.ndarray, product_info: dict, 
                          error_callback=None, mark_color=None, 
                          enable_scratch=True, enable_shift=True):
```

### 2.2 设计特点

1. **统一接口**: 所有调用方使用相同的函数签名
2. **灵活的错误处理**: 支持自定义错误回调函数（信号或print）
3. **可配置的Mark颜色**: 通过参数控制Mark绘制颜色
4. **完整的检测类型支持**: 包含所有检测类型（size, ball, mark, scratch, shift）
5. **向后兼容**: 支持tuple和dict两种数据格式
6. **边界检查**: 统一的边界检查逻辑

### 2.3 参数说明

- `image_result`: 输入图像（BGR或灰度）
- `product_info`: 产品信息字典，包含所有检测结果
- `error_callback`: 错误回调函数，接收错误消息字符串
- `mark_color`: Mark绘制颜色（BGR格式），默认COLOR_RED
- `enable_scratch`: 是否启用划痕检测绘制，默认True
- `enable_shift`: 是否启用偏移检测绘制，默认True

## 三、重构建议

### 3.1 重构步骤

#### 步骤1: 更新 dry_thread.py

```python
from src.support.support_funs import draw_detection_results

# 在类方法中调用
def draw_detection_results(self, image_result: np.ndarray, product_info: dict):
    return draw_detection_results(
        image_result, 
        product_info,
        error_callback=lambda msg: self._update_message_signal.emit(msg),
        mark_color=(0, 0, 255),  # COLOR_RED
        enable_scratch=False,  # dry_thread不绘制scratch
        enable_shift=False     # dry_thread不绘制shift
    )
```

#### 步骤2: 更新 transfer_thread.py

```python
from src.support.support_funs import draw_detection_results

# 在类方法中调用
def draw_detection_results(self, image_result: np.ndarray, product_info: dict):
    return draw_detection_results(
        image_result, 
        product_info,
        error_callback=lambda msg: self._update_message_signal.emit(msg),
        mark_color=(0, 255, 0),  # COLOR_GREEN
        enable_scratch=False,  # transfer_thread不绘制scratch
        enable_shift=False     # transfer_thread不绘制shift
    )
```

#### 步骤3: 更新 DryPramasSetDialog.py

```python
from src.support.support_funs import draw_detection_results

# 在类方法中调用
def _draw_detection_results(self, image_result: np.ndarray, product_info: dict):
    return draw_detection_results(
        image_result, 
        product_info,
        error_callback=print,  # 使用print
        mark_color=(0, 255, 0),  # COLOR_GREEN
        enable_scratch=True,
        enable_shift=True
    )
```

#### 步骤4: 更新 TransferPramasSetDialog.py

```python
from src.support.support_funs import draw_detection_results

# 在类方法中调用
def _draw_detection_results(self, image_result: np.ndarray, product_info: dict):
    return draw_detection_results(
        image_result, 
        product_info,
        error_callback=print,  # 使用print
        mark_color=(0, 0, 255),  # COLOR_RED
        enable_scratch=True,
        enable_shift=True
    )
```

### 3.2 重构优势

1. **代码复用**: 消除约200行重复代码
2. **维护性**: 只需在一个地方维护绘制逻辑
3. **一致性**: 确保所有调用方使用相同的绘制逻辑
4. **扩展性**: 新增检测类型只需修改通用方法
5. **测试性**: 可以集中测试绘制逻辑

### 3.3 注意事项

1. **向后兼容**: 通用方法支持tuple和dict两种格式，确保现有代码可以正常工作
2. **错误处理**: 通过error_callback参数保持原有的错误处理方式
3. **性能**: 通用方法不会引入明显的性能开销
4. **测试**: 重构后需要测试所有调用场景

## 四、结论

**可以且应该**将 `draw_detection_results` 提取为通用方法：

1. ✅ 代码重复率高（70-80%）
2. ✅ 功能逻辑相同
3. ✅ 差异可以通过参数配置
4. ✅ 提高代码可维护性
5. ✅ 统一绘制逻辑，减少bug

通用方法已创建在 `src/support/support_funs.py` 中，可以直接使用。
