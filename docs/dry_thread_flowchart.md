# DryThread 处理流程图

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    DryThread 类                          │
│  (继承自 QThread，用于后台执行检测任务)                    │
└─────────────────────────────────────────────────────────┘
```

## 2. 初始化流程 (__init__)

```
开始
  │
  ├─→ 初始化硬件管理器 (HM)
  ├─→ 初始化ModBus管理器 (MM)
  ├─→ 初始化UI界面 (ui)
  ├─→ 创建检测器实例:
  │     ├─→ BallDetector (锡球检测器)
  │     ├─→ SizeDetector (尺寸检测器)
  │     ├─→ TemplateDetector (模板检测器)
  │     ├─→ ShiftDetector (偏移检测器)
  │     └─→ MarkDetector (标记检测器)
  ├─→ 调用 update_params() 更新参数
  └─→ 初始化结果对象 (DryThread_Result)
结束
```

## 3. 主循环流程 (run)

```
开始主循环 (while True)
  │  
  ├─→ [1] 采集图像
  │     └─→ HM.capture_image("dry_cam")
  │         ├─→ 成功? ──否──→ 发送错误信号 → 继续循环
  │         └─→ 是 ──→ 继续
  │
  ├─→ [2] 读取ModBus数据
  │     ├─→ 读取离散输入 (地址0, 4个)
  │     │     └─→ trigger_camera, trigger_front, trigger_back, trigger_finished
  │     ├─→ 读取输入寄存器 (地址2, 3个)
  │     │     └─→ mode, trigger_count, trigger_finished
  │     ├─→ 读取Lot号 (地址16, 10个寄存器)
  │     └─→ 读取SN号 (地址26, 10个寄存器)
  │
  ├─→ [3] 初始化BGA条带信息
  │     └─→ 如果 trigger_count == 1:
  │         ├─→ 判断正反面 (trigger_front == 1 ? "front" : "back")
  │         └─→ 创建 Bga_Strip 对象
  │
  ├─→ [4] 判断触发条件
  │     └─→ (trigger_camera上升沿) OR (trigger_finished上升沿)?
  │         ├─→ 是 ──→ [5] 执行检测流程
  │         └─→ 否 ──→ [6] 处理ModBus输出
  │
  ├─→ [5] 检测流程 (仅在检测模式下)
  │     ├─→ UI是否为检测模式?
  │     │     └─→ 否 ──→ 跳过检测
  │     │     └─→ 是 ──→ 继续
  │     │
  │     ├─→ 模板匹配检测
  │     │     └─→ template_detector.detect() → 获取产品位置列表
  │     │
  │     ├─→ 遍历每个产品位置
  │     │     ├─→ 提取产品图像区域
  │     │     ├─→ 调用 _detect_product() 检测单个产品
  │     │     └─→ 将检测结果绘制到图像上
  │     │
  │     ├─→ 更新显示图像
  │     ├─→ 更新动画
  │     ├─→ 写入BGA条带数据
  │     │
  │     └─→ 如果 trigger_finished 上升沿:
  │         ├─→ 获取日志信息
  |         ├─→ 将结果写入modbus通讯
  │         └─→ 调用 write_log_to_file() 写入Excel
  │
  ├─→ [6] ModBus输出控制
  │     └─→ 如果 trigger_camera == 0 AND trigger_finished == 0:
  │         └─→ 写入线圈0 = 0 作为初始化信号
  │     └─→ 否则:
  │         ├─→ 写入线圈1 = 0
  │         └─→ 写入线圈2 = 0
  │
  └─→ 更新触发状态 (trigger_camera_last, trigger_finished_last)
      └─→ 继续循环
```

## 4. 产品检测流程 (_detect_product)

```
开始检测产品
  │
  ├─→ 初始化 product_info 结构
  │     ├─→ x, y 坐标
  │     ├─→ 检测结果字典 (size_result, ball_result, mark_result, shift_result)
  │     └─→ defect_type = ["OK"]
  │
  ├─→ [检测1] Mark检测 (如果启用)
  │     ├─→ mark_detector.detect(product_image)
  │     ├─→ 检测有效?
  │     │     ├─→ 是 ──→ 记录结果 → 继续
  │     │     └─→ 否 ──→ 标记为"Mark"缺陷 → 绘制结果 → 返回
  │     │
  │
  ├─→ [检测2] Size检测 (如果启用)
  │     ├─→ size_detector.detect(product_image)
  │     ├─→ 检测有效?
  │     │     ├─→ 是 ──→ 记录结果 → 继续
  │     │     └─→ 否 ──→ 标记为"Size"缺陷 → 绘制结果 → 返回
  │     │
  │
  ├─→ [检测3] Ball检测 (如果启用)
  │     ├─→ ball_detector.detect(product_image)
  │     ├─→ 检测有效?
  │     │     ├─→ 是 ──→ 记录结果 → 继续
  │     │     └─→ 否 ──→ 
  │     │         ├─→ 判断球数量是否匹配?
  │     │         │     ├─→ 是 ──→ 标记为"Ball"缺陷
  │     │         │     └─→ 否 ──→ 标记为"Ball Count"缺陷
  │     │         └─→ 绘制结果 → 返回
  │     │
  │
  ├─→ [检测4] Shift检测 (如果启用)
  │     ├─→ shift_detector.detect(product_image)
  │     ├─→ 检测有效?
  │     │     ├─→ 是 ──→ 记录结果 → 继续
  │     │     └─→ 否 ──→ 标记为"Shift"缺陷 → 绘制结果 → 返回
  │     │
  │
  └─→ 所有检测通过 → 返回 product_info (defect_type = ["OK"])
```

## 5. 检测结果绘制流程 (draw_detection_results)

```
开始绘制
  │
  ├─→ 转换图像为BGR格式 (如果需要)
  │
  ├─→ 绘制尺寸检测结果
  │     ├─→ 如果 size_result 存在:
  │     │     ├─→ 有效? ──→ 绿色矩形框
  │     │     └─→ 无效? ──→ 红色矩形框
  │     │
  │
  ├─→ 绘制球检测结果
  │     ├─→ 如果 ball_result 存在:
  │     │     ├─→ OK球 → 绿色矩形框
  │     │     └─→ NG球 → 红色矩形框
  │     │
  │
  └─→ 绘制Mark检测结果
      └─→ 如果 mark_result 存在:
          └─→ 绘制红色轮廓
```

## 6. 日志写入流程 (write_log_to_file)

```
开始写入日志
  │
  ├─→ 检查日志信息是否为空
  │     └─→ 是 ──→ 打印警告 → 返回
  │
  ├─→ 创建日志目录 (Log/)
  │
  ├─→ 生成文件名
  │     └─→ {lot_id}_{sn_id}_dry_{side}_{timestamp}.xlsx
  │
  ├─→ 创建Excel工作簿
  │
  ├─→ [1] 写入检测流程信息
  │     ├─→ 开始时间
  │     ├─→ 结束时间
  │     ├─→ 持续时间
  │     ├─→ 总产品数
  │     ├─→ NG总数 (黄色高亮)
  │     └─→ 已启用检测项目
  │
  ├─→ [2] 写入统计信息
  │     ├─→ 宽度/高度极值
  │     ├─→ 平均尺寸
  │     ├─→ 平均球半径
  │     ├─→ 偏移X/Y极值
  │     ├─→ 偏移X/Y CPK值
  │     └─→ 各检测项目不良统计
  │
  ├─→ [3] 写入产品详细数据表头
  │     └─→ ["序号", "宽度(mm)", "高度(mm)", "有Mark", 
  │          "NG球数", "NG球信息", "偏移量X(mm)", 
  │          "偏移量Y(mm)", "是否NG", "NG类型"]
  │
  ├─→ [4] 写入产品详细数据
  │     └─→ 遍历产品列表，NG产品用黄色高亮
  │
  ├─→ 自动调整列宽
  │
  └─→ 保存Excel文件
```

## 7. 参数更新流程 (update_params)

```
开始更新参数
  │
  ├─→ 保存参数字典
  │
  ├─→ 构建各检测器参数:
  │     ├─→ size_detect_params (尺寸检测参数)
  │     ├─→ ball_detect_params (球检测参数)
  │     ├─→ shift_detect_params (偏移检测参数)
  │     ├─→ mark_detect_params (标记检测参数)
  │     └─→ template_detect_params (模板检测参数)
  │
  │
  └─→ 更新所有检测器的参数
```

## 8. 关键数据流

```
硬件层
  │
  ├─→ 相机 → 图像采集 → image
  │
  └─→ ModBus → 触发信号、Lot、SN → 控制流程

检测层
  │
  ├─→ TemplateDetector → 产品位置列表
  │
  └─→ 各检测器 → 检测结果 → product_info

数据层
  │
  ├─→ Bga_Strip → 存储产品数据 → 统计信息
  │
  └─→ Excel文件 → 日志记录
```

## 9. 检测顺序说明

**重要**: 检测按以下顺序执行，一旦发现缺陷立即返回：

1. **Mark检测** (最高优先级)
   - 如果检测到Mark缺陷，立即标记为NG并返回

2. **Size检测**
   - 如果尺寸不合格，立即标记为NG并返回

3. **Ball检测**
   - 如果球检测不合格，判断是数量问题还是质量问题
   - 立即标记为NG并返回

4. **Shift检测** (最低优先级)
   - 如果偏移不合格，立即标记为NG并返回

**注意**: 只有当前面的检测都通过时，才会执行后续检测。

## 10. 触发条件说明

- **trigger_camera**: 相机触发信号（上升沿触发检测）
- **trigger_finished**: 完成触发信号（上升沿触发日志写入）
- **trigger_count**: 触发计数（==1时初始化BGA条带）
- **trigger_front/trigger_back**: 正反面判断信号

## 11. ModBus通信说明

**输入读取**:
- 离散输入 (地址0): 4个信号
  - trigger_camera: 相机触发信号
  - trigger_front: 正面触发信号
  - trigger_back: 背面触发信号
  - trigger_finished: 完成触发信号
- 输入寄存器 (地址2): 3个值
  - mode: 写入模式
  - trigger_count: 触发计数
  - trigger_finished: 完成触发信号（重复）
- Lot号 (地址16): 10个寄存器（字符串）
- SN号 (地址26): 10个寄存器（字符串）

**输出控制**:

### 11.1 线圈控制逻辑（主循环中）

```
触发条件判断
  │
  ├─→ 如果 (trigger_camera上升沿) OR (trigger_finished上升沿):
  │     └─→ 写入线圈0 = 1 (检测完成信号)
  │
  ├─→ 如果 trigger_camera == 0 AND trigger_finished == 0:
  │     └─→ 写入线圈0 = 0 (初始化/空闲状态)
  │
  └─→ 否则:
        ├─→ 写入线圈1 = 0 (正面控制信号复位)
        └─→ 写入线圈2 = 0 (背面控制信号复位)
```

### 11.2 寄存器写入流程 (_write_modbus_registers)

```
开始写入寄存器
  │
  ├─→ [1] 数据预处理
  │     ├─→ 将 value_array 转换为 numpy 数组
  │     ├─→ 创建掩码: 值不在 [0,1,2,3] 范围内的位置
  │     └─→ 将无效值替换为 3
  │
  ├─→ [2] 确定写入地址
  │     ├─→ 获取 strip_side (front/back)
  │     ├─→ 如果 side == "front":
  │     │     ├─→ coil_address = 1 (正面完成线圈)
  │     │     └─→ register_start = 2 (正面寄存器起始地址)
  │     └─→ 如果 side == "back":
  │           ├─→ coil_address = 2 (背面完成线圈)
  │           └─→ register_start = 2002 (背面寄存器起始地址)
  │
  ├─→ [3] 数据转换
  │     ├─→ 调用 value_transmit(value_array, mode) 转换数据
  │     └─→ 将结果限制在 0-65535 范围内（Modbus寄存器范围）
  │
  ├─→ [4] 数据验证
  │     └─→ 如果 result_list 为空:
  │           └─→ 打印警告 → 返回（不写入）
  │
  ├─→ [5] 批量写入判断
  │     └─→ MAX_REGISTERS = 123 (单次最大写入数量)
  │
  ├─→ [6] 写入寄存器数据
  │     ├─→ 如果 total_registers > MAX_REGISTERS:
  │     │     ├─→ 计算批次数量
  │     │     ├─→ 循环写入每个批次:
  │     │     │     ├─→ 计算当前批次起始索引和结束索引
  │     │     │     ├─→ 提取批次数据
  │     │     │     ├─→ 计算当前寄存器起始地址
  │     │     │     ├─→ 写入多个寄存器 (WRITE_MULTIPLE_REGISTERS)
  │     │     │     └─→ 延迟 0.01秒（避免通信冲突）
  │     │     └─→ 如果任何批次写入失败 → 返回 False
  │     │
  │     └─→ 否则:
  │           └─→ 直接写入所有寄存器 (WRITE_MULTIPLE_REGISTERS)
  │
  └─→ [7] 写入完成信号
        └─→ 写入线圈 (coil_address) = 1
            ├─→ front: 线圈1 = 1 (正面检测完成)
            └─→ back: 线圈2 = 1 (背面检测完成)
```

**输出控制详细说明**:
- **线圈0**: 
  - 置1: 当检测触发时（trigger_camera上升沿或trigger_finished上升沿）
  - 置0: 当无触发时（trigger_camera == 0 AND trigger_finished == 0）
  - 用途: 检测完成/触发确认信号
  
- **线圈1**: 
  - 置1: 正面检测完成时（在 _write_modbus_registers 中）
  - 置0: 其他情况下复位（主循环中）
  - 用途: 正面检测完成信号
  
- **线圈2**: 
  - 置1: 背面检测完成时（在 _write_modbus_registers 中）
  - 置0: 其他情况下复位（主循环中）
  - 用途: 背面检测完成信号

**寄存器写入说明**:
- **正面寄存器**: 起始地址 2，最多写入 123 个寄存器（超过则分批）
- **背面寄存器**: 起始地址 2002，最多写入 123 个寄存器（超过则分批）
- **数据格式**: 
  - 输入值限制在 [0,1,2,3] 范围内，无效值自动替换为 3
  - 转换后限制在 0-65535（Modbus寄存器范围）
- **写入模式**: 根据 mode 参数进行数据转换（value_transmit函数）

