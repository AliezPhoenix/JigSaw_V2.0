"""
NG 数量与良率监控工具

根据 ng_monitor 配置和 strip 统计，判断是否告警，返回报警码。
"""


def check_ng_alarm(stats_info: dict, ng_monitor: dict) -> int:
    """
    根据 strip 统计与 ng_monitor 配置判断告警状态。

    Args:
        stats_info: 来自 Bga_Strip.get_statistics_info()，含 defect_counts、yield_rate、total_count
        ng_monitor: 来自 params["ng_monitor"]，含 monitor_enabled、defect_limits、min_yield_rate、min_yield_sample_size

    Returns:
        0: 正常
        999: 不良数量超限
        998: 良率过低
    """
    if not ng_monitor or not ng_monitor.get("monitor_enabled", True):
        return 0

    defect_limits = ng_monitor.get("defect_limits", {})
    min_yield_rate = ng_monitor.get("min_yield_rate", 100.0)
    min_yield_sample_size = ng_monitor.get("min_yield_sample_size", 100)

    defect_counts = stats_info.get("defect_counts", {})
    total_count = stats_info.get("total_count", 0)
    yield_rate = stats_info.get("yield_rate", 100.0)

    # 1. 不良数量检查：任一类型 count >= defect_limits 对应阈值
    for defect_type, limit in defect_limits.items():
        if limit is None:
            continue
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            continue
        count = defect_counts.get(defect_type, 0)
        if count >= limit_val:
            return 999

    # 2. 良率检查：仅当 total_count > min_yield_sample_size 时
    if total_count > min_yield_sample_size:
        try:
            min_rate = float(min_yield_rate)
        except (TypeError, ValueError):
            min_rate = 100.0
        if yield_rate < min_rate:
            return 998

    return 0
