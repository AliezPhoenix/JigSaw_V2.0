import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import logging
import os
from src.threads.thread_imports import *
import torch
import time
from src.support.operation_log import log_operation
from src.support.support_funs import (
    fulltray_load_model,
    fulltray_predict_single_image,
    hex_to_string,
    ensure_gray_u8,
    ensure_bgr_u8,
)

MODBUS_INFO_LIST=[
            {"alias": "fulltray_modbus", "host_ip": "192.168.1.50", "port": 505}
        ]
MM = ModBus_Manager(MODBUS_INFO_LIST)
MM.connect("fulltray_modbus")

change_signal_last = 0
while(1):
    success, change_signal = MM.read("fulltray_modbus",address= 3,count = 1,function_code=cst.READ_INPUT_REGISTERS)
    change_signal = change_signal[0]
    print(change_signal)
    if change_signal == 100 and change_signal_last == 0 :
        success, config_path_data = MM.read("fulltray_modbus",address= 4,count = 20,function_code=cst.READ_INPUT_REGISTERS)
        config_path = hex_to_string(config_path_data)
        print("str",config_path)
        time.sleep(1)
        MM.write("fulltray_modbus",address = 3, value_list=[200],function_code = cst.WRITE_SINGLE_REGISTER)
        break
    change_signal_last = change_signal
    