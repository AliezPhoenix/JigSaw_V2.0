import numpy as np
import cv2 as cv
import time
from PIL import Image
from ctypes import *

from tools.MvImport.MvCameraControl_class import *
from tools.MvImport.CameraParams_header import MVCC_ENUMVALUE

class _CameraCommon:

    @staticmethod
    def to_hex_str(num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2**32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr
        return hexStr
    
    @staticmethod
    def frame_to_opencv_image(frame_out_info, src_buf):
        width = int(frame_out_info.stFrameInfo.nWidth)
        height = int(frame_out_info.stFrameInfo.nHeight)
        pixel_type = int(frame_out_info.stFrameInfo.enPixelType)
        frame_len = int(frame_out_info.stFrameInfo.nFrameLen)

        np_buffer = np.ctypeslib.as_array(src_buf, shape=(frame_len,)).copy()

        if pixel_type == PixelType_Gvsp_Mono8:
            return np_buffer.reshape(height, width)

        if pixel_type == PixelType_Gvsp_RGB8_Packed:
            rgb_img = np_buffer.reshape(height, width, 3)
            return cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)

        if pixel_type == PixelType_Gvsp_BGR8_Packed:
            return np_buffer.reshape(height, width, 3)

        if pixel_type in (
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerBG8,
        ):
            bayer_img = np_buffer.reshape(height, width)
            bayer_code_map = {
                PixelType_Gvsp_BayerRG8: cv.COLOR_BAYER_RG2BGR,
                PixelType_Gvsp_BayerGB8: cv.COLOR_BAYER_GB2BGR,
                PixelType_Gvsp_BayerGR8: cv.COLOR_BAYER_GR2BGR,
                PixelType_Gvsp_BayerBG8: cv.COLOR_BAYER_BG2BGR,
            }
            return cv.cvtColor(bayer_img, bayer_code_map[pixel_type])

        return None

    _FLOAT_PARAM_NAMES = frozenset(
        {"ExposureTime", "Gain", "Gamma", "AcquisitionFrameRate"}
    )

    def Set_parameter(self, parameter_name, parameter_type=None, value=None):
        """设置相机参数。两参数 (name, value) 或三参数 (name, type, value)。"""
        if parameter_name is None:
            return 1, "Please Enter a Parameter Type"
        if value is not None:
            return self._set_parameter_typed(parameter_name, parameter_type, value)
        value = parameter_type
        if value is None:
            return 1, "Please Enter a Parameter Value"
        param_type = "FloatValue"
        return self._set_parameter_typed(parameter_name, param_type, value)

    def _set_parameter_typed(self, parameter_name, parameter_type, value):
        if not self.b_open_device:
            return 1, "Please Open Device First"
        if parameter_type == "EnumValue":
            ret = self.obj_cam.MV_CC_SetEnumValue(parameter_name, int(value))
            if ret != 0:
                return ret, "set enum value fail! ret[0x%x]" % ret
        elif parameter_type == "FloatValue":
            ret = self.obj_cam.MV_CC_SetFloatValue(parameter_name, float(value))
            if ret != 0:
                return ret, "set float value fail! ret[0x%x]" % ret
        elif parameter_type == "IntValue":
            ret = self.obj_cam.MV_CC_SetIntValueEx(parameter_name, int(value))
            if ret != 0:
                return ret, "set int value fail! ret[0x%x]" % ret
        elif parameter_type == "BoolValue":
            ret = self.obj_cam.MV_CC_SetBoolValue(parameter_name, c_bool(value))
            if ret != 0:
                return ret, "set bool value fail! ret[0x%x]" % ret
        elif parameter_type == "StringValue":
            ret = self.obj_cam.MV_CC_SetStringValue(parameter_name, value)
            if ret != 0:
                return ret, "set string value fail! ret[0x%x]" % ret
        else:
            return 1, f"Unsupported parameter type: {parameter_type}"
        return 0, "Success"

    def Get_parameter(self, parameter_name, parameter_type=None):
        if parameter_type is None:
            return None
        if parameter_type == "FloatValue":
            st_float_param = MVCC_FLOATVALUE()
            memset(byref(st_float_param), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue(parameter_name, st_float_param)
            if ret != 0:
                return None
            return st_float_param.fCurValue
        elif parameter_type == "IntValue":
            st_int_param = MVCC_INTVALUE()
            memset(byref(st_int_param), 0, sizeof(MVCC_INTVALUE))
            ret = self.obj_cam.MV_CC_GetIntValue(parameter_name, st_int_param)
            if ret != 0:
                return None
            return st_int_param.nCurValue
        elif parameter_type == "BoolValue":
            st_bool_param = c_bool()
            ret = self.obj_cam.MV_CC_GetBoolValue(parameter_name, st_bool_param)
            if ret != 0:
                return None
            return st_bool_param.bCurValue
        elif parameter_type == "StringValue":
            st_string_param = MVCC_STRINGVALUE()
            memset(byref(st_string_param), 0, sizeof(MVCC_STRINGVALUE))
            ret = self.obj_cam.MV_CC_GetStringValue(parameter_name, st_string_param)
            if ret != 0:
                return None
            return st_string_param.sCurValue
        elif parameter_type == "EnumValue":
            st_enum_param = MVCC_ENUMVALUE()
            memset(byref(st_enum_param), 0, sizeof(MVCC_ENUMVALUE))
            ret = self.obj_cam.MV_CC_GetEnumValue(parameter_name, st_enum_param)
            if ret != 0:
                return None
            return st_enum_param.nCurValue
        else:
            return None


class CameraController(_CameraCommon):

    def __init__(self,obj_cam,netIp,deviceIp,b_open_device=False,b_start_grabbing = False,h_thread_handle=None,\
                b_thread_closed=False,st_frame_info=None,b_exit=False,b_save_bmp=False,b_save_jpg=False,buf_save_image=None,\
                n_save_image_size=0,n_win_gui_id=0,frame_rate=0,exposure_time=0,gain=0):

        self.obj_cam:MvCamera= obj_cam
        self.b_open_device = b_open_device
        self.netIp = netIp
        self.deviceIp = deviceIp
        self.b_start_grabbing = b_start_grabbing 
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_win_gui_id = n_win_gui_id
        self.n_save_image_size = n_save_image_size
        self.b_thread_closed
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain

    def To_hex_str(self,num):
        return self.to_hex_str(num)

    def Open_device(self):
        if self.b_open_device:
            return 0, "Success"
        if False == self.b_open_device:
            stDevInfo = MV_CC_DEVICE_INFO()
            stGigEDev = MV_GIGE_DEVICE_INFO()
            deviceIpList = self.deviceIp.split('.')
            stGigEDev.nCurrentIp = (int(deviceIpList[0]) << 24) | (int(deviceIpList[1]) << 16) | (int(deviceIpList[2]) << 8) | int(deviceIpList[3])
            netIpList = self.netIp.split('.')
            stGigEDev.nNetExport =  (int(netIpList[0]) << 24) | (int(netIpList[1]) << 16) | (int(netIpList[2]) << 8) | int(netIpList[3])
            stDevInfo.nTLayerType = MV_GIGE_DEVICE
            stDevInfo.SpecialInfo.stGigEInfo = stGigEDev
            self.cam = MvCamera()
            g_bExit = False
            ret = self.obj_cam.MV_CC_CreateHandle(stDevInfo)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                print('show error','create handle fail! ret = '+ self.To_hex_str(ret))
                return ret, "create handle fail! ret = " + self.To_hex_str(ret)

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                print('show error','open device fail! ret = '+ self.To_hex_str(ret))
                return ret, "open device fail! ret = " + self.To_hex_str(ret)
            print ("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDevInfo.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                    if ret != 0:
                        print ("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print ("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret =self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print ("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print ("set trigger mode fail! ret[0x%x]" % ret)
            return 0, "Success"

    def Start_grabbing(self):
        if not self.b_open_device:
            return 1, "device not open!"
        if self.b_start_grabbing:
            return 0, "Success"
        self.b_exit = False
        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != 0:
            print('show error','start grabbing fail! ret = '+ self.To_hex_str(ret))
            return ret, "start grabbing fail! ret = " + self.To_hex_str(ret)
        self.b_start_grabbing = True
        print ("start grabbing successfully!")
        return 0, "Success"

    def Stop_grabbing(self):
        if not self.b_start_grabbing or not self.b_open_device:
            return 0, "Success"
        ret = self.obj_cam.MV_CC_StopGrabbing()
        if ret != 0:
            print('show error','stop grabbing fail! ret = '+self.To_hex_str(ret))
            return ret, "stop grabbing fail! ret = " + self.To_hex_str(ret)
        print ("stop grabbing successfully!")
        self.b_start_grabbing = False
        self.b_exit  = True
        return 0, "Success"

    def Close_device(self):
        if not self.b_open_device:
            return 1, "device not open!"
        ret = self.obj_cam.MV_CC_CloseDevice()
        if ret != 0:
            print('show error','close deivce fail! ret = '+self.To_hex_str(ret))
            return ret, "close device fail! ret = " + self.To_hex_str(ret)
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit  = True
        print ("close device successfully!")
        return 0, "Success"

    def Set_trigger_mode(self,strMode):
        if True == self.b_open_device:
            if "continuous" == strMode: 
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode",0)
                if ret != 0:
                    print('show error','set triggermode fail! ret = '+self.To_hex_str(ret))
            if "triggermode" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode",1)
                if ret != 0:
                    print('show error','set triggermode fail! ret = '+self.To_hex_str(ret))
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource",7)
                if ret != 0:
                    print('show error','set triggersource fail! ret = '+self.To_hex_str(ret))

    def Trigger_once(self,nCommand):
        if True == self.b_open_device:
            if 1 == nCommand: 
                ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
                if ret != 0:
                    print('show error','set triggersoftware fail! ret = '+self.To_hex_str(ret))

    def Get_parameter(self,parameter_name, parameter_type = None):
        return super().Get_parameter(parameter_name, parameter_type)


    def Get_current_user_set(self):
        """获取当前使用的用户集
        
        Returns:
            (ret, user_set_index): ret=0表示成功，非0表示失败；user_set_index为用户集索引(0/1/2)，失败时返回0
        """
        if not self.b_open_device:
            return 1, 0
        
        try:
            st_enum_value = MVCC_ENUMVALUE()
            memset(byref(st_enum_value), 0, sizeof(MVCC_ENUMVALUE))
            
            # 尝试获取UserSetSelector（当前选择的用户集）
            ret = self.obj_cam.MV_CC_GetEnumValue("UserSetSelector", st_enum_value)
            if ret == 0:
                return 0, st_enum_value.nCurValue
            else:
                # 如果获取失败，尝试获取UserSetDefault（默认用户集）
                ret = self.obj_cam.MV_CC_GetEnumValue("UserSetDefault", st_enum_value)
                if ret == 0:
                    return 0, st_enum_value.nCurValue
                else:
                    # 如果都失败，返回默认值0
                    print(f'获取用户集失败，使用默认UserSet0，ret = {self.To_hex_str(ret)}')
                    return 0, 0
        except Exception as e:
            print(f'获取用户集异常：{str(e)}，使用默认UserSet0')
            return 0, 0
    
    def Save_to_user_set(self, user_set_index=None):
        """保存当前参数到相机用户集
        
        Args:
            user_set_index: 用户集索引 (0=UserSet0, 1=UserSet1, 2=UserSet2)，如果为None则使用当前用户集
        
        Returns:
            (ret, msg): ret=0表示成功，非0表示失败；msg为消息描述
        """
        if not self.b_open_device:
            return 1, "设备未打开"
        
        try:
            # 如果没有指定用户集，则获取当前使用的用户集
            if user_set_index is None:
                ret_get, user_set_index = self.Get_current_user_set()
                if ret_get != 0:
                    user_set_index = 0  # 默认使用UserSet0
            
            # 步骤1: 选择要保存到的用户集
            ret = self.obj_cam.MV_CC_SetEnumValue("UserSetSelector", user_set_index)
            if ret != 0:
                return ret, f"设置UserSetSelector失败，ret = {self.To_hex_str(ret)}"
            
            # 步骤2: 执行保存命令
            ret = self.obj_cam.MV_CC_SetCommandValue("UserSetSave")
            if ret != 0:
                return ret, f"执行UserSetSave失败，ret = {self.To_hex_str(ret)}"
            
            return 0, f"参数已成功保存到UserSet{user_set_index}"
        except Exception as e:
            return 1, f"保存到用户集失败：{str(e)}"
        
    def Get_image(self):
        """获取图像（快速模式，优化采图速度）
        
        Returns:
            numpy.ndarray: 图像数组，失败返回None
        """
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        retry_counter = 0
        max_retries = 2  # 快速模式：减少重试次数
        timeout_ms = 1000 # 快速模式：减少超时时间
        
        while True:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, timeout_ms)
            if ret != 0 or stOutFrame.pBufAddr is None:
                retry_counter += 1
                if retry_counter >= max_retries:
                    return None
                continue

            self.st_frame_info = stOutFrame.stFrameInfo
            opencv_image = self.frame_to_opencv_image(stOutFrame, stOutFrame.pBufAddr)
            self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            if opencv_image is None:
                retry_counter += 1
                if retry_counter >= max_retries:
                    return None
                continue
            return opencv_image

    def Save_jpg(self,buf_cache):
        if(None == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".jpg"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg;                                        # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType                               # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth      = self.st_frame_info.nWidth                                    # ch:相机对应的宽 | en:Width
        stParam.nHeight     = self.st_frame_info.nHeight                                   # ch:相机对应的高 | en:Height
        stParam.nDataLen    = self.st_frame_info.nFrameLen
        stParam.pData       = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer=  cast(byref(self.buf_save_image), POINTER(c_ubyte)) 
        stParam.nBufferSize = self.n_save_image_size                                 # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80;                                                    # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)            

        if return_code != 0:
            print('show error','save jpg fail! ret = '+self.To_hex_str(return_code))
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            print('show info','save jpg success!')
        except Exception as e:
            self.b_save_jpg = False
            raise Exception("get one frame failed:%s" % str(e))
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Save_Bmp(self,buf_cache):
        if(0 == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".bmp"    
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp;                                        # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType                               # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth      = self.st_frame_info.nWidth                                    # ch:相机对应的宽 | en:Width
        stParam.nHeight     = self.st_frame_info.nHeight                                   # ch:相机对应的高 | en:Height
        stParam.nDataLen    = self.st_frame_info.nFrameLen
        stParam.pData       = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer=  cast(byref(self.buf_save_image), POINTER(c_ubyte)) 
        stParam.nBufferSize = self.n_save_image_size                                 # ch:存储节点的大小 | en:Buffer node size
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)            
        if return_code != 0:
            print('show error','save bmp fail! ret = '+self.To_hex_str(return_code))
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            print('show info','save bmp success!')
        except Exception as e:
            self.b_save_bmp = False
            raise Exception("get one frame failed:%s" % str(e))
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image



class LineScanCamera(_CameraCommon):
    def __init__(self,obj_cam:MvCamera,netIp,deviceIp,device_index=0,b_open_device=False,b_start_grabbing = False,h_thread_handle=None,\
                b_thread_closed=False,st_frame_info=None,b_exit=False,b_save_bmp=False,b_save_jpg=False,buf_save_image=None,\
                n_save_image_size=0,n_win_gui_id=0,frame_rate=0,exposure_time=0,gain=0):

        self.obj_cam = obj_cam
        self.b_open_device = b_open_device
        self.device_index = device_index
        self.netIp = netIp
        self.deviceIp = deviceIp
        self.b_start_grabbing = b_start_grabbing 
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_win_gui_id = n_win_gui_id
        self.n_save_image_size = n_save_image_size
        self.b_thread_closed
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain

    def Open_device(self):
        if self.b_open_device:
            return 0, "Success"
        if False == self.b_open_device:
            MvCamera.MV_CC_Initialize()
            device_list = MV_CC_DEVICE_INFO_LIST()
            t_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                            | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
            ret = MvCamera.MV_CC_EnumDevices(t_layer_type, device_list)
            if ret != 0:
         
                return ret,"enum devices fail! ret[0x%x]" % ret
            if device_list.nDeviceNum == 0:
                return ret,"find no device!"
            self.mvcc_dev_info = cast(device_list.pDeviceInfo[self.device_index], POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(self.mvcc_dev_info)
            if ret != 0:
                return ret,"create handle fail! ret[0x%x]" % ret
            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                return ret,"open device fail! ret[0x%x]" % ret
            print("open device successfully!")
            ret = self.obj_cam.MV_CC_SetEnumValue("UserSetSelector", 1)
            if ret != 0:
                return ret,"set user set selector fail! ret[0x%x]" % ret

            ret = self.obj_cam.MV_CC_SetCommandValue("UserSetLoad")
            if ret != 0:
                return ret,"load user set fail! ret[0x%x]" % ret

            self.b_open_device = True
            self.b_thread_closed = False
            return 0,"Success"
    
    def Close_device(self):
        if False == self.b_open_device:
            return 1, "device not open!"
        ret = self.obj_cam.MV_CC_CloseDevice()
        if ret != 0:
            return ret, "close device fail! ret[0x%x]" % ret
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_thread_closed = True
        self.b_exit = True
        print("close device successfully!")
        return 0, "Success"

    def Start_grabbing(self):
        if False == self.b_open_device:
            return 1,"device not open!"
        if self.b_start_grabbing:
            return 0, "Success"
        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != 0:
            return ret,"start grabbing fail! ret[0x%x]" % ret
        self.b_start_grabbing = True
        print("start grabbing successfully!")
        return 0,"Success"

    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                return ret, "stop grabbing fail! ret[0x%x]" % ret
            self.b_start_grabbing = False
            print("stop grabbing successfully!")
            return 0, "Success"
        return 0, "Success"

    def Get_parameter(self, parameter_name, parameter_type=None):
        return super().Get_parameter(parameter_name, parameter_type)

    def Get_image(self):
        self.st_frame_info = MV_FRAME_OUT()
        memset(byref(self.st_frame_info), 0, sizeof(self.st_frame_info))
        attempt_count = 0
        while(True):
            ret =self.obj_cam.MV_CC_GetImageBuffer(self.st_frame_info, 1000)
            if ret != 0:
                print("waiting for image buffer")
                time.sleep(0.1)
                attempt_count += 1
                if attempt_count > 10:
                    print("failed to get image buffer")
                    return None
                continue
            opencv_image =self.frame_to_opencv_image(self.st_frame_info, self.st_frame_info.pBufAddr)
            self.obj_cam.MV_CC_FreeImageBuffer(self.st_frame_info)
            return opencv_image

    def Get_current_user_set(self):
        if not self.b_open_device:
            return 1, 0
        try:
            st_enum_value = MVCC_ENUMVALUE()
            memset(byref(st_enum_value), 0, sizeof(MVCC_ENUMVALUE))
            ret = self.obj_cam.MV_CC_GetEnumValue("UserSetSelector", st_enum_value)
            if ret == 0:
                return 0, st_enum_value.nCurValue
            ret = self.obj_cam.MV_CC_GetEnumValue("UserSetDefault", st_enum_value)
            if ret == 0:
                return 0, st_enum_value.nCurValue
            print(f"获取用户集失败，使用默认UserSet0，ret = {self.to_hex_str(ret)}")
            return 0, 0
        except Exception as e:
            print(f"获取用户集异常：{str(e)}，使用默认UserSet0")
            return 0, 0

    def Save_to_user_set(self, user_set_index=None):

        if not self.b_open_device:
            return 1, "设备未打开"
        try:
            if user_set_index is None:
                ret_get, user_set_index = self.Get_current_user_set()
                if ret_get != 0:
                    user_set_index = 0
            ret = self.obj_cam.MV_CC_SetEnumValue("UserSetSelector", user_set_index)
            if ret != 0:
                return ret, f"设置UserSetSelector失败，ret = {self.to_hex_str(ret)}"
            ret = self.obj_cam.MV_CC_SetCommandValue("UserSetSave")
            if ret != 0:
                return ret, f"执行UserSetSave失败，ret = {self.to_hex_str(ret)}"
            return 0, f"参数已成功保存到UserSet{user_set_index}"
        except Exception as e:
            return 1, f"保存到用户集失败：{str(e)}"
