"""AscendCL (pyACL) OM 模型推理封装。

面向 YOLOv8n-pose 转换出的 OM 模型（单输入 NCHW float32，单输出）。
写法参考昇腾社区 pyACL 官方样例；不同 CANN 版本 API 基本一致，
若个别函数名随版本变化，以板上 pyACL 安装目录中的 acl.py 为准
（通常在 /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/acl/）。
"""

import numpy as np

try:
    import acl

    ACL_AVAILABLE = True
except ImportError:
    acl = None
    ACL_AVAILABLE = False

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

# ACL 错误码检查
ACL_SUCCESS = 0


def _check(ret, what):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"acl call failed: {what}, ret={ret}")


class OmModel:
    """OM 模型加载与同步推理。进程内只应实例化一次（节点启动时）。"""

    def __init__(self, model_path: str, device_id: int = 0):
        if not ACL_AVAILABLE:
            raise RuntimeError("pyACL 不可用：请在 Atlas 200I DK A2 上 source CANN 环境后运行")
        self.device_id = device_id
        _check(acl.init(), "acl.init")
        _check(acl.rt.set_device(device_id), "acl.rt.set_device")
        self.context, ret = acl.rt.create_context(device_id)
        _check(ret, "acl.rt.create_context")

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        _check(ret, f"acl.mdl.load_from_file({model_path})")
        self.desc = acl.mdl.create_desc()
        _check(acl.mdl.get_desc(self.desc, self.model_id), "acl.mdl.get_desc")

        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()
        self._input_buffers = []   # (device_ptr, size)
        self._output_buffers = []  # (device_ptr, size, host_shape, host_dtype)

        num_inputs = acl.mdl.get_num_inputs(self.desc)
        for i in range(num_inputs):
            size = acl.mdl.get_input_size_by_index(self.desc, i)
            dev_ptr, ret = acl.rt.malloc(size, 0)  # 0 = ACL_MEM_MALLOC_HUGE_FIRST
            _check(ret, "acl.rt.malloc(input)")
            buf = acl.create_data_buffer(dev_ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, buf)
            _check(ret, "acl.mdl.add_dataset_buffer(input)")
            self._input_buffers.append((dev_ptr, size))

        num_outputs = acl.mdl.get_num_outputs(self.desc)
        for i in range(num_outputs):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            dev_ptr, ret = acl.rt.malloc(size, 0)
            _check(ret, "acl.rt.malloc(output)")
            buf = acl.create_data_buffer(dev_ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, buf)
            _check(ret, "acl.mdl.add_dataset_buffer(output)")
            dims, ret = acl.mdl.get_output_dims(self.desc, i)
            _check(ret, "acl.mdl.get_output_dims")
            dim_count = dims["dimCount"]
            shape = tuple(dims["dims"][:dim_count]) if dim_count > 0 else (size // 4,)
            dtype = acl.mdl.get_output_data_type(self.desc, i)
            self._output_buffers.append((dev_ptr, size, shape, dtype))

    def infer(self, *inputs: np.ndarray) -> list:
        """同步推理。输入为连续内存的 numpy 数组（通常 float32），返回输出 numpy 数组列表。"""
        if len(inputs) != len(self._input_buffers):
            raise ValueError(f"expect {len(self._input_buffers)} inputs, got {len(inputs)}")
        for arr, (dev_ptr, size) in zip(inputs, self._input_buffers):
            arr = np.ascontiguousarray(arr)
            nbytes = arr.nbytes
            if nbytes > size:
                raise ValueError(f"input bytes {nbytes} exceed device buffer {size}")
            # acl.util.numpy_to_ptr 依赖 numpy 1.x C API（numpy>=2 报 _ARRAY_API），
            # 按 CANN 提示改用 bytes_to_ptr
            payload = arr.tobytes()
            host_ptr = acl.util.bytes_to_ptr(payload)
            _check(acl.rt.memcpy(dev_ptr, size, host_ptr, nbytes, ACL_MEMCPY_HOST_TO_DEVICE),
                   "acl.rt.memcpy(H2D)")

        _check(acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset),
               "acl.mdl.execute")

        results = []
        for dev_ptr, size, shape, dtype in self._output_buffers:
            np_dtype = np.float16 if dtype == 1 else np.float32  # 1=ACL_FLOAT16, 0=ACL_FLOAT
            buf = bytes(size)
            host_ptr = acl.util.bytes_to_ptr(buf)
            _check(acl.rt.memcpy(host_ptr, size, dev_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST),
                   "acl.rt.memcpy(D2H)")
            host = np.frombuffer(buf, dtype=np_dtype).reshape(shape)
            results.append(host.astype(np.float32))
        return results

    def close(self):
        try:
            for dataset in (self.input_dataset, self.output_dataset):
                num, _ = acl.mdl.get_dataset_num_buffers(dataset)
                for i in range(num):
                    buf, _ = acl.mdl.get_dataset_buffer(dataset, i)
                    acl.destroy_data_buffer(buf)
                acl.mdl.destroy_dataset(dataset)
            for dev_ptr, _ in self._input_buffers:
                acl.rt.free(dev_ptr)
            for entry in self._output_buffers:
                acl.rt.free(entry[0])
            acl.mdl.unload(self.model_id)
            acl.mdl.destroy_desc(self.desc)
            acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
            acl.finalize()
        except Exception:  # noqa: BLE001 - 析构阶段不抛异常
            pass
