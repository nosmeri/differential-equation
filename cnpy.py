try:
    import cupy as cp

    # GPU 장치 확인 (없으면 예외 발생)
    _ = cp.cuda.runtime.getDeviceCount()
    if _ < 1:
        raise RuntimeError("No CUDA device")
    xp = cp
    to_cpu = cp.asnumpy
    on_gpu = True
except Exception:
    import numpy as np

    xp = np
    to_cpu = lambda a: a  # NumPy일 땐 그대로 사용
    on_gpu = False
    print("[INFO] GPU 미사용: NumPy(CPU)로 실행합니다.")
