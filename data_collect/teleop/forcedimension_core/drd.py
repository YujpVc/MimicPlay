import ctypes
import os

# Load the shared library
_lib_path = "/home/yujp/vuuz/sdk-3.17.6/lib/release/lin-x86_64-gcc/libdrd.so.3.17.6"
# Note: libdrd usually depends on libdhd, so we might need to load dhd first (which we did in dhd.py, but here standalone)
# CDLL usually handles dependencies if they are in RPATH or LD_LIBRARY_PATH. 
# We might need to preload libdhd here too or rely on system.
# Let's load libdhd explicitly just in case to ensure symbols are available.
_libdhd_path = "/home/yujp/vuuz/sdk-3.17.6/lib/release/lin-x86_64-gcc/libdhd.so.3.17.6"
ctypes.CDLL(_libdhd_path, mode=ctypes.RTLD_GLOBAL)
_lib = ctypes.CDLL(_lib_path)

# int drdOpenID(char ID);
_lib.drdOpenID.argtypes = [ctypes.c_byte]
_lib.drdOpenID.restype = ctypes.c_int

# int drdClose(char ID);
_lib.drdClose.argtypes = [ctypes.c_byte]
_lib.drdClose.restype = ctypes.c_int

# bool drdIsInitialized(char ID);
_lib.drdIsInitialized.argtypes = [ctypes.c_byte]
_lib.drdIsInitialized.restype = ctypes.c_bool

# int drdAutoInit(char ID);
_lib.drdAutoInit.argtypes = [ctypes.c_byte]
_lib.drdAutoInit.restype = ctypes.c_int

# int drdStart(char ID);
_lib.drdStart.argtypes = [ctypes.c_byte]
_lib.drdStart.restype = ctypes.c_int

# int drdStop(bool frc, char ID);
_lib.drdStop.argtypes = [ctypes.c_bool, ctypes.c_byte]
_lib.drdStop.restype = ctypes.c_int

def openID(device_id):
    return _lib.drdOpenID(device_id)

def close(device_id):
    return _lib.drdClose(device_id)

def isInitialized(device_id):
    return _lib.drdIsInitialized(device_id)

def autoInit(device_id):
    return _lib.drdAutoInit(device_id)

def start(device_id):
    return _lib.drdStart(device_id)

def stop(force, device_id):
    return _lib.drdStop(force, device_id)
