import ctypes
import numpy as np
import os

# Load the shared library
_lib_path = "/home/yujp/vuuz/sdk-3.17.6/lib/release/lin-x86_64-gcc/libdhd.so.3.17.6"
_lib = ctypes.CDLL(_lib_path)

# Define argument types for C functions
# int dhdGetDeviceCount(void);
_lib.dhdGetDeviceCount.argtypes = []
_lib.dhdGetDeviceCount.restype = ctypes.c_int

# int dhdOpenID(char ID);
_lib.dhdOpenID.argtypes = [ctypes.c_byte]
_lib.dhdOpenID.restype = ctypes.c_int

# int dhdGetButton(int index, char ID);
_lib.dhdGetButton.argtypes = [ctypes.c_int, ctypes.c_byte]
_lib.dhdGetButton.restype = ctypes.c_int

# int dhdGetPositionAndOrientationFrame(double *px, double *py, double *pz, double matrix[3][3], char ID);
_lib.dhdGetPositionAndOrientationFrame.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double * 3 * 3),
    ctypes.c_byte
]
_lib.dhdGetPositionAndOrientationFrame.restype = ctypes.c_int

# int dhdSetForceAndTorqueAndGripperForce(double fx, double fy, double fz, double tx, double ty, double tz, double fg, char ID);
_lib.dhdSetForceAndTorqueAndGripperForce.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double,
    ctypes.c_byte
]
_lib.dhdSetForceAndTorqueAndGripperForce.restype = ctypes.c_int

# int dhdGetGripperAngleDeg(double *angle, char ID);
_lib.dhdGetGripperAngleDeg.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_byte
]
_lib.dhdGetGripperAngleDeg.restype = ctypes.c_int

# int dhdGetLinearVelocity(double *vx, double *vy, double *vz, char ID);
_lib.dhdGetLinearVelocity.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_byte
]
_lib.dhdGetLinearVelocity.restype = ctypes.c_int

# int dhdGetAngularVelocityDeg(double *vx, double *vy, double *vz, char ID);
try:
    _lib.dhdGetAngularVelocityDeg.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_byte
    ]
    _lib.dhdGetAngularVelocityDeg.restype = ctypes.c_int
except AttributeError:
    pass

# const char* dhdErrorGetLastStr ();
_lib.dhdErrorGetLastStr.argtypes = []
_lib.dhdErrorGetLastStr.restype = ctypes.c_char_p

# int dhdClose(char ID);
_lib.dhdClose.argtypes = [ctypes.c_byte]
_lib.dhdClose.restype = ctypes.c_int

# Helper to map standard DHD API to what the python code expects
def getDeviceCount():
    return _lib.dhdGetDeviceCount()

def openID(device_id):
    ret = _lib.dhdOpenID(device_id)
    if ret < 0:
        err = _lib.dhdErrorGetLastStr()
        if err:
            try:
                err_str = err.decode('utf-8')
            except:
                err_str = str(err)
            print(f"DHD Error for openID({device_id}): {err_str}")
    return ret

def getButton(index, device_id):
    return _lib.dhdGetButton(index, device_id)

def getPositionAndOrientationFrame(pos, matrix, device_id):
    # pos is np.array of shape (3,)
    # matrix is np.array of shape (3, 3)
    
    px = ctypes.c_double()
    py = ctypes.c_double()
    pz = ctypes.c_double()
    
    # Create a 3x3 double array
    MatrixType = ctypes.c_double * 3 * 3
    c_matrix = MatrixType()
    
    ret = _lib.dhdGetPositionAndOrientationFrame(
        ctypes.byref(px),
        ctypes.byref(py),
        ctypes.byref(pz),
        ctypes.byref(c_matrix),
        device_id
    )
    
    # Copy back results
    pos[0] = px.value
    pos[1] = py.value
    pos[2] = pz.value
    
    for r in range(3):
        for c in range(3):
            matrix[r, c] = c_matrix[r][c]
            
    return ret

def setForceAndTorqueAndGripperForce(fx, fy, fz, tx, ty, tz, fg, device_id):
    return _lib.dhdSetForceAndTorqueAndGripperForce(fx, fy, fz, tx, ty, tz, fg, device_id)

def getGripperAngleDeg(angle_ptr, device_id):
    return _lib.dhdGetGripperAngleDeg(angle_ptr, device_id)

def getLinearVelocity(vel, device_id):
    # vel is np.array of shape (3,)
    vx = ctypes.c_double()
    vy = ctypes.c_double()
    vz = ctypes.c_double()
    
    ret = _lib.dhdGetLinearVelocity(
        ctypes.byref(vx),
        ctypes.byref(vy),
        ctypes.byref(vz),
        device_id
    )
    
    vel[0] = vx.value
    vel[1] = vy.value
    vel[2] = vz.value
    
    return ret

def getAngularVelocityDeg(vel, device_id):
    # vel is np.array of shape (3,)
    vx = ctypes.c_double()
    vy = ctypes.c_double()
    vz = ctypes.c_double()
    
    if not hasattr(_lib, 'dhdGetAngularVelocityDeg'):
        return -1

    ret = _lib.dhdGetAngularVelocityDeg(
        ctypes.byref(vx),
        ctypes.byref(vy),
        ctypes.byref(vz),
        device_id
    )
    
    vel[0] = vx.value
    vel[1] = vy.value
    vel[2] = vz.value
    
    return ret

def close(device_id):
    return _lib.dhdClose(device_id)

class OSIndependent:
    def kbHit(self):
        # TODO: Implement proper kbhit for Linux if needed
        return False
        
    def kbGet(self):
        return ''
        
    def sleep(self, seconds):
        import time
        time.sleep(seconds)

os_independent = OSIndependent()
