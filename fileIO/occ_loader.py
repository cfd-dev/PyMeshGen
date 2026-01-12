"""
OpenCASCADE (pythonocc-core) DLL 预加载模块

该模块负责在 Windows 平台上预加载 OpenCASCADE 的 DLL 库，
解决 DLL 依赖问题，确保 pythonocc-core 模块能够正常导入。

使用方法：
    from occ_loader import ensure_occ_loaded
    ensure_occ_loaded()
    from OCC.Core.StlAPI import StlAPI_Reader
"""
import os
import sys
import site
import ctypes


def ensure_occ_loaded():
    """
    确保 OpenCASCADE DLL 已正确加载
    
    该函数会：
    1. 动态查找 OpenCASCADE 安装路径
    2. 将 DLL 目录添加到系统 DLL 搜索路径
    3. 按照依赖顺序预加载所有必要的 DLL
    
    注意：
    - 该函数应该在导入任何 OCC 模块之前调用
    - 如果 DLL 加载失败，函数会静默处理，不会抛出异常
    - 该函数可以多次调用，不会重复加载 DLL
    """
    try:
        # -------------------------------------------------------------------------
        # 步骤 1: 动态查找 OpenCASCADE 安装路径
        # -------------------------------------------------------------------------
        # 获取当前 Python 环境的 site-packages 目录
        # 这是 Python 第三方包的标准安装位置
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else os.path.join(sys.prefix, 'Lib', 'site-packages')
        
        # 查找 OCC (OpenCASCADE) 目录
        # pythonocc-core 安装后，其 Python 模块位于 site-packages/OCC/ 目录下
        occ_dir = os.path.join(site_packages, 'OCC')
        occ_core_dir = os.path.join(occ_dir, 'Core')
        
        # 查找 Library/bin 目录
        # 在 conda 环境中，OpenCASCADE 的 DLL 通常安装在 $PREFIX/Library/bin/ 目录
        # 这个目录包含了完整的 DLL 依赖关系
        if 'conda' in sys.prefix.lower():
            lib_bin_dir = os.path.join(sys.prefix, 'Library', 'bin')
        else:
            # 尝试其他常见的 Python 环境路径
            lib_bin_dir = os.path.join(os.path.dirname(sys.prefix), 'Library', 'bin')
        
        # -------------------------------------------------------------------------
        # 步骤 2: 将 DLL 目录添加到系统 DLL 搜索路径
        # -------------------------------------------------------------------------
        # os.add_dll_directory() 是 Python 3.8+ 提供的功能
        # 它将指定目录添加到 DLL 搜索路径，使得 LoadLibrary 能够找到这些 DLL
        # 这比修改 PATH 环境变量更安全，不会影响其他程序
        if hasattr(os, 'add_dll_directory'):
            if os.path.exists(lib_bin_dir):
                os.add_dll_directory(lib_bin_dir)
            if os.path.exists(occ_core_dir):
                os.add_dll_directory(occ_core_dir)
        
        # -------------------------------------------------------------------------
        # 步骤 3: 准备显式加载 DLL 的工具
        # -------------------------------------------------------------------------
        # 获取 Windows kernel32.dll 的句柄
        # kernel32.dll 是 Windows 的核心系统库，提供了 LoadLibraryExW 等函数
        kernel32 = ctypes.windll.kernel32
        
        # LOAD_WITH_ALTERED_SEARCH_PATH 标志
        # 这个标志告诉系统在加载 DLL 时，使用 DLL 文件的目录作为搜索路径
        # 而不是使用调用进程的目录，这对于解决 DLL 依赖问题非常重要
        # 值 0x00000008 来自 Windows API 定义
        LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
        
        # -------------------------------------------------------------------------
        # 步骤 4: 确定 DLL 加载路径优先级
        # -------------------------------------------------------------------------
        # 优先从 Library/bin 目录加载，因为那里有完整的 DLL 依赖关系
        # 如果 Library/bin 不存在，则从 OCC/Core 目录加载
        dll_load_paths = []
        if os.path.exists(lib_bin_dir):
            dll_load_paths.append(lib_bin_dir)
        if os.path.exists(occ_core_dir):
            dll_load_paths.append(occ_core_dir)
        
        # -------------------------------------------------------------------------
        # 步骤 5: 定义需要预加载的 DLL 列表（按依赖顺序）
        # -------------------------------------------------------------------------
        # OpenCASCADE 的 DLL 之间存在复杂的依赖关系，必须按照正确的顺序加载
        # 下面是按照依赖层次组织的 DLL 列表：
        #
        # 第一层：核心基础库（不依赖其他 OCC DLL）
        #   - TKernel.dll: OpenCASCADE 核心库，提供基础数据结构和工具
        #   - TKMath.dll: 数学计算库，提供向量、矩阵、几何计算等功能
        #
        # 第二层：几何基础库（依赖第一层）
        #   - TKG2d.dll: 2D 几何库
        #   - TKG3d.dll: 3D 几何库
        #   - TKGeomBase.dll: 几何基础库，提供曲线、曲面的抽象接口
        #   - TKGeomAlgo.dll: 几何算法库，提供几何计算算法
        #
        # 第三层：拓扑库（依赖几何库）
        #   - TKBRep.dll: 边界表示库，提供拓扑形状（顶点、边、面等）的实现
        #   - TKTopAlgo.dll: 拓扑算法库，提供拓扑操作算法
        #
        # 第四层：高级几何库（依赖拓扑库）
        #   - TKPrim.dll: 基本体素库，提供基本几何体（立方体、球体等）
        #   - TKBO.dll: 布尔操作库，提供布尔运算（并、交、差）
        #   - TKBool.dll: 布尔运算库（TKBO 的别名）
        #   - TKMesh.dll: 网格化库，提供网格生成功能
        #   - TKService.dll: 服务库，提供各种辅助服务
        #
        # 第五层：可视化库（依赖几何和拓扑库）
        #   - TKV3d.dll: 3D 可视化库
        #   - TKOpenGl.dll: OpenGL 渲染库
        #   - TKHLR.dll: 隐藏线消除库
        #
        # 第六层：数据交换库（依赖几何和拓扑库）
        #   - TKDE.dll: 数据交换基础库
        #   - TKDESTEP.dll: STEP 格式支持
        #   - TKDEIGES.dll: IGES 格式支持
        #   - TKDESTL.dll: STL 格式支持
        #
        # 第七层：数据模型和持久化库
        #   - TKBin.dll: 二进制格式支持
        #   - TKBinL.dll: 二进制格式支持（轻量级）
        #   - TKBinTObj.dll: 二进制 TObj 格式支持
        #   - TKBinXCAF.dll: 二进制 XCAF 格式支持
        #   - TKCAF.dll: CAF (Component Application Framework) 基础库
        #   - TKCDF.dll: CDF (Common Data Framework) 库
        #   - TKLCAF.dll: 轻量级 CAF 库
        #   - TKStd.dll: 标准数据模型库
        #   - TKStdL.dll: 标准数据模型库（轻量级）
        #   - TKTObj.dll: TObj 数据模型库
        #   - TKXml.dll: XML 格式支持
        #   - TKXmlL.dll: XML 格式支持（轻量级）
        #   - TKXmlTObj.dll: XML TObj 格式支持
        #   - TKXmlXCAF.dll: XML XCAF 格式支持
        #   - TKXCAF.dll: XCAF (Extended Component Application Framework) 库
        #   - TKXSBase.dll: 数据交换基础库
        #   - TKRWMesh.dll: 网格读写库
        #
        # 第八层：高级功能库
        #   - TKShHealing.dll: 形状修复库
        #   - TKOffset.dll: 偏移操作库
        #   - TKFeat.dll: 特征造型库
        #   - TKFillet.dll: 倒角库
        #
        # 注意：这个列表包含了大部分常用的 OpenCASCADE 库
        # 如果使用其他功能，可能需要添加相应的 DLL
        # -------------------------------------------------------------------------
        dll_list = [
            # 第一层：核心基础库
            'TKernel.dll',
            'TKMath.dll',
            
            # 第二层：几何基础库
            'TKG2d.dll',
            'TKG3d.dll',
            'TKGeomBase.dll',
            'TKGeomAlgo.dll',
            
            # 第三层：拓扑库
            'TKBRep.dll',
            'TKTopAlgo.dll',
            
            # 第四层：高级几何库
            'TKPrim.dll',
            'TKBO.dll',
            'TKBool.dll',
            'TKMesh.dll',
            'TKService.dll',
            
            # 第五层：可视化库
            'TKV3d.dll',
            'TKOpenGl.dll',
            'TKHLR.dll',
            
            # 第六层：数据交换库
            'TKDE.dll',
            'TKDESTEP.dll',
            'TKDEIGES.dll',
            'TKDESTL.dll',
            
            # 第七层：数据模型和持久化库
            'TKBin.dll',
            'TKBinL.dll',
            'TKBinTObj.dll',
            'TKBinXCAF.dll',
            'TKCAF.dll',
            'TKCDF.dll',
            'TKLCAF.dll',
            'TKStd.dll',
            'TKStdL.dll',
            'TKTObj.dll',
            'TKXml.dll',
            'TKXmlL.dll',
            'TKXmlTObj.dll',
            'TKXmlXCAF.dll',
            'TKXCAF.dll',
            'TKXSBase.dll',
            'TKRWMesh.dll',
            
            # 第八层：高级功能库
            'TKShHealing.dll',
            'TKOffset.dll',
            'TKFeat.dll',
            'TKFillet.dll',
        ]
        
        # -------------------------------------------------------------------------
        # 步骤 6: 按顺序加载所有 DLL
        # -------------------------------------------------------------------------
        # 遍历所有需要加载的 DLL，按照预定义的顺序依次加载
        # 对于每个 DLL，尝试从多个路径加载，直到成功为止
        # 使用 LOAD_WITH_ALTERED_SEARCH_PATH 标志确保 DLL 能找到其依赖
        for dll_name in dll_list:
            dll_loaded = False
            for dll_path in dll_load_paths:
                full_dll_path = os.path.join(dll_path, dll_name)
                if os.path.exists(full_dll_path):
                    try:
                        # 显式加载 DLL
                        # LoadLibraryExW 参数：
                        #   - full_dll_path: DLL 文件的完整路径
                        #   - None: 不使用 hFile 参数
                        #   - LOAD_WITH_ALTERED_SEARCH_PATH: 使用 DLL 文件目录作为搜索路径
                        kernel32.LoadLibraryExW(full_dll_path, None, LOAD_WITH_ALTERED_SEARCH_PATH)
                        dll_loaded = True
                        break
                    except Exception:
                        # 如果加载失败，忽略并继续尝试下一个路径
                        pass
    except Exception:
        # 如果预加载过程出现任何错误，静默处理
        # 这样即使预加载失败，后续的导入尝试仍然可能成功
        pass


# 模块导入时自动调用 ensure_occ_loaded()
# 这样其他模块只需要导入 occ_loader，就会自动完成 DLL 加载
ensure_occ_loaded()
