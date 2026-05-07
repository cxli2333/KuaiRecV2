# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__all__ = []

# 使用try-except来处理可能缺失的模块
try:
    from model.SASRec.model import SASRec
except ImportError:
    SASRec = None
    print("Warning: SASRec model not found, skipping import")
    
try:
    from model.TIGER.model import TIGER
except ImportError:
    TIGER = None
    print("Warning: TIGER model not found, skipping import")
    
try:
    from model.ActionPiece.model import ActionPiece
except ImportError:
    ActionPiece = None
    print("Warning: ActionPiece model not found, skipping import")
    
try:
    from model.DiffGRM.model import DiffGRM
except ImportError:
    DiffGRM = None
    print("Warning: DiffGRM model not found, skipping import")
    
try:
    from model.LLaDA_Rec.model import LLaDA_Rec
except ImportError:
    LLaDA_Rec = None
    print("Warning: LLaDA_Rec model not found, skipping import")
    traceback.print_exc()
# try:
#     from model.PGQ.DiT4Rec import PGQ
# except ImportError:
#     PGQ = None
#     print("Warning: PGQ model not found, skipping import")

# try:
#     from model.PGQ.DiT4Rec import PGQ
# except Exception as e:  # 临时先别只抓 ImportError
#     PGQ = None
#     import traceback
#     print("Warning: PGQ import failed!")
#     traceback.print_exc()

# 只导出存在的模型
if SASRec is not None:
    __all__.append('SASRec')
if TIGER is not None:
    __all__.append('TIGER')
if ActionPiece is not None:
    __all__.append('ActionPiece')    
if DiffGRM is not None:
    __all__.append('DiffGRM')
if LLaDA_Rec is not None:
    __all__.append('LLaDA_Rec')
# if PGQ is not None:
#     __all__.append('PGQ')
