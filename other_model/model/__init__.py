# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__all__ = []

# Use try-except to handle potentially missing modules
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
    


# Export only available models
if SASRec is not None:
    __all__.append('SASRec')
if TIGER is not None:
    __all__.append('TIGER')
if ActionPiece is not None:
    __all__.append('ActionPiece')    
if DiffGRM is not None:
    __all__.append('DiffGRM')
