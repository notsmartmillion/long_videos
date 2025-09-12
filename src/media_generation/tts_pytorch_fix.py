#!/usr/bin/env python3
"""
TTS PyTorch 2.6+ Compatibility Wrapper
Monkey-patches torch.load to allow loading TTS models with PyTorch 2.6+
"""

import sys
import torch
import warnings

# Store original torch.load
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for TTS compatibility"""
    # If weights_only is not specified, set it to False for TTS models
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
        
    # Suppress the PyTorch 2.6+ weights_only warning for TTS
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", 
                              message=".*weights_only.*", 
                              category=FutureWarning)
        return _original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load

# Now run TTS with the patch applied
if __name__ == "__main__":
    # Import TTS after patching
    from TTS.bin.synthesize import main
    
    # Run TTS main function
    main()

