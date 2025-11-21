#!/bin/bash
# Test GPU computation - one line command

python3 -c 'import torch; import warnings; warnings.filterwarnings("ignore"); print("Testing GPU computation..."); x = torch.randn(100, 100).cuda(); y = torch.randn(100, 100).cuda(); z = torch.matmul(x, y); print("OK: Small tensor computation OK!"); x = torch.randn(1000, 1000).cuda(); y = torch.randn(1000, 1000).cuda(); z = torch.matmul(x, y); print("OK: Large tensor computation OK!"); model = torch.nn.Linear(100, 100).cuda(); x = torch.randn(10, 100).cuda(); y = model(x); print("OK: Model computation OK!"); print("OK: GPU hoat dong tot! Co the chay training.")'

