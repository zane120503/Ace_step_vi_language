import torch
import warnings
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    print('Testing GPU computation...')
    try:
        # Test với tensor nhỏ
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print('✅ Small tensor computation OK!')
        
        # Test với tensor lớn hơn
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print('✅ Large tensor computation OK!')
        
        # Test với model nhỏ
        model = torch.nn.Linear(100, 100).cuda()
        x = torch.randn(10, 100).cuda()
        y = model(x)
        print('✅ Model computation OK!')
        print('✅ GPU hoạt động tốt! Có thể chạy training.')
    except RuntimeError as e:
        print(f'❌ GPU computation failed: {e}')
        if 'no kernel image' in str(e):
            print('⚠️  Vẫn còn lỗi kernel image. Có thể cần PyTorch build mới hơn.')
        else:
            print('⚠️  Lỗi khác:', e)
else:
    print('❌ CUDA not available')

