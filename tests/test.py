import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import sys
sys.path.append('/data/jw/projects/vit-pytorch_jw')
from vit_pytorch import ViT

def test():
    device = torch.device('npu:1')
    torch.npu.set_device(1)
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)

    img = torch.randn(1, 3, 256, 256).to(device)

    preds = v(img)
    print(preds)
    assert preds.shape == (1, 1000), 'correct logits outputted'
    preds_class = torch.argmax(preds, dim = -1)
    print(preds_class)




if __name__ == '__main__':
    test()