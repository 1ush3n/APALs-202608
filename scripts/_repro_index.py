import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模拟 worker_embs_i [1, W, H] 的高级索引 backward
x = torch.randn(1, 8, 32, device=dev, requires_grad=True)
idx = torch.tensor(3, device=dev)
y = x[:, idx, :].reshape(1, -1)
print("fwd shape:", tuple(y.shape))
try:
    y.sum().backward()
    print("reshape+adv-index backward OK")
except RuntimeError as exc:
    print("ERROR:", exc)

# 对比：不加 reshape 的原始高级索引 backward
x2 = torch.randn(1, 8, 32, device=dev, requires_grad=True)
idx2 = torch.tensor(3, device=dev)
y2 = x2[:, idx2, :]
try:
    y2.sum().backward()
    print("plain adv-index backward OK")
except RuntimeError as exc:
    print("ERROR2:", exc)
