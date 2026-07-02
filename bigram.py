"""
=============================================================================
 Bigram 语言模型 —— 最简单的字符级语言模型
=============================================================================
 核心思想：给定当前字符，直接从一张"查找表"中查出下一个字符的概率分布。
 模型只有一个 nn.Embedding 层，本质上是统计训练数据中每对相邻字符的出现频率。

 这不是线性回归！这是一个 65 分类任务（vocab_size = 65 个不同字符），
 用交叉熵损失函数 (cross_entropy) 来优化。

 作者：Andrej Karpathy (前 Tesla AI 总监, OpenAI 联合创始人)
 来源：Neural Networks: Zero to Hero 视频系列
=============================================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# 超参数设置
# ============================================================================
batch_size = 32      # 每批并行处理多少条独立序列
block_size = 8       # 预测时能看到的最大上下文长度（即往前看几个字符）
max_iters = 3000     # 总共训练多少步
eval_interval = 300  # 每隔多少步评估一次 train/val 损失
learning_rate = 1e-2 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择 GPU 或 CPU
eval_iters = 200     # 评估时采样多少个 batch 取平均（让评估更稳定）
# ============================================================================

torch.manual_seed(1337)  # 固定随机种子，保证每次运行结果可复现

# ---------------------------------------------------------------------------
# 数据准备：读取莎士比亚文本
# ---------------------------------------------------------------------------
# 数据来源（可 wget 下载）：
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 统计文本中所有不重复的字符，建立字符 ↔ 整数 的映射表
chars = sorted(list(set(text)))          # 去重并排序，得到所有唯一字符
vocab_size = len(chars)                  # 词汇表大小（约 65 个字符）
stoi = { ch:i for i,ch in enumerate(chars) }   # string → integer：字符转数字
itos = { i:ch for i,ch in enumerate(chars) }   # integer → string：数字转回字符
encode = lambda s: [stoi[c] for c in s]        # 编码：字符串 → 整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # 解码：整数列表 → 字符串

# ---------------------------------------------------------------------------
# 划分训练集和验证集（90% 训练，10% 验证）
# ---------------------------------------------------------------------------
data = torch.tensor(encode(text), dtype=torch.long)  # 整个文本变成一维整数张量
n = int(0.9 * len(data))   # 前 90% 作为训练集
train_data = data[:n]      # 训练数据
val_data = data[n:]        # 验证数据

# ---------------------------------------------------------------------------
# 数据加载器：从数据中随机取一个 batch
# ---------------------------------------------------------------------------
def get_batch(split):
    """
    生成一小批训练数据。
    
    输入 x: 形状 (batch_size, block_size)，每行是一段连续的字符索引序列
    标签 y: 形状同 x，但每个位置是 x 中对应位置的下一个字符
           即 y[i] = x[i+1]，模型的任务就是"看到当前字符，预测下一个字符"
    
    举例（block_size=4, batch_size=1）：
        x = [5, 3, 8, 2]     →  输入序列
        y = [3, 8, 2, 1]     →  目标序列（每个位置是 x 的下一个字符）
    """
    data = train_data if split == 'train' else val_data
    # 随机选择 batch_size 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 从每个起始位置截取长度为 block_size 的序列作为输入 x
    x = torch.stack([data[i:i+block_size] for i in ix])
    # 对应的标签 y 是每个位置向后偏移一位
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ---------------------------------------------------------------------------
# 损失评估函数：在训练和验证集上分别计算平均损失
# ---------------------------------------------------------------------------
@torch.no_grad()  # 禁用梯度计算，评估时不需要反向传播，节省显存
def estimate_loss():
    """在 train 和 val 上各跑 eval_iters 个 batch，取平均损失"""
    out = {}
    model.eval()  # 切换到评估模式（本模型中暂无 dropout/batchnorm，但保留好习惯）
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  # 取平均，减少随机波动
    model.train()  # 切回训练模式
    return out

# ============================================================================
# Bigram 语言模型
# ============================================================================
# 这是最简单的语言模型：仅用一个 Embedding 查找表。
# 每个 token 的 embedding 向量直接就是下一个 token 的 logits（未归一化的概率）。
# 
# 直观理解：
#   - nn.Embedding(vocab_size, vocab_size) 就是一张 (65 × 65) 的表
#   - 输入字符 'a' (索引=0)，查表得到 65 个分数，代表下一个字符是各字符的可能程度
#   - 训练过程中，这张表自动学会：'a' 后面常跟 'n', 't', 'l' 等
# ============================================================================
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # 核心：一张 vocab_size × vocab_size 的嵌入查找表
        # 输入 token 索引 → 输出所有可能的下一个 token 的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        前向传播。
        
        参数：
            idx:     (B, T) 输入 token 索引序列，B=batch_size, T=block_size
            targets: (B, T) 目标 token 索引，为 None 时只做推理不计算损失
        
        返回：
            logits: (B*T, vocab_size) 每个位置对 vocab_size 个字符的预测分数
            loss:   交叉熵损失，targets 为 None 时返回 None
        """
        # 查表：每个输入 token → vocab_size 维的 logits 向量
        # (B, T) → (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # 把 (B, T, C) 展平成 (B*T, C)，方便 cross_entropy 计算
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # 交叉熵损失：衡量预测分布与真实标签之间的差距
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        自回归生成：每次预测下一个字符，然后把它拼到序列末尾，循环往复。
        
        参数：
            idx:            (B, T) 初始上下文（生成起点）
            max_new_tokens: 要生成多少个新字符
        
        返回：
            idx: (B, T + max_new_tokens) 包含原始上下文 + 生成内容的完整序列
        """
        for _ in range(max_new_tokens):
            # 1. 拿当前序列过一遍模型，得到 logits
            logits, loss = self(idx)
            # 2. 只取最后一个时间步的预测结果 → (B, vocab_size)
            logits = logits[:, -1, :]
            # 3. softmax 把 logits 转化成概率分布
            probs = F.softmax(logits, dim=-1)
            # 4. 按概率随机采样下一个字符（不是取最大概率的，而是按分布采样，增加多样性）
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 5. 把新字符拼到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# ---------------------------------------------------------------------------
# 实例化模型并移到 GPU/CPU
# ---------------------------------------------------------------------------
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# ---------------------------------------------------------------------------
# 优化器：AdamW 是目前最常用的优化器之一
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ============================================================================
# 训练循环
# ============================================================================
for iter in range(max_iters): // 3000

    # 每隔 eval_interval 步，评估一次 train/val 损失
    if iter % eval_interval == 0: //300
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 取一个 batch 的训练数据
    xb, yb = get_batch('train') // xb

    # 标准的三步训练法：
    # ① 前向传播：算 logits 和 loss
    logits, loss = model(xb, yb)
    # ② 梯度清零：把上一轮的梯度清掉（set_to_none=True 比 zero_grad() 更高效）
    optimizer.zero_grad(set_to_none=True)
    # ③ 反向传播：计算每个参数的梯度
    loss.backward()
    # ④ 更新参数：按梯度和学习率调整模型权重
    optimizer.step()

# ============================================================================
# 生成：从头开始（context = 换行符索引 0），生成 500 个字符
# ============================================================================
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # 起始符：索引 0
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
