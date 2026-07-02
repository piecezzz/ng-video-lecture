"""
=============================================================================
 GPT 语言模型 —— 从零实现的 Transformer 解码器
=============================================================================
 这是 Andrej Karpathy "Neural Networks: Zero to Hero" 系列的核心代码。
 用约 230 行纯 PyTorch，从零构建了一个完整的 GPT-2 风格语言模型。

 架构层级（自底向上）：
   Token Embedding + Position Embedding
        ↓
   Transformer Block × n_layer（默认 6 层）
     ├── LayerNorm → Multi-Head Self-Attention (因果掩码) → 残差连接
     └── LayerNorm → FeedForward (4× 扩展) → 残差连接
        ↓
   LayerNorm → Linear(vocab_size) → 输出 logits

 这不是线性回归！这是基于 Transformer 的多分类语言建模任务。
 2026 年的 GPT-4、Claude、Gemini 等大模型，底层架构和这里一模一样，
 只是把参数量从 10M 扩展到了数百 B~数万亿，并加入了 MoE、RLHF 等技巧。

 作者：Andrej Karpathy (前 Tesla AI 总监, OpenAI 联合创始人)
 来源：Neural Networks: Zero to Hero — nanoGPT 讲座
=============================================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# 超参数
# ============================================================================
batch_size = 64      # 每批处理多少条独立序列（比 bigram 的 32 更大）
block_size = 256      # 上下文窗口大小：模型能看到的最长字符数（bigram 只有 8）
max_iters = 5000      # 训练总步数
eval_interval = 500   # 每隔多少步评估一次损失
learning_rate = 3e-4  # 学习率（比 bigram 的 1e-2 小很多，因为模型更大更敏感）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200      # 评估时采样多少个 batch 取平均
# ------------
# 模型结构参数
n_embd = 384          # 嵌入维度：每个 token 用 384 维向量表示
n_head = 6            # 注意力头数：并行跑 6 个不同的"关注模式"
n_layer = 6           # Transformer 层数：堆叠 6 个 Block
dropout = 0.2         # Dropout 比例：训练时随机丢弃 20% 的神经元，防止过拟合
# ============================================================================

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# 数据准备（与 bigram.py 完全相同）
# ---------------------------------------------------------------------------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 建立字符 ↔ 整数映射表
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 划分训练集和验证集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---------------------------------------------------------------------------
# 数据加载器
# ---------------------------------------------------------------------------
def get_batch(split):
    """
    随机采样一个 batch。
    x: (batch_size, block_size) 输入序列
    y: (batch_size, block_size) 目标序列（每个位置都是 x 对应位置的下一个字符）
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ---------------------------------------------------------------------------
# 损失评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """在 train 和 val 上各评估 eval_iters 个 batch，返回平均损失"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ============================================================================
# 第 1 层：单头自注意力 (Single Head of Self-Attention)
# ============================================================================
# 自注意力是 Transformer 的核心。它的直观含义是：
#   "序列中的每个位置，都应该看看它之前的所有位置，从中聚合信息。"
#
# 具体计算步骤（以"看"字为例）：
#   1. Query（查询）："我（'看'这个位置）想要关注什么信息？"
#   2. Key（键）：  "其他每个位置能提供什么信息？"
#   3. Q·K^T 得到注意力分数矩阵：每个位置对每个位置的"关注程度"
#   4. 用下三角掩码 (tril) 把未来位置的分数设为 -∞（因果掩码：只能看过往）
#   5. Softmax 归一化：分数变成 0~1 的概率权重
#   6. 用这些权重对 Value（值）做加权求和 → 最终输出
# ============================================================================
class Head(nn.Module):
    """单头自注意力"""

    def __init__(self, head_size):
        super().__init__()
        # 三个线性变换：把输入 x 投影成 Q, K, V 三个矩阵
        # head_size = n_embd // n_head = 384 // 6 = 64
        self.key = nn.Linear(n_embd, head_size, bias=False)    # 键投影
        self.query = nn.Linear(n_embd, head_size, bias=False)  # 查询投影
        self.value = nn.Linear(n_embd, head_size, bias=False)  # 值投影
        # 下三角掩码矩阵：保证第 i 个位置只能看到 0~i 位置，不能偷看未来
        # register_buffer 表示这不是可训练参数，但会随模型一起移到 GPU
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: x 形状 (B, T, C)  B=batch, T=时间步(序列长度), C=通道数(n_embd)
        输出: out 形状 (B, T, head_size)
        """
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)

        # ---- 计算注意力权重矩阵 ----
        # q @ k^T: 每个查询与所有键做点积 → (B, T, T) 的"亲和度"矩阵
        # 除以 √head_size 是为了防止点积值过大导致 softmax 梯度消失
        # 这被称为"缩放点积注意力"(Scaled Dot-Product Attention)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)

        # ---- 因果掩码：把未来位置的分数设为 -∞ ----
        # tril[:T, :T] 是一个下三角矩阵，上三角为 0
        # masked_fill 把上三角位置的注意力分数变成 -∞
        # 经过 softmax 后，-∞ 变成 0，实现了"不能偷看未来"
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # ---- Softmax 归一化 ----
        wei = F.softmax(wei, dim=-1)  # 每行的权重之和 = 1
        wei = self.dropout(wei)       # 随机丢弃一些注意力连接，防止过拟合

        # ---- 加权聚合 Value ----
        v = self.value(x)             # (B, T, head_size)
        out = wei @ v                 # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
        # out 的每个位置都是前面所有位置的 value 的加权和
        return out

# ============================================================================
# 第 2 层：多头注意力 (Multi-Head Attention)
# ============================================================================
# 为什么需要多个头？
#   不同头可以学到不同的"关注模式"——
#   比如头 1 关注语法结构，头 2 关注语义，头 3 关注位置邻近的词……
#   并行运行 n_head 个独立的自注意力头，然后把它们的输出拼起来。
# ============================================================================
class MultiHeadAttention(nn.Module):
    """并行运行多个自注意力头"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # 创建 num_heads 个独立的 Head（每个 head 有自己的 Q, K, V 权重）
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 把所有头的输出拼接后，做一个线性投影恢复到 n_embd 维度
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 每个头独立计算自注意力，然后在最后一个维度拼接
        # 6 个头 × 64 维 = 384 维 → 刚好等于 n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))  # 投影 + dropout
        return out

# ============================================================================
# 第 3 层：前馈网络 (FeedForward)
# ============================================================================
# Transformer Block 的另一半：在注意力"交流信息"之后，
# 用一个小型 MLP 对每个位置独立做非线性变换。
# 结构：Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd)
# 中间扩展 4 倍是 Transformer 论文的标准做法，给模型更大的"思考空间"
# ============================================================================
class FeedFoward(nn.Module):
    """简单的两层 MLP，中间扩展 4 倍"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # 扩展：384 → 1536
            nn.ReLU(),                         # 非线性激活
            nn.Linear(4 * n_embd, n_embd),    # 压缩回：1536 → 384
            nn.Dropout(dropout),              # 正则化
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# 第 4 层：Transformer Block（把注意力 + 前馈拼在一起）
# ============================================================================
# 这是 Transformer 的"积木"，堆叠 n_layer 个 Block 就构成了整个模型。
#
# 关键设计：Pre-LN（Pre-Layer Normalization）
#   先做 LayerNorm，再做注意力/前馈，最后残差连接。
#   x = x + Attention(LN(x))
#   x = x + FeedForward(LN(x))
#
# 残差连接 (Residual Connection) 的作用：
#   让梯度可以直接"绕过"子层传播，解决了深层网络梯度消失的问题。
#   这也是为什么 Transformer 能堆几十上百层的关键。
#
# 和 2017 年原始 Transformer 的区别：
#   原论文是 Post-LN（先子层再 LN），现代实践中 Pre-LN 训练更稳定。
# ============================================================================
class Block(nn.Module):
    """Transformer Block: 自注意力 + 前馈网络，各带残差连接"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # 每个头的维度：384 // 6 = 64
        self.sa = MultiHeadAttention(n_head, head_size)  # 多头自注意力
        self.ffwd = FeedFoward(n_embd)                    # 前馈网络
        self.ln1 = nn.LayerNorm(n_embd)  # 注意力之前的 LayerNorm
        self.ln2 = nn.LayerNorm(n_embd)  # 前馈之前的 LayerNorm

    def forward(self, x):
        # Pre-LN 结构：先归一化，再计算，最后残差连接
        x = x + self.sa(self.ln1(x))    # 自注意力 + 残差
        x = x + self.ffwd(self.ln2(x))  # 前馈网络 + 残差
        return x

# ============================================================================
# GPT 语言模型（顶层）
# ============================================================================
# 把上面所有组件组合成完整的 GPT 模型。
# GPT 本质上是 Transformer 解码器 (Decoder-only)：
#   - 没有 Encoder（不像 BERT 和原始 Transformer 的 Encoder-Decoder）
#   - 使用因果掩码（只能看过往，不能偷看未来）
#   - 自回归生成（每次预测下一个 token，拼到末尾，循环往复）
# ============================================================================
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # ---- 嵌入层 ----
        # Token 嵌入：每个字符 → n_embd 维向量（表示"这个字符是什么"）
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 位置嵌入：每个位置索引 → n_embd 维向量（表示"这个字符在第几位"）
        # 因为注意力机制本身不感知位置信息，所以需要手动注入位置编码
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # ---- Transformer 主体 ----
        # 堆叠 n_layer 个 Block，数据依次流过
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        # ---- 输出层 ----
        self.ln_f = nn.LayerNorm(n_embd)        # 最后的 LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # 把嵌入向量映射回 vocab_size 个 logits

        # ---- 权重初始化 ----
        # 良好的初始化对训练收敛速度至关重要
        # Karpathy 在视频中没详细展开，但在 README 中特别强调了这一点
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        自定义权重初始化：
        - Linear 层：均值为 0，标准差为 0.02 的正态分布
        - Embedding 层：同上
        - Bias：初始化为 0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播。
        
        参数：
            idx:     (B, T) 输入 token 索引
            targets: (B, T) 目标 token 索引，为 None 则不做损失计算
        
        流程：
            Token嵌入 + 位置嵌入 → 6层Transformer Block → LayerNorm → LM Head → logits
        """
        B, T = idx.shape

        # 第 1 步：Token 嵌入 → (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)
        # 第 2 步：位置嵌入 → (T, n_embd)，广播到 (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # 第 3 步：两者相加（这里用的不是拼接而是相加，因为维度相同）
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # 第 4 步：依次通过所有 Transformer Block
        x = self.blocks(x)  # (B, T, n_embd)

        # 第 5 步：最后的 LayerNorm + 映射到词汇表大小
        x = self.ln_f(x)                     # (B, T, n_embd)
        logits = self.lm_head(x)             # (B, T, vocab_size)

        # 第 6 步：计算损失（训练时）
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)       # 展平为 (B*T, vocab_size)
            targets = targets.view(B*T)        # 展平为 (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        自回归生成文本。
        
        参数：
            idx:            (B, T) 初始上下文
            max_new_tokens: 要生成的字符数量
        
        返回：
            idx: (B, T + max_new_tokens) 完整序列
        """
        for _ in range(max_new_tokens):
            # 如果序列长度超过 block_size，只取最后 block_size 个字符
            # 因为位置嵌入表只有 block_size 行，不能处理更长的序列
            idx_cond = idx[:, -block_size:]

            # 前向传播得到 logits
            logits, loss = self(idx_cond)
            # 只取最后一个位置的预测 → (B, vocab_size)
            logits = logits[:, -1, :]
            # softmax → 概率 → 按概率采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 拼接到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# ---------------------------------------------------------------------------
# 实例化模型
# ---------------------------------------------------------------------------
model = GPTLanguageModel()
m = model.to(device)

# 打印模型参数量（约 10.8M）
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ============================================================================
# 训练循环
# ============================================================================
for iter in range(max_iters):

    # 定期评估损失（第一步、每 eval_interval 步、最后一步）
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 标准训练四步曲
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ============================================================================
# 生成展示：从头开始，生成 500 个字符
# ============================================================================
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# 如果训练得足够好，取消下面注释可以生成 10000 个字符保存到文件
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
