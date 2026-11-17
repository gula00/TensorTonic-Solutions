from collections import Counter
from itertools import pairwise

def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    vocab = {i: bytes([i]) for i in range(256)}
    sequences = [list(text.encode("utf-8")) for text in corpus]
    merges = []

    for new_id in range(256, vocab_size):
        # 1. 一行完成所有相邻 token 对的频次统计
        counts = Counter(pair for seq in sequences for pair in pairwise(seq))
        if not counts:
            break

        # 2. 按 (频次, 左token字节, 右token字节) 选出最优合并对
        left, right = max(counts, key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]]))
        vocab[new_id] = vocab[left] + vocab[right]
        merges.append([left, right, new_id])

        # 3. 序列替换
        for seq in sequences:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == left and seq[i + 1] == right:
                    seq[i : i + 2] = [new_id]  # 切片原地替换，自动跳过已合并元素
                i += 1

    return {
        "vocab": [[i, list(vocab[i])] for i in range(256, len(vocab))],
        "merges": merges,
    }