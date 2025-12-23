import json
import regex as re
from typing import Dict, List, Tuple, Optional


def bytes_to_unicode() -> Dict[int, str]:
    """
    与 CLIP / GPT 系列相同的 bytes -> unicode 映射。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]):
    """
    返回一个 word 中相邻 symbol 的 pair 集合。
    word 是一个由字符串组成的 tuple。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class SimpleCLIPBPETokenizer:
    """
    纯 Python 的 CLIP BPE tokenizer 简化实现：
      - 从 vocab.json / merges.txt 加载词表与 BPE 规则
      - 实现 bytes_to_unicode + BPE + 正则分词
      - 提供 encode(text) -> (input_ids, attention_mask)

    用法示例：
        tokenizer = SimpleCLIPBPETokenizer(
            vocab_file="models/vocab.json",
            merges_file="models/merges.txt",
            max_length=32,
            bos_token_id=49406,
            eos_token_id=49407,
        )
        ids, mask = tokenizer.encode("ear")
    """

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        max_length: int = 32,
        bos_token_id: int = 49406,
        eos_token_id: int = 49407,
        added_tokens: Optional[Dict[str, int]] = None,
        bpe_vocab_size: int = 49152,  # 一般和 merges 文件里取的行数一致
        do_lower_case: bool = True,
    ) -> None:
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.do_lower_case = do_lower_case

        # 特殊 token 映射（可以通过入参覆盖）
        if added_tokens is None:
            added_tokens = {
                "<|startoftext|>": bos_token_id,
                "<|endoftext|>": eos_token_id,
            }
        self.added_tokens: Dict[str, int] = added_tokens
        self.unk_token: str = "<|endoftext|>"  # CLIP 里就是用这个当 unk

        # 正则分词规则
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
            r"""[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        # bytes -> unicode 映射
        self.byte_encoder = bytes_to_unicode()

        # 读取 merges，构建 bpe_ranks
        with open(merges_file, encoding="utf-8") as f:
            # 跳过第一行 "## merges" 之类，从第 2 行开始
            merges = f.read().strip().split("\n")[1 : bpe_vocab_size - 256 - 2 + 1]
        merges_pairs = [tuple(m.split()) for m in merges]
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges_pairs)
        }

        # 读取 vocab.json
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)

    # ----------------- 核心 BPE -----------------

    def _bpe(self, token: str) -> str:
        """
        单个 token 的 BPE 编码，返回空格分隔的子词字符串。
        """
        bpe_ranks = self.bpe_ranks

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float("inf")))
            if bigram not in bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        return " ".join(word)

    # ----------------- 分词与编码 -----------------

    def _tokenize(self, text: str) -> List[str]:
        """
        文本 -> BPE token（字符串）列表。
        """
        if self.do_lower_case:
            text = text.lower()

        bpe_tokens: List[str] = []

        for token in self.pat.findall(text):
            # bytes -> unicode string
            token_encoded = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            bpe_out = self._bpe(token_encoded)
            bpe_tokens.extend(bpe_out.split(" "))

        return bpe_tokens

    def _token_to_id(self, token: str) -> int:
        if token in self.added_tokens:
            return self.added_tokens[token]
        # unk 用 eos 替代（与原实现对齐）
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # ----------------- 对外接口：encode -----------------

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        """
        将文本编码为：
          - input_ids: 长度 max_length 的 id 序列
          - attention_mask: 同长度 mask（有效位置为 1，padding 为 0）
        """
        # 1. 先 BPE 分词
        bpe_tokens = self._tokenize(text)

        # 2. token -> id（此时不含 BOS/EOS）
        ids = [self._token_to_id(tok) for tok in bpe_tokens]
        len_ids = len(ids)

        # 3. 加 BOS / EOS
        num_special = 2  # BOS + EOS
        total_len = len_ids + num_special

        ids = [self.bos_token_id] + ids + [self.eos_token_id]

        # 4. 截断到 max_length
        ids = ids[: self.max_length]

        # 5. 有效长度（包含 BOS/EOS），与 HF attention_mask 逻辑对齐
        valid_len = min(total_len, self.max_length)

        # 6. 不足补 pad（用 eos_token_id 填充，CLIP 就是这么做的）
        if len(ids) < self.max_length:
            ids = ids + [self.eos_token_id] * (self.max_length - len(ids))

        # 7. attention_mask：前 valid_len 为 1，后面为 0
        attention_mask = [1] * valid_len + [0] * (self.max_length - valid_len)

        return ids, attention_mask


if __name__ == "__main__":
    # 简单测试
    vocab_file = "models/vocab.json"
    merges_file = "models/merges.txt"

    tokenizer = SimpleCLIPBPETokenizer(
        vocab_file=vocab_file,
        merges_file=merges_file,
        max_length=32,
        bos_token_id=49406,
        eos_token_id=49407,
        bpe_vocab_size=49152,
    )

    prompt = "person"
    ids, mask = tokenizer.encode(prompt)
    print("ids:", ids)
    print("len(ids):", len(ids))
    print("mask:", mask)
    print("sum(mask):", sum(mask))