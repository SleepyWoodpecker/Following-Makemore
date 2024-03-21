"""Final, inefficient implementation of BPE for tiktoken ck100L"""

import tiktoken
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


enc = tiktoken.get_encoding("cl100k_base")  # this is the GPT-4 tokenizer


def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )
    return parts


def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges


class GPT4Tokenizer:
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self):
        # get the official tokenizer and its merges
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        # the merges are those of gpt4, but we have to recover them
        self.merges = recover_merges(mergeable_ranks)
        self.reversed_merges = list(
            reversed(
                {
                    new_token: original_token
                    for original_token, new_token in self.merges.items()
                }.items()
            )
        )
        # reconstruct the vocab from the merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # now here is another tricky part.
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here.
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

    def _merge(self, text_bytes, target_pair, new_token):
        """Do a pure merge"""
        new_chunk = []
        skip_next = False

        for pair in zip(text_bytes, text_bytes[1:]):
            if skip_next:
                skip_next = False
                continue

            if pair == target_pair:
                # print(pair, new_token)
                new_chunk.append(new_token)
                # since 2 tokens were compressed into 1, skip the next token
                skip_next = True

            else:
                new_chunk.append(pair[0])

        new_chunk += [text_bytes[-1]] if not skip_next else []
        return new_chunk

    def encode(self, text):
        """given text, encode it based on the merge dictionary"""
        # split the text up into chunks before doing the encoding (since that was how it was trained)
        txt_bytes = [
            list(chunk.encode("utf-8"))
            for chunk in re.findall(GPT4_SPLIT_PATTERN, text)
        ]
        # account for byte shuffling
        shuffled_bytes = []
        for chunk in txt_bytes:
            new_chunk = []
            for by in chunk:
                new_chunk.append(self.byte_shuffle[by])

            shuffled_bytes.append(new_chunk)

        # follow the defined merges in order
        for target, new_token in self.merges.items():
            # print(shuffled_bytes)
            new_bytes = []
            for chunk in shuffled_bytes:
                # print(chunk)
                new_bytes.append(self._merge(chunk, target, new_token))
            shuffled_bytes = new_bytes

        # flatten the final array
        return [b for word in shuffled_bytes for b in word]

    def decode(self, tokens):
        "given a set of encoded tokens, decode them into text"

        for new_token, target in self.reversed_merges:
            new_sequence = []
            for token in tokens:
                if token == new_token:
                    new_sequence.extend(target)

                else:
                    new_sequence.append(token)

            tokens = new_sequence

        # unshuffle the tokens
        unshuffled = [self.inverse_byte_shuffle[t] for t in tokens]

        text = (b"".join([self.vocab[u] for u in unshuffled])).decode("utf-8")
        return text


if __name__ == "__main__":
    gpt4 = GPT4Tokenizer()
    r = gpt4.encode("hello world!!!? (안녕하세요!) lol123 😉")
    ids = enc.encode("hello world!!!? (안녕하세요!) lol123 😉")

    print(r)
    print(ids)
    print("\n\n")

    print(gpt4.decode(r))
