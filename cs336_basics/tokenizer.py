from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
import json
import regex as re
from .pretokenization import GPT_PRETOKEN_REGEX


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = {}  # bytes -> int
        for k, v in vocab.items():
            self.vocab[v] = k
        if special_tokens is not None:
            size = len(self.vocab)
            for special_token in special_tokens:
                if special_token.encode() in self.vocab:
                    continue
                self.vocab[special_token.encode()] = size
                size += 1

        self.reversed_vocab = {}
        for k, v in self.vocab.items():
            self.reversed_vocab[v] = k

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab, rev_vocab = {}, {}
        with open(vocab_filepath) as f:
            vocab_data = json.load(f)
            for k, v in vocab_data.items():
                vocab[k.encode()] = v
        if special_tokens is not None:
            size = len(vocab)
            for special_token in special_tokens:
                if special_token.encode() in vocab:
                    continue
                vocab[special_token.encode()] = size
                size += 1
        for k, v in vocab.items():
            rev_vocab[v] = k

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                elements = line.split()
                if len(elements) == 2:
                    merges.append((elements[0].encode(), elements[1].encode()))

        tokenizer = cls()
        tokenizer.vocab = vocab
        tokenizer.merges = merges
        tokenizer.reverse_vocab = rev_vocab
        tokenizer.special_tokens = special_tokens
        return tokenizer

    def encode(self, text: str) -> list[int]:
        segments = [text]
        if self.special_tokens is not None:
            pattern = "|".join([re.escape(st) for st in self.special_tokens])
            segments = re.split(f'({pattern})', text)

        result = []
        # for segment in segments:
        #     result += self._encode_segment(segment)
        with ProcessPoolExecutor() as executor:
            for x in executor.map(self._encode_segment, segments):
                result += x
        return result

    def _encode_segment(
        self, text: str
    ) -> list[int]:  # segment without special_token
        if self.special_tokens is not None and text in self.special_tokens:
            return [ self.vocab[text.encode()] ]

        result = []
        matches = re.finditer(GPT_PRETOKEN_REGEX, text)
        for match in matches:
            pretoken = match.group().encode()
            result += self._encode_token(pretoken)
        return result

    def _encode_token(self, token: bytes) -> list[int]:
        bytes_list = [bytes([i]) for i in token]
        for merge in self.merges:
            merged_bytes = []
            i, size = 0, len(bytes_list)
            while i < size:
                if i == size - 1 or merge != (bytes_list[i], bytes_list[i + 1]):
                    merged_bytes.append(bytes_list[i])
                else:
                    merged_bytes.append(bytes(merge[0] + merge[1]))
                    i += 1
                i += 1
            bytes_list = merged_bytes
        return [self.vocab[x] for x in bytes_list]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        result = bytes([])
        for id in ids:
            value = self.reversed_vocab[id]
            if value is not None:
                result += value
        return result.decode("utf-8", errors="replace")
