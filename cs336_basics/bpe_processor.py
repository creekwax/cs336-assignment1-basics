class BpeProcessor:
    def get_vocabulary(self) -> dict[int, bytes]:
        pass

    def process(self, max_vocab_size: int) -> list[tuple[bytes, bytes]]:
        pass


def gen_bpe_freq(
    freq_dict: dict[str, int],
) -> tuple[
    dict[tuple[bytes, bytes], int],  # bpe_frequency
    dict[tuple[bytes, bytes], set[str]],  # bpe pretokens
]:
    bpe_freq = dict()
    bpe_tokens = dict()

    for pretoken, count in freq_dict.items():
        raw_bytes = pretoken.encode()
        for i in range(len(raw_bytes) - 1):
            key = (bytes([raw_bytes[i]], bytes([raw_bytes[i + 1]])))
            if bpe_freq.get(key) is None:
                bpe_freq[key] = 0
                bpe_tokens = set()

            bpe_freq[key] = bpe_freq.get(key, 0) + count
            bpe_tokens[key].add(pretoken)

    return (bpe_freq, bpe_tokens)


class RawBpeProcessor(BpeProcessor):
    def __init__(
        self,
        token_freq: dict[str, int],
        special_tokens: list[str],
    ):
        initial_vocab = dict()  # dict[int, bytes]
        for i in range(256):
            initial_vocab[i] = bytes([i])
        for st in special_tokens:
            initial_vocab[len(initial_vocab)] = st.encode()
        self.vocab = initial_vocab

        # It seems hard to maintain the bpe_tokens mapping from tuple[bytes, bytes] to the token set, when
        # it comes to merging the pairs.
        #
        # For each token in the token set, when the tuple pair is found, we would like to check the bytes prior
        # to the pair so they can assemble a new pair. But the prior bytes could also be a merged one.
        # Example:
        #   A B C D E ---> A BC D E (first merge) ---> A BC DE (second merge)
        # For the second merge, after removing `D E` pair from the bpe_tokens, we need to construct a new pair
        # of "BC DE" instead of "C DE" as "B C" has already been merged.
        #
        # We can go over the token and construct the merged prior bytes according to the already merged ones.
        # This is a repeated process, as each time we want to go through a token, we'll adopt this process to
        # reconstruct the merged pairs again.
        # ----------------
        # self.token_freq = token_freq
        # self.bpe_freq, self.bpe_tokens = gen_bpe_freq(token_freq)

        # Instead, let's use the vocabulary index to represent each pre-token.
        #
        # We store the pre-token as list[int], where the element is the indice in the vocabulary. When we are to merge
        # the bytes pair, we can manipulate the list[int] to reflect on the merge by replacing 2 indice values as the new one.
        # While the token frequency remains unchanged.
        #
        # We store the bpe_freq as dict[tuple[int,int], int] where the key is the indice pair instead of the bytes pair.
        # In this way, we're free to go through each list[int] of token to perform the merge.
        tokens, freqs = list(), list()
        bpe_freq, bpe_tokens = dict(), dict()
        for token, freq in token_freq.items():
            indices = [int(x) for x in token.encode() ]  # In the initial vocabulary, the indice is the same as the byte
            for i in range(len(indices) - 1):
                key = (indices[i], indices[i + 1])
                bpe_freq[key] = bpe_freq.get(key, 0) + freq
                bpe_tokens.setdefault(key, set()).add(len(tokens))
            tokens.append(indices)
            freqs.append(freq)

        self.tokens, self.freqs, self.bpe_freq, self.bpe_tokens = tokens, freqs, bpe_freq, bpe_tokens

    def _find_merge(self) -> tuple[int, int]:
        pair, cnt = None, 0
        for bp, count in self.bpe_freq.items():
            if count > cnt:
                pair = bp
                cnt = count
            elif count == cnt and self._compare_bytes_pair(pair, bp) < 0:
                pair = bp
        return pair

    def _compare_bytes_pair(self, pair1: tuple[int, int], pair2: tuple[int, int]) -> int:
        x1, y1 = self.vocab[pair1[0]], self.vocab[pair1[1]]
        x2, y2 = self.vocab[pair2[0]], self.vocab[pair2[1]]
        return -1 if (x1 < x2 or (x1 == x2 and y1 < y2)) else 1

    def _do_merge(self, pair: tuple[int, int]):
        merged_bytes = bytes(self.vocab[pair[0]] + self.vocab[pair[1]])
        indice = len(self.vocab)
        self.vocab[indice] = merged_bytes

        # Replace all occurances of (pair[0], pair[1]) to (indice).
        self.bpe_freq.pop(pair)
        token_set = self.bpe_tokens.pop(pair)
        for i in token_set:
            ii, indices, new_indices = 0, self.tokens[i], []
            while ii < len(indices):
                if ii < len(indices) - 1 and (indices[ii], indices[ii + 1]) == pair:
                    new_indices.append(indice)
                    ii += 2
                else:
                    new_indices.append(indices[ii])
                    ii += 1
            self.tokens[i] = new_indices

            old_pair_freq, pair_freq, token_freq = dict(), dict(), self.freqs[i]
            for j in range(len(indices) - 1):
                key = (indices[j], indices[j + 1])
                old_pair_freq[key] = old_pair_freq.get(key, 0) + token_freq
            for j in range(len(new_indices) - 1):
                key = (new_indices[j], new_indices[j + 1])
                pair_freq[key] = pair_freq.get(key, 0) + token_freq

            for key, freq in old_pair_freq.items():
                if key == pair:
                    continue

                if key not in pair_freq:  # Removed pairs
                    self.bpe_tokens[key].remove(i)
                    self.bpe_freq[key] -= freq
                    continue

                if pair_freq.get(key) != freq:  # Updated pairs
                    self.bpe_freq[key] += pair_freq[key] - freq
                pair_freq.pop(key)

            assert pair not in pair_freq
            for key, freq in pair_freq.items():  # New pairs
                self.bpe_tokens.setdefault(key, set()).add(i)
                self.bpe_freq[key] = self.bpe_freq.get(key, 0) + freq

    def get_vocabulary(self) -> dict[int, bytes]:
        return dict(self.vocab)

    def process(self, max_vocab_size: int) -> list[tuple[bytes, bytes]]:
        merges = list()

        while len(self.vocab) < max_vocab_size:
            pair = self._find_merge()
            self._do_merge(pair)

            bytes_pair = (self.vocab[pair[0]], self.vocab[pair[1]])
            merges.append(bytes_pair)
        return merges

