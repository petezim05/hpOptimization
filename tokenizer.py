import torch

# integers 1-1024 map to token IDs 1-1024 (0 is padding, never a valid layer width)
# token ID 1025 is reserved as end-of-sequence (stop adding layers)
VOCAB_SIZE = 513
EOS = 513

class ArchTokenizer:
    vocab_size: int = VOCAB_SIZE

    def encode(self, params: list[int]) -> torch.LongTensor:
        """params list -> token IDs with EOS appended"""
        return torch.tensor(params + [EOS], dtype=torch.long)

    def decode(self, tokens: torch.LongTensor) -> list[int]:
        """token IDs -> params list, stopping at EOS"""
        result = []
        for t in tokens:
            t = t.item()
            if t == EOS:
                break
            result.append(t)
        return result
