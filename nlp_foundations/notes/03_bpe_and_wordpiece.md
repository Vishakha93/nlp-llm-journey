# Byte Pair Encoding (BPE) and WordPiece

## Why do we need subword methods?

Full-word vocabularies have problems:

- **Out-of-vocabulary (OOV)** words: `unbelievableness`, typos, names, etc.
- **Huge vocabularies**: memory + softmax become expensive.
- **Morphology**: words like `unbelievable`, `unbelievably`, `unbelievableness` all share structure.

Subword methods (BPE, WordPiece, SentencePiece) try to balance:
- Not too many unique tokens (like full-word vocab),
- Not too long sequences (like pure character-level models).

---

## BPE: High-level idea

1. Start with a **character-level vocabulary** (e.g., `u`, `n`, `b`, `e`, …).
2. Look at a large corpus and **count how often adjacent symbol pairs** occur.
3. Repeatedly **merge the most frequent pair** into a new symbol:
   - merge `e` + `r` → `er`
   - merge `un` + `believable` → `unbelievable` (eventually)
4. After many merges, you get a vocabulary of **subword units** that capture common pieces.

---

## Example: Human morpheme-style split

Consider the word:

> `unbelievable`

From a *morpheme* perspective, a human might segment it as:

- `un` (prefix, “not”)
- `believe` (root)
- `able` (suffix, “capable of”)

So:
```text
unbelievable → un + believe + able
```

Aha moment: Visit: https://www.bpe-visualizer.com/ to vizulaize how BPE works! 

TODO:Add a notebook bpe_demo.ipynb 
