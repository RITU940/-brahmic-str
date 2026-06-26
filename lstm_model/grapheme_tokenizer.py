"""
Bengali Grapheme Tokenizer
==========================

Segments Bengali text into grapheme clusters and provides encode/decode
functionality for use in sequence models (e.g., BiLSTM-CTC OCR).

A Bengali grapheme cluster is defined as:
  base + (halant + consonant)* + optional_vowel_signs + optional_modifiers

Where:
  - base: a consonant, vowel (independent), digit, or other character
  - halant (্, U+09CD): virama that joins consonants into conjuncts
  - vowel signs: dependent vowel matras (া ি ী ু ূ ৃ ে ৈ ো ৌ)
  - modifiers: anusvara (ং), visarga (ঃ), chandrabindu (ঁ), nukta (়)

Examples:
  ক        -> ['ক']           (single consonant)
  ক্ষ      -> ['ক্ষ']         (conjunct: ক + ্ + ষ)
  স্ত্র    -> ['স্ত্র']       (three consonants: স + ্ + ত + ্ + র)
  বাংলা    -> ['বা', 'ং', 'লা']
"""

import json
import unicodedata
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Unicode constants for Bengali script (U+0980 – U+09FF)
# ---------------------------------------------------------------------------

# Bengali consonants
BENGALI_CONSONANTS = set(
    chr(c) for c in range(0x0995, 0x09B0 + 1)  # ক-র
) | {
    chr(0x09B2),  # ল
    chr(0x09B6),  # শ
    chr(0x09B7),  # ষ
    chr(0x09B8),  # স
    chr(0x09B9),  # হ
    chr(0x09DC),  # ড়
    chr(0x09DD),  # ঢ়
    chr(0x09DF),  # য়
    chr(0x09F0),  # ৰ (Assamese ra, sometimes in Bengali datasets)
    chr(0x09F1),  # ৱ (Assamese wa)
}

# Bengali independent vowels
BENGALI_VOWELS = set(
    chr(c) for c in range(0x0985, 0x0994 + 1)  # অ-ঔ
)

# Bengali dependent vowel signs (matras)
BENGALI_VOWEL_SIGNS = set(
    chr(c) for c in range(0x09BE, 0x09CC + 1)  # া-ৌ
) | {chr(0x09D7)}  # ৗ (au length mark)

# Halant / Virama
BENGALI_HALANT = chr(0x09CD)  # ্

# Modifiers that attach to a grapheme cluster
BENGALI_MODIFIERS = {
    chr(0x0981),  # ঁ chandrabindu
    chr(0x0982),  # ং anusvara
    chr(0x0983),  # ঃ visarga
    chr(0x09BC),  # ় nukta
}

# Bengali digits
BENGALI_DIGITS = set(chr(c) for c in range(0x09E6, 0x09EF + 1))  # ০-৯

# Special Bengali characters
BENGALI_SPECIAL = {
    chr(0x09CE),  # ৎ (khanda ta)
    chr(0x09F2),  # ৲ (rupee mark)
    chr(0x09F3),  # ৳ (rupee sign)
    chr(0x09F4),  # ৴
    chr(0x09F5),  # ৵
    chr(0x09F6),  # ৶
    chr(0x09F7),  # ৷
    chr(0x09F8),  # ৸
    chr(0x09F9),  # ৹
    chr(0x09FA),  # ৺
}

# All Bengali base characters (can start a grapheme cluster)
BENGALI_BASES = BENGALI_CONSONANTS | BENGALI_VOWELS | BENGALI_DIGITS | BENGALI_SPECIAL


def is_bengali_char(ch: str) -> bool:
    """Check if a character is in the Bengali Unicode block (U+0980-U+09FF)."""
    cp = ord(ch)
    return 0x0980 <= cp <= 0x09FF


def segment_graphemes(text: str) -> List[str]:
    """
    Segment Bengali text into grapheme clusters.
    
    Algorithm:
    1. Start with a base character (consonant, vowel, digit, etc.)
    2. If base is a consonant, greedily consume (halant + consonant) sequences
       to form conjuncts
    3. Then consume any dependent vowel signs
    4. Then consume any modifiers (anusvara, visarga, chandrabindu, nukta)
    5. Non-Bengali characters are emitted individually (space, punctuation, etc.)
    
    Returns:
        List of grapheme cluster strings
    """
    graphemes = []
    i = 0
    n = len(text)
    
    while i < n:
        ch = text[i]
        
        # Case 1: Bengali consonant -> start of potential conjunct
        if ch in BENGALI_CONSONANTS:
            cluster = ch
            i += 1
            
            # Consume halant + consonant sequences (conjuncts)
            while i < n and text[i] == BENGALI_HALANT:
                # Check if there's a consonant after the halant
                if i + 1 < n and text[i + 1] in BENGALI_CONSONANTS:
                    cluster += text[i] + text[i + 1]  # halant + consonant
                    i += 2
                else:
                    # Halant at end (hasanta form) — include it
                    cluster += text[i]
                    i += 1
                    break
            
            # Consume nukta if present (e.g., ড + ় = ড়)
            if i < n and text[i] == chr(0x09BC):
                cluster += text[i]
                i += 1
            
            # Consume vowel sign(s) — typically one, but handle edge cases
            while i < n and text[i] in BENGALI_VOWEL_SIGNS:
                cluster += text[i]
                i += 1
            
            # Consume modifiers (anusvara, visarga, chandrabindu)
            while i < n and text[i] in BENGALI_MODIFIERS:
                cluster += text[i]
                i += 1
            
            graphemes.append(cluster)
        
        # Case 2: Bengali independent vowel
        elif ch in BENGALI_VOWELS:
            cluster = ch
            i += 1
            
            # Independent vowels can have modifiers
            while i < n and text[i] in BENGALI_MODIFIERS:
                cluster += text[i]
                i += 1
            
            graphemes.append(cluster)
        
        # Case 3: Bengali digit or special character
        elif ch in BENGALI_DIGITS or ch in BENGALI_SPECIAL:
            graphemes.append(ch)
            i += 1
        
        # Case 4: Standalone modifier (shouldn't happen in well-formed text)
        elif ch in BENGALI_MODIFIERS or ch in BENGALI_VOWEL_SIGNS:
            # Attach to previous grapheme if possible, else emit standalone
            if graphemes:
                graphemes[-1] += ch
            else:
                graphemes.append(ch)
            i += 1
        
        # Case 5: Non-Bengali character (space, punctuation, Latin, etc.)
        else:
            graphemes.append(ch)
            i += 1
    
    return graphemes


class BengaliGraphemeTokenizer:
    """
    Tokenizer that operates on Bengali grapheme clusters instead of
    individual Unicode characters.
    
    Special tokens:
        0: <blank> (CTC blank)
        1: <unk>   (unknown grapheme)
    """
    
    BLANK_TOKEN = "<blank>"
    UNK_TOKEN = "<unk>"
    BLANK_IDX = 0
    UNK_IDX = 1
    
    def __init__(self):
        self.grapheme2idx: Dict[str, int] = {
            self.BLANK_TOKEN: self.BLANK_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2grapheme: Dict[int, str] = {
            self.BLANK_IDX: self.BLANK_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }
        self._next_idx = 2
        self._frozen = False  # When True, no new graphemes are added
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary from a list of text strings.
        
        Args:
            texts: List of ground-truth text strings
            min_freq: Minimum frequency for a grapheme to be included
        """
        from collections import Counter
        freq = Counter()
        
        for text in texts:
            graphemes = segment_graphemes(text)
            freq.update(graphemes)
        
        # Sort graphemes by frequency (descending) for deterministic ordering
        sorted_graphemes = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        
        for grapheme, count in sorted_graphemes:
            if count >= min_freq and grapheme not in self.grapheme2idx:
                self.grapheme2idx[grapheme] = self._next_idx
                self.idx2grapheme[self._next_idx] = grapheme
                self._next_idx += 1
        
        self._frozen = True
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.grapheme2idx)
    
    @property
    def num_classes(self) -> int:
        """Alias for vocab_size, compatible with CTC models."""
        return self.vocab_size
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of grapheme token indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of integer indices
        """
        graphemes = segment_graphemes(text)
        indices = []
        for g in graphemes:
            if g in self.grapheme2idx:
                indices.append(self.grapheme2idx[g])
            else:
                indices.append(self.UNK_IDX)
        return indices
    
    def decode(self, indices: List[int], remove_blanks: bool = True) -> str:
        """
        Decode a list of indices back into text.
        
        Args:
            indices: List of integer indices
            remove_blanks: If True, skip blank tokens
            
        Returns:
            Reconstructed text string
        """
        chars = []
        for idx in indices:
            if remove_blanks and idx == self.BLANK_IDX:
                continue
            if idx == self.UNK_IDX:
                chars.append("�")  # replacement character for unknowns
                continue
            if idx in self.idx2grapheme:
                chars.append(self.idx2grapheme[idx])
            else:
                chars.append("�")
        return "".join(chars)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Segment text into grapheme cluster tokens (strings).
        
        Args:
            text: Input text string
            
        Returns:
            List of grapheme cluster strings
        """
        return segment_graphemes(text)
    
    def get_vocab_stats(self) -> Dict:
        """Return vocabulary statistics."""
        grapheme_lengths = {}
        for g in self.grapheme2idx:
            if g in (self.BLANK_TOKEN, self.UNK_TOKEN):
                continue
            length = len(g)
            grapheme_lengths[length] = grapheme_lengths.get(length, 0) + 1
        
        return {
            "vocab_size": self.vocab_size,
            "num_special_tokens": 2,
            "num_graphemes": self.vocab_size - 2,
            "grapheme_length_distribution": grapheme_lengths,
        }
    
    def save(self, filepath: str) -> None:
        """Save tokenizer vocabulary to JSON file."""
        data = {
            "grapheme2idx": self.grapheme2idx,
            "idx2grapheme": {str(k): v for k, v in self.idx2grapheme.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "BengaliGraphemeTokenizer":
        """Load tokenizer vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.grapheme2idx = data["grapheme2idx"]
        tokenizer.idx2grapheme = {int(k): v for k, v in data["idx2grapheme"].items()}
        tokenizer._next_idx = max(tokenizer.idx2grapheme.keys()) + 1
        tokenizer._frozen = True
        return tokenizer


# ---------------------------------------------------------------------------
# Utility: compare character-level vs grapheme-level tokenization
# ---------------------------------------------------------------------------

def compare_char_vs_grapheme(texts: List[str]) -> Dict:
    """
    Compare character-level vs grapheme-level tokenization statistics.
    
    Returns:
        Dictionary with comparison metrics
    """
    all_chars = set()
    all_graphemes = set()
    total_chars = 0
    total_graphemes = 0
    
    for text in texts:
        chars = list(text)
        graphemes = segment_graphemes(text)
        
        all_chars.update(chars)
        all_graphemes.update(graphemes)
        total_chars += len(chars)
        total_graphemes += len(graphemes)
    
    return {
        "num_texts": len(texts),
        "unique_chars": len(all_chars),
        "unique_graphemes": len(all_graphemes),
        "total_char_tokens": total_chars,
        "total_grapheme_tokens": total_graphemes,
        "avg_chars_per_text": total_chars / max(len(texts), 1),
        "avg_graphemes_per_text": total_graphemes / max(len(texts), 1),
        "compression_ratio": total_chars / max(total_graphemes, 1),
        "vocab_reduction_pct": (1 - len(all_graphemes) / max(len(all_chars), 1)) * 100,
    }
