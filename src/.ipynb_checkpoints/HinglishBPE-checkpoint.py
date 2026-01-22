import regex as re
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Dict
import json
import time
import os

# these two functions from Andrej Karpathy's minbpe(Let's build the GPT Tokenizer YT video)

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


#used Ai Tools for this

HINGLISH_PATTERN = r"""
(?:https?://\S+)                              # URLs
|(?:@\w{1,64})                                # @mentions
|(?:\#\w+)                                    # Hashtags
|(?:\p{Extended_Pictographic}+)               # Emojis (grouped)
|(?i:(?:'s|'t|'re|'ve|'m|'ll|'d))             # English contractions (case-insensitive)
|(?:\s?\p{Script=Devanagari}+)                # Devanagari words (optional leading whitespace)
|(?:\s?\p{Script=Latin}+)                     # Latin words (English / Romanized, optional leading whitespace)
|(?:\p{N}+(?:[.,:/\-]\p{N}+)*)                # Numbers, dates, ranges
|(?:[^\s\p{L}\p{N}]+)                         # Punctuation / symbols
|\s+                                          # Whitespace (preserved separately)
"""

# Tokenizer class

class HinglishBPE:
    def __init__(self,pattern:str= HINGLISH_PATTERN):
        self.pattern=pattern.strip()
        self.compiled=re.compile(self.pattern,re.VERBOSE)
        self.merges:Dict[Tuple[int,int],int]={}
        self.merges_list:List[Tuple[int,int]]=[]
        self.vocab:Dict[int,bytes]={i:bytes([i]) for i in range(256)}
        self.special_tokens:Dict[str,int]={}
        self.inverse_special:Dict[int,str]={}
    def train(self,
              filename:str,
              vocab_size:int,
              min_word_freq:int=2,
              max_unique_words:int=3000000,
              verbose:bool=True,
              checkpoint_prefix: str=None,
              checkpoint_interval: int=1000,
             ):
        num_merges=vocab_size-256
        print(f"[1/3] Counting pre-tokens in {filename}")
        word_counts=Counter()
        with open(filename,"r",encoding="utf-8",errors="replace") as fin:
            for line in tqdm(fin,desc="counting lines",unit="line"):
                chunks=self.compiled.findall(line)
                if not chunks:
                    continue
                filtered=[c for c in chunks if c and not c.isspace()]
                if filtered:
                    word_counts.update(filtered)
        print(f"Raw unique pre-tokens:{len(word_counts)}")
        if min_word_freq>1:
            before=len(word_counts)
            word_counts=Counter({w:c for w,c in word_counts.items() if c>=min_word_freq})
            print(f"After min_freq >= {min_word_freq}: {len(word_counts):,} (removed {before - len(word_counts):,})")

        if len(word_counts)> max_unique_words:
            print(f"Capping unique pre-tokens to top {max_unique_words:,}")
            
            word_counts=Counter(dict(word_counts.most_common(max_unique_words)))

        train_data:List[List]=[]
        for w,c in word_counts.items():
            train_data.append([list(w.encode("utf-8")),c])
        word_counts=None

        #step 2
        print(f"[2/3] Performing {num_merges:,} BPE merges ...")
        start_time=time.time()
        merges:Dict[Tuple[int,int],int]={}
        merges_list:List[Tuple[int,int]]={}
        local_vocab={i:bytes([i]) for i in range(256)}

        for i in tqdm(range(num_merges),desc="Merges",unit="merge"):
            stats: Dict[Tuple[int,int],int]={}
            for ids,freq in train_data:
                if(len(ids))<2:
                    continue
                for pair in zip(ids,ids[1:]):
                    stats[pair]=stats.get(pair,0)+freq
            if not stats:
                if verbose:
                    print("No more pairs to merge — early termination.")
                break

            best_pair=max(stats,key=lambda p:(stats[p],-p[0],-p[1]))
            best_freq=stats[best_pair]
            new_id=256+i
            for item in train_data:
                if len(item[0])<2:
                    continue
                item[0]=merge(item[0],best_pair,new_id)

            self.merges[best_pair]=new_id
            self.merges_list.append(best_pair)
            local_vocab[new_id]= local_vocab[best_pair[0]]+local_vocab[best_pair[1]]
            if checkpoint_prefix and (i + 1) % checkpoint_interval == 0:
                saved_prefix = f"{checkpoint_prefix}_merge_{i+1}"
                self._save_partial(saved_prefix, local_vocab)
                if verbose:
                    print(f"[checkpoint] saved {saved_prefix} at merge {i+1}")

            if verbose and (i < 20 or (i + 1) % 500 == 0 or i == num_merges - 1):
                elapsed = time.time() - start_time
                print(f"  Merge {i+1:>6}/{num_merges}: {best_pair} -> {new_id}  (freq={best_freq:,}) elapsed={elapsed:.1f}s")
        self.vocab = local_vocab

        print(f"[3/3] Training complete. Final vocab size:{len(self.vocab):,}")


    def _save_partial(self,prefix:str,local_vocab:Dict[int,bytes]):
        
        model_file = prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(self.pattern.replace("\n", "\\n") + "\n")
            f.write(f"{len(self.special_tokens)}\n")
            for s, idx in self.special_tokens.items():
                f.write(f"{s} {idx}\n")
            for pair in self.merges_list:
                f.write(f"{pair[0]} {pair[1]}\n")
        # write vocab
        vocab_file = prefix + ".vocab.json"
        hexmap = {str(k): v.hex() for k, v in local_vocab.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(hexmap, f, ensure_ascii=False, indent=2)

    # Encoding and Decoding

    def _encode_chunk(self,text_bytes: bytes) -> List[int]:
        ids= list(text_bytes)
        while len(ids)>=2:
            stats=get_stats(ids)
            pair=min(stats,key=lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            ids= merge(ids,pair,self.merges[pair])
        return ids

    def encode(self,text: str)-> List[int]:
        chunks=self.compiled.findall(text)
        out: List[int]=[]
        for chunk in chunks:
            if chunk is None:
                continue
            if chunk.isspace():
                out.extend(list(chunk.encode("utf-8")))
                continue
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def decode(self,ids:List[int])-> str:
        parts: List[bytes]=[]
        for i in ids:
            b=self.vocab.get(i)
            if b is not None:
                parts.append(b)
            else:
                if 0<=i<=255:
                    parts.append(nytes([i]))
                else:
                    continue
        return b"".join(parts).decode("utf-8", errors="replace")

    # save and load

    def save(self, prefix: str):
        model_file = prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(self.pattern.replace("\n", "\\n") + "\n")
            f.write(f"{len(self.special_tokens)}\n")
            for s, idx in self.special_tokens.items():
                f.write(f"{s} {idx}\n")
            for a, b in self.merges_list:
                f.write(f"{a} {b}\n")

        vocab_file = prefix + ".vocab.json"
        hexmap = {str(k): v.hex() for k, v in self.vocab.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(hexmap, f, ensure_ascii=False, indent=2)

        print(f"Saved → {model_file} and {vocab_file}")

    def load(self, prefix: str):
        model_file = prefix + ".model"
        vocab_file = prefix + ".vocab.json"
        with open(model_file, "r", encoding="utf-8") as f:
            assert f.readline().strip() == "minbpe v1"
            self.pattern = f.readline().rstrip("\n").replace("\\n", "\n")
            self.compiled = re.compile(self.pattern, re.VERBOSE)
            num_special = int(f.readline().strip())
            self.special_tokens = {}
            self.inverse_special = {}
            for _ in range(num_special):
                line = f.readline().strip()
                if not line:
                    continue
                s, idx_str = line.split()
                idx = int(idx_str)
                self.special_tokens[s] = idx
                self.inverse_special[idx] = s
            self.merges = {}
            self.merges_list = []
            next_id = 256
            for line in f:
                if not line.strip():
                    continue
                a, b = map(int, line.strip().split())
                pair = (a, b)
                self.merges_list.append(pair)
                self.merges[pair] = next_id
                next_id += 1
        # load vocab
        with open(vocab_file, "r", encoding="utf-8") as f:
            hexmap = json.load(f)
            self.vocab = {int(k): bytes.fromhex(v) for k, v in hexmap.items()}
        print(f"Loaded tokenizer from {prefix}. (vocab size: {len(self.vocab):,})")