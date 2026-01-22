import time
import os
from HinglishBPE import HinglishBPE

if __name__ == "__main__":

    CORPUS_FILE = "shuffled_corpus.txt"  
    VOCAB_SIZE = 32768
    MODEL_DIR = "../models"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    MODEL_PREFIX = os.path.join(MODEL_DIR, "hinglish_32k")
    
    # 2. Initialize
    tok = HinglishBPE()
    t0 = time.time()
    
    print(f"🚀 Starting training on {CORPUS_FILE}...")
    print(f"   Target Vocab Size: {VOCAB_SIZE}")
    print(f"   Model Output: {MODEL_PREFIX}.model")

    # 3. Train
    try:
        tok.train(
            filename=CORPUS_FILE,
            vocab_size=VOCAB_SIZE,
            min_word_freq=2,
            max_unique_words=3_000_000,
            verbose=True,
            checkpoint_prefix=f"{MODEL_PREFIX}_chk",
            checkpoint_interval=1000,
        )
        
        print(f"\n✅ Training took {(time.time() - t0) / 60:.2f} minutes")
        
        # 4. Save Final Model
        tok.save(MODEL_PREFIX)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find '{CORPUS_FILE}'.")
        print("   Make sure you ran 'python data/data.py' first to generate the corpus.")