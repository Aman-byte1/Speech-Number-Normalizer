import os
import re
import json
import torch
import torchaudio
from datasets import load_dataset
from word2number import w2n
from speech_number_norm import SpeechNumberNormalizer
from tqdm import tqdm

# List of common number words to filter for
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion"
}

def denormalize_text(text):
    """
    Heuristically convert number words back to digits.
    Example: "I have twenty two apples" -> "I have 22 apples"
    """
    words = text.lower().split()
    new_words = []
    current_num_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in NUMBER_WORDS or (clean_word == "and" and current_num_words):
            current_num_words.append(word)
        else:
            if current_num_words:
                try:
                    # Try to convert the sequence of words to a number
                    num_str = " ".join(current_num_words)
                    # Handle cases where "and" is just a conjunction
                    if num_str.strip() == "and":
                        new_words.append("and")
                    else:
                        num = w2n.word_to_num(num_str)
                        new_words.append(str(num))
                except:
                    new_words.extend(current_num_words)
                current_num_words = []
            new_words.append(word)
            
    if current_num_words:
        try:
            num = w2n.word_to_num(" ".join(current_num_words))
            new_words.append(str(num))
        except:
            new_words.extend(current_num_words)
            
    return " ".join(new_words)

def main():
    print("Loading a small subset of LibriSpeech dataset (clean, test)...")
    # Load 5% of the test split to avoid streaming issues
    try:
        dataset = load_dataset("librispeech_asr", "clean", split="test[:5%]")
        print(f"Loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    normalizer = SpeechNumberNormalizer(device="cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    count = 0
    max_samples = 50
    
    print(f"Evaluating on up to {max_samples} samples containing numbers...")
    
    pbar = tqdm(total=max_samples)
    for item in dataset:
        original_text = item["text"].lower()
        
        # Check if text contains numbers
        has_number = any(word in original_text.split() for word in NUMBER_WORDS)
        if not has_number:
            continue
            
        # Denormalize (create our test input)
        raw_text = denormalize_text(original_text)
        
        # Skip if denormalization didn't change anything (no digits found)
        if raw_text == original_text:
            continue
            
        # Save audio to temp file for the normalizer
        audio_path = "temp_eval.wav"
        waveform = torch.tensor(item["audio"]["array"]).unsqueeze(0)
        sample_rate = item["audio"]["sampling_rate"]
        torchaudio.save(audio_path, waveform, sample_rate)
        
        # Normalize
        norm_result = normalizer.normalize(raw_text, audio_path, language="en")
        normalized_text = norm_result.normalized_text.lower()
        
        # Score: We check if the numbers were correctly expanded back to the original words
        # Note: LibriSpeech transcript might have slight variations (e.g. "twenty-two" vs "twenty two")
        # We clean both for a fairer comparison
        def clean(t):
            return re.sub(r'[^a-z0-9\s]', '', t.replace("-", " "))
        
        is_correct = clean(normalized_text) == clean(original_text)
        
        results.append({
            "original": original_text,
            "raw_input": raw_text,
            "normalized": normalized_text,
            "correct": is_correct
        })
        
        count += 1
        pbar.update(1)
        if count >= max_samples:
            break
            
    pbar.close()
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = (correct_count / len(results)) * 100 if results else 0
    
    print(f"\nEvaluation complete!")
    print(f"Total samples with numbers: {len(results)}")
    print(f"Accuracy (exact match): {accuracy:.2f}%")
    
    # Show some examples
    print("\nSample Results:")
    for i, r in enumerate(results[:5]):
        print(f"\nExample {i+1}:")
        print(f"  Original: {r['original']}")
        print(f"  Raw Input: {r['raw_input']}")
        print(f"  Normalized: {r['normalized']}")
        print(f"  Result: {'✅' if r['correct'] else '❌'}")

    # Clean up
    if os.path.exists("temp_eval.wav"):
        os.remove("temp_eval.wav")

if __name__ == "__main__":
    main()
