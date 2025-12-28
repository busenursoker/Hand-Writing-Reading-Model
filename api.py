import os
import uuid
import tempfile
import shutil
import subprocess
import hashlib
import re
from pathlib import Path
from itertools import product
from collections import Counter
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# AttentionHTR model/ klasörü
ATTN_MODEL_DIR = Path("/home/beys/Text_Reco/HTR/model")

# Model dosyası (absolute)
SAVED_MODEL = ATTN_MODEL_DIR / "saved_model" / "AttentionHTR-General-sensitive.pth"


@app.get("/health")
def health():
    return {"ok": True}


# -----------------------------
# Spell Correction (Norvig's approach)
# -----------------------------

# Word dictionaries - will be loaded on demand
WORD_DICTS = {
    "en": None,  # English
    "tr": None,  # Turkish
}

def load_word_dict(language: str) -> Counter:
    """Load word dictionary for a language."""
    if WORD_DICTS[language] is not None:
        return WORD_DICTS[language]
    
    dict_path = Path(__file__).parent / "word_dicts" / f"{language}.txt"
    
    if not dict_path.exists():
        print(f"[SPELL] Warning: Dictionary not found at {dict_path}, using empty dict")
        WORD_DICTS[language] = Counter()
        return WORD_DICTS[language]
    
    try:
        words = re.findall(r'\w+', dict_path.read_text(encoding='utf-8').lower())
        WORD_DICTS[language] = Counter(words)
        print(f"[SPELL] Loaded {len(WORD_DICTS[language])} words for language '{language}'")
        return WORD_DICTS[language]
    except Exception as e:
        print(f"[SPELL] Error loading dictionary: {e}")
        WORD_DICTS[language] = Counter()
        return WORD_DICTS[language]

def P(word: str, WORDS: Counter) -> float:
    """Probability of `word`."""
    if not WORDS:
        return 0.0
    N = sum(WORDS.values())
    return WORDS[word] / N if N > 0 else 0.0

def edits1(word: str) -> set:
    """All edits that are one edit away from `word`."""
    letters = 'abcdefghijklmnopqrstuvwxyzçğıöşüÇĞİÖŞÜ'  # Include Turkish chars
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word: str) -> set:
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def known(words: set, WORDS: Counter) -> set:
    """The subset of `words` that appear in the dictionary."""
    return set(w for w in words if w in WORDS)

def candidates(word: str, WORDS: Counter) -> list:
    """Generate possible spelling corrections for word."""
    word_lower = word.lower()
    known_words = known({word_lower}, WORDS)
    if known_words:
        return list(known_words)
    
    # Only try edits1 if word is not too short (short words are often correct)
    if len(word_lower) >= 3:
        known_edits1 = known(edits1(word_lower), WORDS)
        if known_edits1:
            return list(known_edits1)
    
    # Only try edits2 for longer words (4+ chars) to avoid over-correction
    if len(word_lower) >= 4:
        known_edits2 = known(edits2(word_lower), WORDS)
        if known_edits2:
            return list(known_edits2)
    
    return [word_lower]  # Return original if no corrections found

def spell_correct(word: str, language: str = "en") -> str:
    """Most probable spelling correction for word."""
    if not word or len(word) < 2:
        return word
    
    WORDS = load_word_dict(language)
    if not WORDS:
        return word  # No dictionary, return as-is
    
    # Check if word is already in dictionary (exact match, case-insensitive)
    word_lower = word.lower()
    if word_lower in WORDS:
        # Word is in dictionary, don't correct it
        return word
    
    cands = candidates(word, WORDS)
    if not cands:
        return word
    
    # If the only candidate is the original word, return it
    if len(cands) == 1 and cands[0] == word_lower:
        return word
    
    # Return the candidate with highest probability
    best = max(cands, key=lambda c: P(c, WORDS))
    
    # Only correct if the correction is significantly better
    # or if the original word is not in dictionary
    original_prob = P(word.lower(), WORDS)
    best_prob = P(best, WORDS)
    
    # If original is in dict, always keep it (don't correct)
    if original_prob > 0:
        return word
    
    # For words not in dictionary, be more conservative:
    # Only correct if:
    # 1. The correction has significant probability (not just a random match)
    # 2. The word is long enough (short words like "dil" should be kept if close)
    # 3. The edit distance is reasonable
    
    # Calculate edit distance between original and best candidate
    def edit_distance(s1, s2):
        if len(s1) < len(s2):
            return edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    distance = edit_distance(word.lower(), best)
    word_len = len(word.lower())
    
    # Don't correct if:
    # - Word is very short (2-3 chars) and distance > 1
    # - Word is medium (4-5 chars) and distance > 2
    # - Best candidate has very low probability (likely wrong match)
    if word_len <= 3 and distance > 1:
        return word  # Keep short words if edit distance is large
    if word_len <= 5 and distance > 2:
        return word  # Keep medium words if edit distance is too large
    if best_prob < 1e-6:  # Very low probability, likely wrong
        return word
    
    # Only correct if best has reasonable probability
    if best_prob < 1e-5:
        return word
    
    # Preserve case
    if word.isupper():
        return best.upper()
    elif word[0].isupper():
        return best.capitalize()
    else:
        return best

def spell_correct_sentence(sentence: str, language: str = "en") -> str:
    """Apply spell correction to each word in a sentence."""
    if not sentence:
        return sentence
    
    words = sentence.split()
    corrected = [spell_correct(word, language) for word in words]
    return " ".join(corrected)

# -----------------------------
# Turkish post-processing (conservative)
# -----------------------------
TR_MAP = {
    "c": ["c", "ç"],
    "g": ["g", "ğ"],
    "o": ["o", "ö"],
    "u": ["u", "ü"],
    "s": ["s", "ş"],
    "C": ["C", "Ç"],
    "G": ["G", "Ğ"],
    "O": ["O", "Ö"],
    "U": ["U", "Ü"],
    "S": ["S", "Ş"],
    "I": ["I", "İ"],  # İstanbul gibi
}

def _basic_cleanup(s: str) -> str:
    s = (s or "").strip()
    # Remove special tokens that might appear in model output
    s = re.sub(r'\[s\]', '', s, flags=re.IGNORECASE)  # Remove end-of-sentence token
    s = re.sub(r'\[GO\]', '', s, flags=re.IGNORECASE)  # Remove start token
    s = re.sub(r'\[UNK\]', '', s, flags=re.IGNORECASE)  # Remove unknown token
    s = re.sub(r'\[PAD\]', '', s, flags=re.IGNORECASE)  # Remove padding token
    
    # Remove hyphens and underscores specifically (they might be model artifacts)
    s = s.replace('-', '')
    s = s.replace('_', '')
    
    # Remove other unwanted punctuation but keep letters (including Turkish) and numbers
    # \w in Python includes Unicode word characters (letters, digits, underscore)
    # But we already removed underscore, so this keeps letters and digits
    # We'll be more explicit to preserve Turkish characters
    s = re.sub(r'[^\w\s]', '', s, flags=re.UNICODE)  # Keep word chars (letters, digits) and spaces
    
    # model boşluk üretmediği için: boşlukları tamamen kaldırmak makul
    s = re.sub(r"\s+", "", s)
    return s

def _generate_candidates_conservative(word: str, max_changes: int = 2, max_candidates: int = 120) -> list[str]:
    options = [TR_MAP.get(ch, [ch]) for ch in word]
    cands = []
    for tup in product(*options):
        cand = "".join(tup)
        changes = sum(1 for a, b in zip(word, cand) if a != b)
        if changes <= max_changes:
            cands.append(cand)
            if len(cands) >= max_candidates:
                break
    return cands

def _score_candidate(src: str, cand: str) -> float:
    changes = sum(1 for a, b in zip(src, cand) if a != b)
    score = -1.2 * changes

    # Türkçe karakter küçük bonus
    tr_bonus = sum(1 for ch in cand if ch in "çğıöşüÇĞİÖŞÜ")
    score += 0.35 * tr_bonus

    # ş/ç biraz daha güvenli
    score += 0.05 * sum(1 for ch in cand if ch in "şçŞÇ")

    return score

def segment_words_from_image(image_path: str, debug_out_path: str | None = None, adaptive: bool = True):
    """
    Segment a (mostly single-line) handwriting image into word crops using spacing.
    Returns: (img_bgr, boxes) where boxes are (x,y,w,h) in reading order.
    
    Args:
        image_path: Path to input image
        debug_out_path: Optional path to save debug visualization
        adaptive: If True, tries multiple segmentation approaches and picks the best
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions early for use in thresholding
    h, w = gray.shape
    print(f"[SEGMENT] Image size: {w}x{h}")
    
    if adaptive:
        # Try multiple segmentation approaches and pick the one with most reasonable boxes
        results = []
        
        # Approach 1: Standard (current method)
        try:
            boxes1 = _segment_words_core(img, gray, w, h, kx_ratio=80, debug_out_path=None)
            if boxes1 and len(boxes1) >= 2:  # At least 2 words
                results.append((boxes1, "standard"))
        except Exception as e:
            print(f"[SEGMENT] Standard approach failed: {e}")
        
        # Approach 2: More aggressive dilation (for closely spaced words)
        try:
            boxes2 = _segment_words_core(img, gray, w, h, kx_ratio=100, debug_out_path=None)
            if boxes2 and len(boxes2) >= 2:
                results.append((boxes2, "aggressive"))
        except Exception as e:
            print(f"[SEGMENT] Aggressive approach failed: {e}")
        
        # Approach 3: Less aggressive (for well-spaced words)
        try:
            boxes3 = _segment_words_core(img, gray, w, h, kx_ratio=60, debug_out_path=None)
            if boxes3 and len(boxes3) >= 2:
                results.append((boxes3, "conservative"))
        except Exception as e:
            print(f"[SEGMENT] Conservative approach failed: {e}")
        
        # Pick the best result: prefer reasonable number of boxes (not too few, not too many)
        if results:
            # Estimate expected number of words based on image width
            # Rough estimate: average word width is about 5-10% of image width
            estimated_words = max(2, min(15, w // 100))  # Reasonable range: 2-15 words
            
            # Filter out results with too many boxes (likely over-segmentation)
            # Allow up to 3x estimated words to account for some over-segmentation
            max_boxes = max(20, estimated_words * 3)
            filtered = [(b, name) for b, name in results if len(b) <= max_boxes]
            
            if filtered:
                # Prefer results closest to estimated word count
                best_boxes, best_name = min(filtered, key=lambda x: abs(len(x[0]) - estimated_words))
                print(f"[SEGMENT] Selected {best_name} approach with {len(best_boxes)} boxes (estimated: {estimated_words})")
                boxes = best_boxes
            else:
                # Fall back to result with fewest boxes (less over-segmentation)
                best_boxes, best_name = min(results, key=lambda x: len(x[0]))
                print(f"[SEGMENT] Selected {best_name} approach with {len(best_boxes)} boxes (fallback - fewest)")
                boxes = best_boxes
        else:
            # All approaches failed, use standard
            print(f"[SEGMENT] All adaptive approaches failed, using standard")
            boxes = _segment_words_core(img, gray, w, h, kx_ratio=80, debug_out_path=debug_out_path)
    else:
        boxes = _segment_words_core(img, gray, w, h, kx_ratio=80, debug_out_path=debug_out_path)
    
    if debug_out_path:
        dbg = img.copy()
        for (x, y, ww, hh) in boxes:
            cv2.rectangle(dbg, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.imwrite(debug_out_path, dbg)
        print(f"[SEGMENT] Debug image saved to: {debug_out_path}")
    
    return img, boxes

def _segment_words_core(img, gray, w, h, kx_ratio=80, debug_out_path=None):
    """Core segmentation logic with configurable dilation."""

    # 1) Normalize lighting a bit (safe for handwriting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # 2) Binary (ink=white) for contour detection ONLY
    # Use larger block size and higher C value to reduce noise
    # This helps avoid detecting empty spaces and background artifacts
    # blockSize must be odd and > 1
    block_size = max(31, w // 20)
    if block_size % 2 == 0:
        block_size += 1  # Make it odd
    C = 12  # Slightly higher to reduce false positives
    bw = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )
    print(f"[SEGMENT] Adaptive threshold: block_size={block_size}, C={C}")
    
    # Apply morphological operations to remove small noise
    # This helps clean up isolated pixels that might be detected as boxes
    # Be more conservative to avoid removing actual text
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, noise_kernel, iterations=1)  # Less aggressive
    # Also apply closing to fill small gaps within letters
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, noise_kernel, iterations=1)

    # 3) Connect letters within a word (horizontal dilation)
    # Use kx_ratio parameter to control dilation aggressiveness
    kx = max(6, w // kx_ratio)   # Dilation based on ratio (lower ratio = more aggressive)
    ky = max(2, h // 200)  # Small vertical dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    merged = cv2.dilate(bw, kernel, iterations=1)
    print(f"[SEGMENT] Dilation kernel: {kx}x{ky} (ratio={kx_ratio}, image size: {w}x{h})")

    # 4) Find connected components => word blobs
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"[SEGMENT] Found {len(cnts)} initial contours")
    
    boxes = []
    for i, c in enumerate(cnts):
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        img_area = w * h
        
        # Balanced filtering - not too strict, not too lenient
        min_area = img_area * 0.0001  # Back to more lenient to catch valid words
        min_height = max(5, h * 0.05)  # More lenient height threshold
        min_width = max(3, w * 0.01)  # More lenient width threshold
        
        print(f"[CONTOUR {i}] x={x} y={y} w={ww} h={hh} area={area} min_area={min_area} min_height={min_height} min_width={min_width}")
        
        # filter tiny noise
        if area < min_area:
            print(f"  -> REJECTED: area too small ({area} < {min_area})")
            continue
        if hh < min_height:
            print(f"  -> REJECTED: height too small ({hh} < {min_height})")
            continue
        if ww < min_width:
            print(f"  -> REJECTED: width too small ({ww} < {min_width}) - likely artifact")
            continue
        
        # Check pixel density: ensure the bounding box actually contains foreground pixels
        # Extract the region from the binary image
        box_region = merged[y:y+hh, x:x+ww]
        if box_region.size == 0:
            print(f"  -> REJECTED: empty region")
            continue
        
        # Count foreground pixels (white in our inverted binary image)
        foreground_pixels = np.sum(box_region == 255)
        total_pixels = box_region.size
        density = foreground_pixels / total_pixels if total_pixels > 0 else 0
        
        # Require at least 5% foreground density to avoid empty boxes
        min_density = 0.05
        if density < min_density:
            print(f"  -> REJECTED: density too low ({density:.3f} < {min_density}) - likely empty space")
            continue
        
        # Also check contour area vs bounding box area ratio
        # Real text should have a reasonable fill ratio (not too sparse)
        contour_area = cv2.contourArea(c)
        fill_ratio = contour_area / area if area > 0 else 0
        min_fill_ratio = 0.15  # More lenient - 15% fill ratio
        
        if fill_ratio < min_fill_ratio:
            print(f"  -> REJECTED: fill ratio too low ({fill_ratio:.3f} < {min_fill_ratio}) - likely noise")
            continue
        
        # Additional check: box aspect ratio - very narrow boxes are likely noise
        aspect_ratio = ww / hh if hh > 0 else 0
        if aspect_ratio > 10:  # Very wide and short = likely horizontal noise
            print(f"  -> REJECTED: aspect ratio too high ({aspect_ratio:.2f}) - likely noise")
            continue
        if aspect_ratio < 0.1:  # Very tall and narrow = likely vertical noise
            print(f"  -> REJECTED: aspect ratio too low ({aspect_ratio:.2f}) - likely noise")
            continue
        
        print(f"  -> ACCEPTED (density={density:.3f}, fill_ratio={fill_ratio:.3f})")
        boxes.append((x, y, ww, hh))

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])
    
    # Filter out boxes that are likely noise (too far from main text line)
    if len(boxes) > 1:
        # Find the median y-coordinate (main text line)
        y_coords = [b[1] for b in boxes]
        median_y = np.median(y_coords)
        
        # Find the median height
        heights = [b[3] for b in boxes]
        median_h = np.median(heights)
        
        # Allow boxes within 2x median height from the median line
        max_y_deviation = median_h * 2.0
        
        filtered_boxes = []
        for b in boxes:
            y_center = b[1] + b[3] / 2
            deviation = abs(y_center - median_y)
            if deviation <= max_y_deviation:
                filtered_boxes.append(b)
            else:
                print(f"[SEGMENT] Filtered out box at y={b[1]} (deviation={deviation:.1f} > {max_y_deviation:.1f})")
        
        if len(filtered_boxes) < len(boxes):
            print(f"[SEGMENT] Filtered {len(boxes) - len(filtered_boxes)} boxes that were too far from main text line")
            boxes = filtered_boxes
    
    # Smart merging: analyze gaps to distinguish letter gaps (within words) from word gaps (between words)
    # Be VERY conservative - only merge when clearly part of same word
    if len(boxes) > 1:
        gaps = []
        for i in range(len(boxes) - 1):
            prev_x, prev_y, prev_w, prev_h = boxes[i]
            curr_x, curr_y, curr_w, curr_h = boxes[i + 1]
            gap = curr_x - (prev_x + prev_w)
            gaps.append(gap)
        
        if gaps:
            # Use percentiles to better understand gap distribution
            gaps_array = np.array(gaps)
            p25 = np.percentile(gaps_array, 25)  # 25th percentile (small gaps - likely letters)
            p75 = np.percentile(gaps_array, 75)  # 75th percentile (larger gaps - likely words)
            median_gap = np.median(gaps_array)
            
            # Letter gaps are typically in the lower percentiles
            # Word gaps are typically much larger
            # Use a conservative threshold: only merge very small gaps
            max_letter_gap_absolute = min(p25 * 2.0, w * 0.015)  # Very conservative: 1.5% of width max
            max_letter_gap_relative = np.mean([b[2] for b in boxes]) * 0.2  # 20% of average box width
            
            # Use the smaller of the two to be very conservative
            max_letter_gap = min(max_letter_gap_absolute, max_letter_gap_relative)
            
            print(f"[SEGMENT] Gap analysis: p25={p25:.1f}px, median={median_gap:.1f}px, p75={p75:.1f}px")
            print(f"[SEGMENT] Max letter gap threshold: {max_letter_gap:.1f}px")
            
            # Now merge boxes that are clearly part of the same word
            merged_boxes = []
            merged_boxes.append(list(boxes[0]))
            
            for i in range(1, len(boxes)):
                prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
                curr_x, curr_y, curr_w, curr_h = boxes[i]
                
                # Calculate gap between boxes
                gap = curr_x - (prev_x + prev_w)
                
                # Calculate vertical overlap
                prev_bottom = prev_y + prev_h
                curr_bottom = curr_y + curr_h
                vertical_overlap = max(0, min(prev_bottom, curr_bottom) - max(prev_y, curr_y))
                overlap_ratio = vertical_overlap / min(prev_h, curr_h) if min(prev_h, curr_h) > 0 else 0
                
                # Calculate what the merged box would look like
                would_be_width = (curr_x + curr_w) - prev_x
                would_be_height = max(prev_bottom, curr_bottom) - min(prev_y, curr_y)
                avg_height = (prev_h + curr_h) / 2
                
                # Very strict merging criteria:
                # 1. Gap must be VERY small (definitely a letter gap, not word gap)
                # 2. High vertical overlap (same baseline - at least 70%)
                # 3. Merged box shouldn't be too wide (max 4x height for a word)
                # 4. Gap should be less than average box height (letters are close)
                max_reasonable_width_ratio = 4.0  # Words shouldn't be more than 4x height
                gap_vs_height = gap / avg_height if avg_height > 0 else float('inf')
                
                should_merge = (
                    gap < max_letter_gap and  # Gap is very small
                    gap < avg_height * 0.8 and  # Gap is less than 80% of average height
                    overlap_ratio >= 0.7 and  # High vertical alignment (70%+)
                    (would_be_height == 0 or would_be_width / would_be_height < max_reasonable_width_ratio) and
                    gap_vs_height < 1.0  # Gap is smaller than box height
                )
                
                if should_merge:
                    # Merge boxes
                    new_x = prev_x
                    new_y = min(prev_y, curr_y)
                    new_w = (curr_x + curr_w) - prev_x
                    new_h = max(prev_bottom, curr_bottom) - new_y
                    merged_boxes[-1] = [new_x, new_y, new_w, new_h]
                    print(f"[SEGMENT] Merged box {i-1} and {i}: gap={gap:.1f}px ({gap/avg_height*100:.1f}% of height), overlap={overlap_ratio:.2f}, size={new_w}x{new_h}")
                else:
                    merged_boxes.append(list(boxes[i]))
                    print(f"[SEGMENT] Kept separate: gap={gap:.1f}px (>{max_letter_gap:.1f}px or overlap={overlap_ratio:.2f}<0.7)")
            
            boxes = [tuple(b) for b in merged_boxes]
        else:
            boxes = boxes  # No gaps to analyze, keep as is
    else:
        boxes = boxes  # Only one box, nothing to merge
    
    print(f"[SEGMENT] Kept {len(boxes)} word boxes after filtering and merging")
    
    return boxes


def crop_words_to_tempdir(img_bgr, boxes, out_dir: Path):
    """
    Crop each word box to an image file with padding.
    Returns list of file paths in reading order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    h, w = img_bgr.shape[:2]
    
    for i, (x, y, box_w, box_h) in enumerate(boxes):
        # Add small padding around the crop
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + box_w + pad)
        y2 = min(h, y + box_h + pad)
        
        crop = img_bgr[y1:y2, x1:x2]
        
        # Save as PNG to avoid JPEG artifacts on thin strokes
        p = out_dir / f"word_{i:02d}.png"
        cv2.imwrite(str(p), crop)
        print(f"[CROP {i}] Saved {p.name} with shape {crop.shape}")
        paths.append(p)
    
    return paths


def turkish_postprocess(word: str) -> dict:
    raw = _basic_cleanup(word)
    if not raw:
        return {"raw": raw, "best": raw, "alternatives": []}

    cands = _generate_candidates_conservative(raw, max_changes=2)
    if not cands:
        return {"raw": raw, "best": raw, "alternatives": []}

    scored = sorted((( _score_candidate(raw, c), c) for c in cands), reverse=True)
    best = scored[0][1]

    # Top 3 alternatif (unique)
    top = []
    for _, c in scored:
        if c not in top:
            top.append(c)
        if len(top) >= 3:
            break

    # Eğer "best" ile raw arasında fark çok küçükse, hamı koru
    raw_score = _score_candidate(raw, raw)
    best_score = _score_candidate(raw, best)
    if best != raw and (best_score - raw_score) < 0.35:
        best = raw

    # Alternatiflere hamı da koy
    if raw not in top:
        top.append(raw)

    return {"raw": raw, "best": best, "alternatives": top}


# -----------------------------
# HTR prediction via logs
# -----------------------------
def _latest_new_log(before_set: set[str]) -> Path:
    all_logs = sorted(
        ATTN_MODEL_DIR.glob("result/**/log_predictions_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_logs:
        raise RuntimeError(f"No prediction log file found under {ATTN_MODEL_DIR / 'result'}")

    new_logs = [p for p in all_logs if p.as_posix() not in before_set]
    return new_logs[0] if new_logs else all_logs[0]


def attentionhtr_predict_single(image_path: str) -> str:
    if not SAVED_MODEL.exists():
        raise FileNotFoundError(f"Saved model not found: {SAVED_MODEL}")
    if not (ATTN_MODEL_DIR / "test.py").exists():
        raise FileNotFoundError(f"test.py not found under: {ATTN_MODEL_DIR}")

    before = {p.as_posix() for p in ATTN_MODEL_DIR.glob("result/**/log_predictions_*.txt")}

    req_id = uuid.uuid4().hex[:10]
    dataset_name = f"api_{req_id}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        input_dir = tmp / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        src = Path(image_path)
        dst = input_dir / src.name
        shutil.copyfile(src, dst)

        dst_bytes = dst.read_bytes()
        print(f"[TMP_INPUT] name={dst.name} bytes={len(dst_bytes)} md5={hashlib.md5(dst_bytes).hexdigest()}")

        gt_path = tmp / "gt.txt"
        gt_path.write_text(f"{dst.name}\t-\n", encoding="utf-8")

        out_lmdb = tmp / dataset_name

        subprocess.check_call(
            [
                "python3",
                str(ATTN_MODEL_DIR / "create_lmdb_dataset.py"),
                "--inputPath",
                str(input_dir),
                "--gtFile",
                str(gt_path),
                "--outputPath",
                str(out_lmdb),
            ],
            cwd=str(ATTN_MODEL_DIR),
        )

        subprocess.check_call(
            [
                "python3",
                str(ATTN_MODEL_DIR / "test.py"),
                "--eval_data",
                str(out_lmdb),
                "--Transformation",
                "TPS",
                "--FeatureExtraction",
                "ResNet",
                "--SequenceModeling",
                "BiLSTM",
                "--Prediction",
                "Attn",
                "--saved_model",
                str(SAVED_MODEL),
                "--sensitive",
            ],
            cwd=str(ATTN_MODEL_DIR),
        )

    log_file = _latest_new_log(before)
    print(f"[LOG] using: {log_file}")

    lines = log_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    pred = ""
    
    # Parse log format: batch,target,prediction,match,cum_match
    # Find the last valid prediction line
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("batch,") or line.lower().startswith("batch,target"):
            continue
        
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            # Format: batch_num,gt,pred,match,cum_match
            # parts[2] is the prediction
            try:
                int(parts[0])  # Verify it's a batch number
                pred = parts[2].strip()
                # Clean prediction immediately to remove artifacts
                pred = _basic_cleanup(pred)
                print(f"[PREDICT] Found prediction: '{pred}' from line: {line}")
                break
            except (ValueError, IndexError):
                # Not a standard format, try to use parts[2] anyway
                pred = parts[2].strip() if len(parts) > 2 else ""
                if pred:
                    # Clean prediction immediately
                    pred = _basic_cleanup(pred)
                    print(f"[PREDICT] Found prediction (non-standard format): '{pred}' from line: {line}")
                    break

    if not pred:
        print(f"[PREDICT] WARNING: No prediction found in log file")
    
    print(f"[PREDICT] Final result: '{pred}'")
    return pred

def attentionhtr_predict_batch(image_paths: list[Path]) -> dict[str, str]:
    """
    Run AttentionHTR test.py ONCE for multiple images.
    Returns dict: {filename: prediction}
    """
    if not SAVED_MODEL.exists():
        raise FileNotFoundError(f"Saved model not found: {SAVED_MODEL}")
    if not (ATTN_MODEL_DIR / "test.py").exists():
        raise FileNotFoundError(f"test.py not found under: {ATTN_MODEL_DIR}")

    print(f"[BATCH_PREDICT] Processing {len(image_paths)} images")

    before = {p.as_posix() for p in ATTN_MODEL_DIR.glob("result/**/log_predictions_*.txt")}

    req_id = uuid.uuid4().hex[:10]
    dataset_name = f"api_batch_{req_id}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_dir = tmp / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Copy all crops into one folder
        for p in image_paths:
            shutil.copyfile(p, input_dir / p.name)
            print(f"[BATCH] Copied {p.name} to batch input")

        # gt file: filename \t ground_truth
        # Using actual filename instead of "-" so we can match predictions
        gt_path = tmp / "gt.txt"
        gt_lines = [f"{p.name}\t{p.name}\n" for p in image_paths]
        gt_path.write_text("".join(gt_lines), encoding="utf-8")

        out_lmdb = tmp / dataset_name

        subprocess.check_call(
            [
                "python3",
                str(ATTN_MODEL_DIR / "create_lmdb_dataset.py"),
                "--inputPath", str(input_dir),
                "--gtFile", str(gt_path),
                "--outputPath", str(out_lmdb),
            ],
            cwd=str(ATTN_MODEL_DIR),
        )

        subprocess.check_call(
            [
                "python3",
                str(ATTN_MODEL_DIR / "test.py"),
                "--eval_data", str(out_lmdb),
                "--Transformation", "TPS",
                "--FeatureExtraction", "ResNet",
                "--SequenceModeling", "BiLSTM",
                "--Prediction", "Attn",
                "--saved_model", str(SAVED_MODEL),
                "--sensitive",
            ],
            cwd=str(ATTN_MODEL_DIR),
        )

    log_file = _latest_new_log(before)
    print(f"[BATCH_LOG] Using: {log_file}")
    
    log_content = log_file.read_text(encoding="utf-8", errors="ignore")
    print(f"[BATCH_LOG] Full log content:\n{log_content}\n")
    
    lines = log_content.strip().splitlines()

    # Parse predictions - log format is: batch,target,prediction,match,cum_match
    # Each data line: batch_num,gt,pred,match,cum_match
    # We expect exactly len(image_paths) prediction lines for our batch
    preds: dict[str, str] = {}
    prediction_lines = []
    
    # First, collect all valid prediction lines
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip header line
        if line.lower().startswith("batch,") or line.lower().startswith("batch,target"):
            print(f"[BATCH_LOG] Skipping header: {line}")
            continue
        
        # Parse CSV line - format: batch_num,gt,pred,match,cum_match
        parts = [p.strip() for p in line.split(",")]
        
        if len(parts) < 3:
            print(f"[BATCH_LOG] Skipping malformed line (too few parts): {line}")
            continue
        
        # Format: batch_num,gt,pred,match,cum_match
        # We need parts[1] (gt) and parts[2] (pred)
        try:
            batch_num = int(parts[0])  # Verify it's a batch number
            gt_text = parts[1] if len(parts) > 1 else ""
            pred_text = parts[2] if len(parts) > 2 else ""
            
            # Clean prediction immediately to remove artifacts
            pred_text = _basic_cleanup(pred_text)
            
            if gt_text and pred_text:
                prediction_lines.append((batch_num, gt_text, pred_text))
                print(f"[BATCH_LOG] Parsed: batch={batch_num}, gt='{gt_text}', pred='{pred_text}'")
        except (ValueError, IndexError) as e:
            print(f"[BATCH_LOG] Skipping line (not standard format): {line} (error: {e})")
            continue
    
    print(f"[BATCH_LOG] Extracted {len(prediction_lines)} total prediction lines")
    
    # Since log files can be appended to, we need to find the predictions for OUR batch
    # Our GT file uses filenames, so match by filename first
    filename_to_pred = {}
    matched_indices = set()
    
    # First pass: try to match by filename (GT text should be the filename)
    for batch_num, gt_text, pred_text in prediction_lines:
        for i, img_path in enumerate(image_paths):
            if i in matched_indices:
                continue
            if img_path.name == gt_text:
                filename_to_pred[img_path.name] = pred_text
                matched_indices.add(i)
                print(f"[BATCH_RESULT] Matched by filename: {img_path.name} -> '{pred_text}'")
                break
    
    # Second pass: if we didn't match all, try order-based matching with remaining predictions
    if len(filename_to_pred) < len(image_paths):
        print(f"[BATCH_LOG] Only matched {len(filename_to_pred)} by filename, trying order-based matching")
        # Get predictions that weren't matched by filename, sorted by batch number
        unmatched_predictions = [
            (batch_num, gt_text, pred_text) 
            for batch_num, gt_text, pred_text in prediction_lines
            if gt_text not in [p.name for p in image_paths if p.name in filename_to_pred]
        ]
        unmatched_predictions.sort(key=lambda x: x[0])
        
        # Match remaining images to remaining predictions by order
        unmatched_images = [img_path for i, img_path in enumerate(image_paths) if i not in matched_indices]
        for i, img_path in enumerate(unmatched_images):
            if i < len(unmatched_predictions):
                _, _, pred_text = unmatched_predictions[i]
                filename_to_pred[img_path.name] = pred_text
                print(f"[BATCH_RESULT] Matched by order: {img_path.name} -> '{pred_text}'")
            else:
                filename_to_pred[img_path.name] = ""
                print(f"[BATCH_WARNING] No prediction available for {img_path.name}")
    
    # Final check: ensure all images have predictions
    for img_path in image_paths:
        if img_path.name not in filename_to_pred:
            filename_to_pred[img_path.name] = ""
            print(f"[BATCH_WARNING] Missing prediction for {img_path.name}, using empty string")

    print(f"[BATCH_PREDICT] Final: {len(filename_to_pred)} predictions for {len(image_paths)} images")
    for img_path in image_paths:
        print(f"  {img_path.name}: '{filename_to_pred.get(img_path.name, '')}'")
    
    return filename_to_pred


@app.post("/predict-word")
async def predict_word_api(
    image: UploadFile = File(...),
    infer_orientation: str = "auto",
    language: str = Query("tr", description="Language code: 'tr' for Turkish, 'en' for English"),
    spell_correct_enabled: bool = Query(True, description="Enable spell correction"),
):
    ext = os.path.splitext(image.filename)[1].lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    data = await image.read()
    sha = hashlib.md5(data).hexdigest()
    print(f"[UPLOAD] filename={image.filename} saved_as={path} bytes={len(data)} md5={sha}")

    with open(path, "wb") as f:
        f.write(data)

    try:
        raw_text = attentionhtr_predict_single(path)
        
        # Apply post-processing based on language
        if language == "tr":
            pp = turkish_postprocess(raw_text)
            best = pp["best"]
        else:
            # For English, just clean up
            best = _basic_cleanup(raw_text)
            pp = {"raw": raw_text, "best": best, "alternatives": []}
        
        # Apply spell correction if enabled
        if spell_correct_enabled and best:
            corrected = spell_correct(best, language)
            if corrected != best:
                print(f"[SPELL] Corrected '{best}' -> '{corrected}'")
                best = corrected
        
        return {
            "prediction": best,
            "raw": pp["raw"],
            "alternatives": pp["alternatives"],
        }
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"AttentionHTR failed: {e}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
@app.post("/predict-sentence")
async def predict_sentence_api(
    image: UploadFile = File(...),
    language: str = Query("tr", description="Language code: 'tr' for Turkish, 'en' for English"),
    spell_correct_enabled: bool = Query(True, description="Enable spell correction"),
):
    ext = os.path.splitext(image.filename)[1].lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    data = await image.read()
    with open(path, "wb") as f:
        f.write(data)

    print(f"[SENTENCE] Processing: {path}")

    try:
        # 1) word boxes
        debug_boxes_path = os.path.join(UPLOAD_DIR, f"boxes_{filename}.png")
        img_bgr, boxes = segment_words_from_image(path, debug_out_path=debug_boxes_path)

        # fallback: if no boxes, run single
        if not boxes:
            print("[SENTENCE] No word boxes found, falling back to single prediction")
            raw_text = attentionhtr_predict_single(path)
            
            # Apply post-processing based on language
            if language == "tr":
                pp = turkish_postprocess(raw_text)
                best = pp["best"]
            else:
                best = _basic_cleanup(raw_text)
                pp = {"raw": raw_text, "best": best, "alternatives": []}
            
            # Apply spell correction if enabled
            if spell_correct_enabled and best:
                corrected = spell_correct(best, language)
                if corrected != best:
                    print(f"[SPELL] Corrected '{best}' -> '{corrected}'")
                    best = corrected
            
            return {
                "prediction": best, 
                "raw": pp["raw"], 
                "alternatives": pp["alternatives"], 
                "words": [],
                "debug_info": "No word boundaries detected, processed as single image"
            }

        print(f"[SENTENCE] Found {len(boxes)} word boxes")

        # 2) crop words to temp
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            crops = crop_words_to_tempdir(img_bgr, boxes, tmp / "crops")

            print(f"[SENTENCE] Cropped {len(crops)} word images")

            # 3) batch predict
            pred_map = attentionhtr_predict_batch(crops)

            # 4) reconstruct in order + postprocess per word based on language
            words = []
            for i, p in enumerate(crops):
                raw = pred_map.get(p.name, "")
                if not raw:
                    print(f"[SENTENCE] WARNING: No prediction found for {p.name} (index {i})")
                
                # Apply post-processing based on language
                if language == "tr":
                    pp = turkish_postprocess(raw)
                else:
                    # For English, just clean up
                    best = _basic_cleanup(raw)
                    pp = {"raw": raw, "best": best, "alternatives": []}
                
                # Apply spell correction if enabled
                best_word = pp["best"]
                if spell_correct_enabled and best_word:
                    corrected = spell_correct(best_word, language)
                    if corrected != best_word:
                        print(f"[SPELL] Word {i}: '{best_word}' -> '{corrected}'")
                        best_word = corrected
                
                words.append({
                    "raw": pp["raw"],
                    "best": best_word,
                    "alternatives": pp["alternatives"],
                    "file": p.name,
                })
                print(f"[SENTENCE] Word {i} ({p.name}): raw='{pp['raw']}' best='{best_word}'")

        # Join words with spaces, filtering out empty predictions
        sentence = " ".join([w["best"] for w in words if w["best"] and w["best"].strip()])
        raw_sentence = " ".join([w["raw"] for w in words if w["raw"] and w["raw"].strip()])
        
        # Final sentence-level spell correction (optional, can help with word combinations)
        if spell_correct_enabled and sentence:
            # For now, just correct individual words (already done above)
            # Could add sentence-level correction here if needed
            pass
        
        # Validate that we got predictions for all words
        if len(words) != len(crops):
            print(f"[SENTENCE] WARNING: Expected {len(crops)} words but got {len(words)}")
        if len([w for w in words if w["best"]]) < len(crops):
            missing = [p.name for i, p in enumerate(crops) if i >= len(words) or not words[i]["best"]]
            print(f"[SENTENCE] WARNING: Missing predictions for: {missing}")

        print(f"[SENTENCE] Final result: '{sentence}'")

        return {
            "prediction": sentence,
            "raw": raw_sentence,
            "words": words,
            "debug_boxes_image": f"boxes_{filename}.png",
        }

    except subprocess.CalledProcessError as e:
        print(f"[SENTENCE] AttentionHTR subprocess error: {e}")
        return JSONResponse({"error": f"AttentionHTR failed: {e}"}, status_code=500)
    except Exception as e:
        print(f"[SENTENCE] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)