import csv
import re
import sys
import argparse
import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Third-party imports
try:
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
    from rapidfuzz import process, fuzz
    from paddleocr import PaddleOCR
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Please install: pip install opencv-python pytesseract rapidfuzz pillow paddlepaddle paddleocr")
    sys.exit(1)

# Configuration
TESSERACT_CMD = r'F:\telechargement\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Force UTF-8 for Windows terminals
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Suppress PaddleOCR logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Common Arabic unit tokens that should not trigger matches alone
UNIT_TOKENS = {'ل', 'سم', 'كلغ', 'د', 'م', 'كغ', 'غ', 'ملم', 'كم'}

@dataclass
class Product:
    item_id: str
    name: str
    price: float
    normalized_name: str

@dataclass
class InvoiceItem:
    original_text: str
    matched_product: Optional[Product]
    quantity: int
    line_total: float
    confidence: float
    match_strategy: str  # Track which strategy worked

class TextNormalizer:
    def __init__(self):
        self.replacements = [
            (r'[إأآا]', 'ا'),
            (r'ى', 'ي'),
            (r'ة', 'ه'),
            (r'[\u064B-\u065F]', ''),  # Remove diacritics
            (r'[^\w\s]', ' '),  # Remove punctuation
            (r'\s+', ' '),  # Collapse whitespace
        ]

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
        # Collapse repeated characters (common OCR error)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        return text.strip().lower()
    
    def has_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        arabic_pattern = r'[\u0600-\u06FF]'
        return bool(re.search(arabic_pattern, text))

    def has_french(self, text: str) -> bool:
        """Check if text contains French/Latin characters"""
        french_pattern = r'[a-zA-Z\u00C0-\u00FF]'
        return bool(re.search(french_pattern, text))
    
    def count_arabic_chars(self, text: str) -> int:
        """Count Arabic characters in text"""
        arabic_pattern = r'[\u0600-\u06FF]'
        return len(re.findall(arabic_pattern, text))

    def count_french_chars(self, text: str) -> int:
        """Count French/Latin characters in text"""
        french_pattern = r'[a-zA-Z\u00C0-\u00FF]'
        return len(re.findall(french_pattern, text))
    
    def get_content_ratio(self, text: str) -> float:
        """Get ratio of meaningful characters (Arabic + French) to total characters"""
        if not text:
            return 0.0
        arabic_count = self.count_arabic_chars(text)
        french_count = self.count_french_chars(text)
        total_count = len([c for c in text if c.strip()])
        return (arabic_count + french_count) / total_count if total_count > 0 else 0.0

class PriceDatabase:
    def __init__(self, csv_path: str, normalizer: TextNormalizer):
        self.products: List[Product] = []
        self.normalizer = normalizer
        self.synonyms: Dict[str, str] = {}
        self._load_prices(csv_path)
        self._load_synonyms()

    def _load_prices(self, csv_path: str):
        if not os.path.exists(csv_path):
            print(f"Error: Price file not found at {csv_path}")
            sys.exit(1)
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    name = clean_row.get('canonical_name', '')
                    price = float(clean_row.get('price', 0))
                    item_id = clean_row.get('item_id', '')
                    
                    if name:
                        self.products.append(Product(
                            item_id=item_id,
                            name=name,
                            price=price,
                            normalized_name=self.normalizer.normalize(name)
                        ))
                except ValueError:
                    continue

    def _load_synonyms(self):
        """Load synonyms from synonyms.csv if it exists"""
        synonym_path = "synonyms.csv"
        if not os.path.exists(synonym_path):
            return
            
        try:
            with open(synonym_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip empty rows or comments
                    if not row or any(str(v).startswith('#') for v in row.values() if v):
                        continue
                        
                    syn_val = row.get('synonym')
                    can_val = row.get('canonical_name')
                    
                    if syn_val is not None and can_val is not None:
                        synonym = self.normalizer.normalize(syn_val.strip())
                        canonical = self.normalizer.normalize(can_val.strip())
                        if synonym and canonical:
                            self.synonyms[synonym] = canonical
        except Exception as e:
            print(f"Warning: Could not load synonyms: {e}")

    def find_match(self, query: str, threshold: float = 60.0, debug: bool = False) -> Tuple[Optional[Product], float, str]:
        """
        Multi-strategy fuzzy matching for noisy OCR input.
        Returns: (Product, confidence_score, strategy_name)
        """
        normalized_query = self.normalizer.normalize(query)
        
        if debug:
            print(f"  [MATCH] Query: '{query}' → Normalized: '{normalized_query}'")
        
        # Strategy 1: Exact Match
        for product in self.products:
            if product.normalized_name == normalized_query:
                if debug:
                    print(f"  [MATCH] ✓ Exact match: {product.name}")
                return product, 100.0, "exact"

        # Strategy 2: Synonym Match
        if normalized_query in self.synonyms:
            canonical = self.synonyms[normalized_query]
            for product in self.products:
                if product.normalized_name == canonical:
                    if debug:
                        print(f"  [MATCH] ✓ Synonym match: {product.name}")
                    return product, 100.0, "synonym"

        # Strategy 3: Word-level Token Matching
        query_words = set(normalized_query.split())
        best_token_match = None
        best_token_score = 0.0
        
        for product in self.products:
            product_words = set(product.normalized_name.split())
            
            if query_words and product_words:
                intersection = query_words & product_words
                union = query_words | product_words
                jaccard = len(intersection) / len(union) * 100
                
                max_word_fuzzy = 0.0
                for qw in query_words:
                    for pw in product_words:
                        word_score = fuzz.ratio(qw, pw)
                        max_word_fuzzy = max(max_word_fuzzy, word_score)
                
                combined_score = max(jaccard, max_word_fuzzy)
                
                if combined_score > best_token_score:
                    best_token_score = combined_score
                    best_token_match = product
        
        if best_token_match and best_token_score >= threshold:
            if debug:
                print(f"  [MATCH] ✓ Token match ({best_token_score:.0f}%): {best_token_match.name}")
            return best_token_match, best_token_score, "token"

        # Strategy 4: Fuzzy Match
        choices = {prod.normalized_name: prod for prod in self.products}
        result = process.extractOne(
            normalized_query,
            choices.keys(),
            scorer=fuzz.token_set_ratio
        )

        if result:
            match_name, score, _ = result
            if debug:
                print(f"  [MATCH] Fuzzy candidate ({score:.0f}%): {choices[match_name].name}")
            if score >= threshold:
                if debug:
                    print(f"  [MATCH] ✓ Fuzzy match accepted")
                return choices[match_name], score, "fuzzy"

        # Strategy 5: Partial Ratio
        result_partial = process.extractOne(
            normalized_query,
            choices.keys(),
            scorer=fuzz.partial_ratio
        )
        
        if result_partial:
            match_name, score, _ = result_partial
            if score >= max(threshold - 10, 50):
                if debug:
                    print(f"  [MATCH] ✓ Partial match ({score:.0f}%): {choices[match_name].name}")
                return choices[match_name], score, "partial"

        if debug:
            print(f"  [MATCH] ✗ No match found")
        return None, 0.0, "none"

class InvoiceProcessor:
    def __init__(self, price_db: PriceDatabase, normalizer: TextNormalizer, debug: bool = False):
        self.price_db = price_db
        self.normalizer = normalizer
        self.debug = debug
        self.paddle = None
        self.matched_items = set()
        try:
            self.paddle = PaddleOCR(lang='ar')
        except Exception as e:
            print(f"Warning: PaddleOCR initialization failed. Fallback will be disabled. Error: {e}")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def extract_text(self, image_path: str) -> str:
        text = ""
        try:
            processed_img = self.preprocess_image(image_path)
            custom_config = r'--oem 3 --psm 11 -l ara+fra'
            text = pytesseract.image_to_string(processed_img, config=custom_config)
        except Exception as e:
            print(f"Tesseract Error: {e}")

        clean_text = text.strip()
        if len(clean_text) < 20:
            if self.paddle:
                print("Tesseract output insufficient. Falling back to PaddleOCR...")
                try:
                    result = self.paddle.ocr(image_path, cls=True)
                    paddle_lines = []
                    if result and result[0]:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                paddle_lines.append(line[1][0])
                    text = "\n".join(paddle_lines)
                except Exception as e:
                    print(f"PaddleOCR Error: {e}")
        return text
    
    def is_noise(self, text: str, normalized: str) -> Tuple[bool, str]:
        arabic_count = self.normalizer.count_arabic_chars(text)
        french_count = self.normalizer.count_french_chars(text)
        
        if (arabic_count + french_count) < 3 and len(normalized) < 3:
            return True, "too short (< 3 chars)"
        
        if not self.normalizer.has_arabic(text) and not self.normalizer.has_french(text):
            return True, "no Arabic/French content"
            
        if normalized in UNIT_TOKENS:
            return True, f"unit token only ('{normalized}')"
        return False, ""
    
    def calculate_weighted_confidence(self, base_score: float, strategy: str, text: str, normalized: str) -> float:
        strategy_weights = {
            "exact": 1.0, "synonym": 1.0, "token": 0.95, "fuzzy": 0.90, "partial": 0.85,
            "token_word": 0.88, "fuzzy_word": 0.85, "partial_word": 0.80
        }
        weight = strategy_weights.get(strategy, 0.85)
        text_length = len(normalized)
        length_factor = min(1.0, text_length / 5.0)
        
        content_ratio = self.normalizer.get_content_ratio(text)
        content_factor = 0.8 + (content_ratio * 0.2)
        
        weighted_score = base_score * weight * length_factor * content_factor
        if self.debug:
            print(f"  [SCORE] Base: {base_score:.0f}% | Strategy: {weight:.2f} | Length: {length_factor:.2f} | Content: {content_factor:.2f} → Final: {weighted_score:.0f}%")
        return weighted_score
    
    def has_semantic_anchor(self, text: str) -> bool:
        normalized = self.normalizer.normalize(text)
        tokens = normalized.split()
        meaningful_count = 0
        for token in tokens:
            is_valid_lang = self.normalizer.has_arabic(token) or self.normalizer.has_french(token)
            if (is_valid_lang and 
                token not in UNIT_TOKENS and 
                len(token) >= 3):
                meaningful_count += 1
        return meaningful_count >= 2

    def parse_line(self, line: str) -> Optional[InvoiceItem]:
        clean_line = line.strip()
        if not clean_line or len(clean_line) < 2:
            return None

        if self.debug:
            print(f"\n[PARSE] Line: '{clean_line}'")
        
        normalized = self.normalizer.normalize(clean_line)
        is_noise_result, noise_reason = self.is_noise(clean_line, normalized)
        if is_noise_result:
            if self.debug:
                print(f"  [SKIP] {noise_reason}")
            return None

        numbers = re.findall(r'\d+', clean_line)
        
        # Strategy 1: Full line
        product, score, strategy = self.price_db.find_match(clean_line, debug=self.debug)
        if product:
            is_valid = True
            if strategy != "synonym" and not self.has_semantic_anchor(clean_line):
                if self.debug: print(f"  [SKIP] unit-only match (no semantic anchor)")
                is_valid = False
            
            if is_valid:
                weighted_score = self.calculate_weighted_confidence(score, strategy, clean_line, normalized)
                if weighted_score < 60 and len(normalized) < 4:
                    if self.debug: print(f"  [REJECT] Low confidence ({weighted_score:.0f}%) + short text")
                    is_valid = False
                
                if is_valid and product.item_id in self.matched_items:
                    if self.debug: print(f"  [REJECT] Duplicate item (already matched)")
                    is_valid = False
                
                if is_valid:
                    quantity = 1
                    if numbers:
                        valid_qs = [int(n) for n in numbers if 0 < int(n) < 1000]
                        if valid_qs: quantity = valid_qs[0]
                    
                    self.matched_items.add(product.item_id)
                    return InvoiceItem(clean_line, product, quantity, product.price * quantity, weighted_score, strategy)
        
        # Strategy 2: Without numbers
        if numbers:
            text_no_nums = clean_line
            for num in numbers: text_no_nums = text_no_nums.replace(num, ' ')
            text_no_nums = ' '.join(text_no_nums.split())
            if self.debug: print(f"[PARSE] Trying without numbers: '{text_no_nums}'")
            
            product, score, strategy = self.price_db.find_match(text_no_nums, debug=self.debug)
            if product:
                is_valid = True
                if strategy != "synonym" and not self.has_semantic_anchor(text_no_nums):
                    if self.debug: print(f"  [SKIP] unit-only match (no semantic anchor)")
                    is_valid = False
                
                if is_valid:
                    weighted_score = self.calculate_weighted_confidence(score, strategy, text_no_nums, self.normalizer.normalize(text_no_nums))
                    if weighted_score < 60 and len(self.normalizer.normalize(text_no_nums)) < 4:
                        if self.debug: print(f"  [REJECT] Low confidence ({weighted_score:.0f}%) + short text")
                        is_valid = False
                    
                    if is_valid and product.item_id in self.matched_items:
                        if self.debug: print(f"  [REJECT] Duplicate item (already matched)")
                        is_valid = False
                    
                    if is_valid:
                        quantity = 1
                        valid_qs = [int(n) for n in numbers if 0 < int(n) < 1000]
                        if valid_qs: quantity = valid_qs[0]
                        
                        self.matched_items.add(product.item_id)
                        return InvoiceItem(clean_line, product, quantity, product.price * quantity, weighted_score, strategy)
        
        # Strategy 3: Individual words
        words = clean_line.split()
        for word in words:
            if len(word) >= 3:
                if self.debug: print(f"[PARSE] Trying word: '{word}'")
                product, score, strategy = self.price_db.find_match(word, threshold=70, debug=self.debug)
                if product:
                    is_valid = True
                    if strategy != "synonym" and not self.has_semantic_anchor(word):
                        if self.debug: print(f"  [SKIP] unit-only match (no semantic anchor)")
                        is_valid = False
                    
                    if is_valid:
                        weighted_score = self.calculate_weighted_confidence(score, f"{strategy}_word", word, self.normalizer.normalize(word))
                        if weighted_score < 60 and len(self.normalizer.normalize(word)) < 4:
                            if self.debug: print(f"  [REJECT] Low confidence ({weighted_score:.0f}%) + short text")
                            is_valid = False
                        
                        if is_valid and product.item_id in self.matched_items:
                            if self.debug: print(f"  [REJECT] Duplicate item (already matched)")
                            is_valid = False
                        
                        if is_valid:
                            quantity = 1
                            if numbers:
                                valid_qs = [int(n) for n in numbers if 0 < int(n) < 1000]
                                if valid_qs: quantity = valid_qs[0]
                            
                            self.matched_items.add(product.item_id)
                            return InvoiceItem(clean_line, product, quantity, product.price * quantity, weighted_score, f"{strategy}_word")
        
        return None

    def safe_print(self, text: str):
        """Print text safely handling encoding issues"""
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'ignore').decode('ascii'))

    def process(self, image_path: str):
        self.matched_items = set()
        self.safe_print(f"Processing: {image_path}")
        self.safe_print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
        self.safe_print("=" * 75)
        
        try:
            raw_text = self.extract_text(image_path)
            # Save raw text for debugging
            with open("extracted_text.txt", "w", encoding="utf-8") as f:
                f.write(raw_text)
        except Exception as e:
            self.safe_print(f"Processing failed: {e}")
            return

        self.safe_print("\nEXTRACTED TEXT (First 500 chars):")
        self.safe_print("-" * 75)
        self.safe_print(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
        self.safe_print("-" * 75)

        total_amount = 0.0
        found_items = []
        lines = raw_text.split('\n')
        self.safe_print(f"\nMATCHING {len(lines)} LINES:")
        self.safe_print("=" * 75)
        
        for line in lines:
            item = self.parse_line(line)
            if item:
                found_items.append(item)
                total_amount += item.line_total

        self.safe_print("\nMATCHED ITEMS:")
        self.safe_print("=" * 75)
        self.safe_print(f"{'Item':<30} | {'Qty':<5} | {'Price':<10} | {'Total':<10} | {'Conf':<5} | {'Strategy'}")
        self.safe_print("-" * 95)
        for item in found_items:
            name = item.matched_product.name[:28]
            self.safe_print(f"{name:<30} | {item.quantity:<5} | {item.matched_product.price:<10.2f} | {item.line_total:<10.2f} | {item.confidence:.0f}% | {item.match_strategy}")
        
        self.safe_print("-" * 95)
        self.safe_print(f"GRAND TOTAL: {total_amount:.2f}")
        self.safe_print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Fuzzy Invoice Processor (Handwriting-Optimized)")
    parser.add_argument("image_path", help="Path to the invoice image")
    parser.add_argument("--prices", default="prices.csv", help="Path to prices.csv")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    normalizer = TextNormalizer()
    price_db = PriceDatabase(args.prices, normalizer)
    processor = InvoiceProcessor(price_db, normalizer, debug=args.debug)
    processor.process(args.image_path)

if __name__ == "__main__":
    main()
