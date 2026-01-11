# Facture Engine OCR 🧾🔍

A robust, hybrid OCR and fuzzy matching engine designed to extract product information and prices from **handwritten Arabic and French invoices**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OCR](https://img.shields.io/badge/OCR-Tesseract%20%2B%20PaddleOCR-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🌟 Features

*   **Hybrid OCR Engine**: Combines **Tesseract** (primary) and **PaddleOCR** (fallback) for maximum accuracy on difficult handwritten text.
*   **Bilingual Support**: Optimized for **Arabic** and **French** text (`ara+fra`).
*   **Advanced Fuzzy Matching**: Uses a multi-strategy approach (Exact, Synonym, Token, Fuzzy, Partial) to match OCR output against a product database.
*   **Noise Filtering**: Smart validation logic rejects "unit-only" matches (e.g., "kg", "L") and requires semantic anchors (meaningful words).
*   **Synonym Mapping**: Handles common OCR errors and handwriting variations via a customizable `synonyms.csv`.
*   **Weighted Confidence Scoring**: Scores matches based on text length, language content (Arabic/French ratio), and matching strategy.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/elbara99/facture-engine-ocr.git
    cd facture-engine-ocr
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install opencv-python pytesseract rapidfuzz pillow paddlepaddle paddleocr
    ```

3.  **Install Tesseract OCR**:
    *   Download and install Tesseract from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
    *   **Important**: During installation, select **Arabic** and **French** language data.
    *   Update the `TESSERACT_CMD` path in `core.py` if necessary:
        ```python
        TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ```

## 🚀 Usage

Run the engine on an invoice image:

```bash
python core.py invoice.jpg
```

### Options

*   `--debug`: Enable detailed debug output to see how lines are parsed and matched.
    ```bash
    python core.py invoice.jpg --debug
    ```
*   `--prices`: Specify a custom prices CSV file (default: `prices.csv`).
    ```bash
    python core.py invoice.jpg --prices my_prices.csv
    ```

## 📂 Project Structure

*   `core.py`: Main logic for OCR, normalization, and matching.
*   `prices.csv`: Database of canonical products and prices.
    *   Format: `item_id, canonical_name, price`
*   `synonyms.csv`: Mappings for common OCR errors to canonical names.
    *   Format: `canonical_name, synonym`
*   `invoice.jpg`: Sample invoice image.

## ⚙️ Configuration

### Synonym Mapping (`synonyms.csv`)
Improve accuracy by mapping common handwriting misinterpretations to the correct product name.
```csv
canonical_name,synonym
فرشاة الطاولة,فرشا الاولة
ملعقة الطاولة,sale الماوله
```

### Price Database (`prices.csv`)
Define your product catalog.
```csv
item_id,canonical_name,price
1,ابريق الشاي 2ل,2000
2,طقم ملاعق,1500
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
