# Receipt Carbon Calculator 🌱

A Streamlit web app that calculates the carbon footprint of your grocery receipts using OCR text extraction and fuzzy string matching.

## What it does

Upload a receipt photo or paste receipt text, and the app will:
- Extract text using OCR (if image uploaded)
- Match items to an emissions database using fuzzy string matching
- Calculate total CO2 emissions for your shopping
- Suggest lower-carbon alternatives where available

## Features

- **Multiple input methods**: Text input, image upload, or file upload (PDF/TXT/images)
- **OCR support**: Automatic text extraction from receipt photos
- **Smart matching**: Handles typos and variations in product names
- **Emissions database**: 80+ common grocery items with emission factors
- **Interactive editing**: Adjust quantities and emission factors before calculation
- **Alternative suggestions**: Get recommendations for lower-carbon swaps
- **Export results**: Download results as CSV

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for image text extraction)

### Installation

1. Clone this repository
```bash
git clone <your-repo-url>
cd receipt-carbon-calculator
```

2. Create virtual environment
```bash
python -m venv venv
```

3. Activate virtual environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Install Tesseract OCR (for image processing)

**Windows:**
```bash
winget install --id UB-Mannheim.TesseractOCR
```

**Mac:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

### Running the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to use

1. **Input your receipt**: Choose one of three options:
   - Paste text directly from your receipt
   - Upload a photo of your receipt (JPG/PNG)
   - Upload a file (PDF, TXT, or image)

2. **Review matched items**: The app will find items in the database and show match confidence scores

3. **Edit quantities**: Adjust quantities and emission factors if needed

4. **Calculate emissions**: Click "Calculate Emissions" to see your total carbon footprint

5. **View suggestions**: See alternative products with lower emissions

6. **Export results**: Download your results as a CSV file

## Database

The emissions database (`data/emissions_food.csv`) contains:
- 80+ common grocery items
- Emission factors (kg CO2e per unit)
- Multiple synonyms per item for better matching
- Alternative product suggestions
- Sources from research papers (primarily Poore & Nemecek 2018)

You can edit this file to:
- Add new products
- Update emission factors
- Add more synonyms for better matching
- Include new alternative suggestions

## Technical details

- **Framework**: Streamlit for web interface
- **OCR**: Tesseract via pytesseract
- **Fuzzy matching**: rapidfuzz library with token set ratio
- **PDF processing**: PyMuPDF for text extraction
- **Data handling**: pandas for CSV operations

## Project structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   ├── emissions_food.csv   # Emissions database
│   └── receipts.csv         # Saved receipt logs (created automatically)
└── README.md                # This file
```

## Contributing

To add more items to the database:
1. Open `data/emissions_food.csv`
2. Add new rows with: name, basis (kg/L/ea), emission factor, synonyms, category, alternatives
3. Use pipe `|` to separate synonyms: `"milk|dairy milk|whole milk"`
4. Restart the app to load new data

## Limitations

- OCR accuracy depends on image quality
- Database limited to included items (expandable)
- Emission factors are estimates from research literature
- Quantities may need manual adjustment after OCR

## Troubleshooting

**OCR not working**: 
- Install Tesseract OCR
- Check image quality and lighting
- Try manual text input instead

**Items not matching**:
- Check spelling variations
- Add synonyms to the CSV database
- Use simpler product names

**Performance issues**:
- Large images may take time to process
- Consider resizing images before upload

## Data sources

Emission factors primarily sourced from:
- Poore, J., & Nemecek, T. (2018). Reducing food's environmental impacts through producers and consumers. Science, 360(6392), 987-992.
- Our World in Data food emissions database

---

Built for sustainability awareness and carbon footprint tracking. Every grocery trip is an opportunity to make better choices for the planet! 🌍
