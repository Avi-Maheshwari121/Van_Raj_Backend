# Hindi OCR NER System

A comprehensive Hindi Optical Character Recognition (OCR) and Named Entity Recognition (NER) system designed specifically for extracting structured data from Hindi government documents, particularly Patta papers.

## Features

🔍 **Advanced Hindi Text Recognition**
- Enhanced TrOCR model with Hindi language support
- Multiple pattern matching for improved accuracy
- Robust error handling and fallback mechanisms

📋 **Entity Extraction**
- Extract 14 different types of entities from Hindi documents
- Multi-pattern regex matching for better recognition
- Data cleaning and validation

🏛️ **Government Document Support**
- Optimized for Patta papers and similar documents
- Recognizes administrative divisions (Gram Panchayat, Tehsil, District)
- Extracts property information (Khasra numbers, area, boundaries)
- Personal information extraction (names, relationships)

🔧 **Robust Processing**
- Batch processing support
- JSON export functionality
- Comprehensive error handling
- Detailed logging

## Entity Types Supported

- **GRAM_PANCHAYAT**: Village council information
- **JANPAD_PANCHAYAT**: Block council information  
- **TEHSIL**: Sub-district information
- **DISTRICT**: District information
- **SERIAL_NO**: Document serial numbers
- **DATE**: Document dates
- **HOLDER_NAME**: Property holder name
- **FATHER_NAME**: Father's/husband's name
- **KHASRA_NO**: Land record numbers
- **TOTAL_AREA_SQFT**: Total area in square feet
- **BOUNDARY_EAST/WEST/NORTH/SOUTH**: Property boundaries

## Files Overview

### Core System Files
- `hindi_ocr_ner.py` - Main comprehensive OCR NER system
- `OCR_NER.py` - Enhanced version of original OCR script
- `extractor.py` - Original entity extraction implementation
- `demo.py` - Comprehensive demonstration script

### Configuration Files
- `config.json` - System configuration parameters
- `requirements.txt` - Python package dependencies

### Test Files
- `test2.png` - Primary test image
- `test_patta.png` - Patta document test image
- `test_patta_new.png` - Additional test image

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the trained TrOCR model in the `./trained_trocr_model` directory.

## Usage

### Basic Usage

```python
from hindi_ocr_ner import HindiOCRNER

# Initialize the system
ocr_ner = HindiOCRNER()

# Process a single image
results = ocr_ner.process_image('test2.png')

# Print extracted entities
for entity, value in results.items():
    if entity != '_metadata' and value:
        print(f"{entity}: {value}")
```

### Batch Processing

```python
# Process multiple images
image_paths = ['test2.png', 'test_patta.png']
results = ocr_ner.process_multiple_images(image_paths)

# Export results
ocr_ner.export_results(results, 'batch_output.json')
```

### Running the Demo

```bash
python demo.py
```

This will:
- Test single image processing
- Demonstrate batch processing
- Test individual entity patterns
- Generate comprehensive test reports

### Running Enhanced OCR

```bash
python OCR_NER.py
```

### Running Original Extractor

```bash
python extractor.py
```

## System Architecture

The system consists of three main components:

1. **OCR Engine**: TrOCR model fine-tuned for Hindi text
2. **Entity Extractor**: Multi-pattern regex engine for structured data extraction
3. **Post-Processor**: Data cleaning, validation, and formatting

## Output Format

The system outputs structured JSON data:

```json
{
  "GRAM_PANCHAYAT": "रामपुर",
  "TEHSIL": "बिलासपुर", 
  "DISTRICT": "रायपुर",
  "SERIAL_NO": "125/2025",
  "DATE": "15/03/2025",
  "HOLDER_NAME": "राम कुमार",
  "KHASRA_NO": "152/1",
  "TOTAL_AREA_SQFT": "2400",
  "_metadata": {
    "image_path": "test2.png",
    "processing_successful": true,
    "extracted_text": "..."
  }
}
```

## Configuration

Modify `config.json` to adjust:
- Model parameters
- Entity patterns
- Processing options
- Validation settings

## Error Handling

The system includes comprehensive error handling:
- Graceful fallback to base model if trained model fails
- Individual entity extraction error handling
- Image processing error management
- Detailed error logging and reporting

## Performance Optimization

- Model loading optimization (single load per session)
- Batch processing for multiple images
- Configurable generation parameters
- Memory-efficient image processing

## Logging

Comprehensive logging system tracks:
- Model loading status
- Processing progress
- Entity extraction results
- Error conditions
- Performance metrics

## Future Enhancements

- Support for more document types
- Integration with OCR confidence scores
- Advanced image preprocessing
- Machine learning-based entity validation
- API endpoint for web integration
- Support for additional Indian languages

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `./trained_trocr_model` directory exists
2. **Dependencies missing**: Run `pip install -r requirements.txt`
3. **Image not loading**: Check image path and format support
4. **Poor extraction quality**: Try image preprocessing or adjust generation parameters

### Performance Tips

- Use batch processing for multiple images
- Optimize image quality before processing
- Adjust generation parameters in config.json
- Monitor memory usage for large batches

## License

This system is designed for educational and research purposes. Please ensure compliance with relevant data protection and privacy regulations when processing government documents.