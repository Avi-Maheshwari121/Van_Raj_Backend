import re
import json
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings

# --- Global Model and Processor ---
# It's best to load the model only once when the script starts.
PROCESSOR = None
MODEL = None

def load_model():
    """Loads the TrOCR model and processor into global variables."""
    global PROCESSOR, MODEL
    
    if MODEL is None:
        print("Step 1: Loading the TrOCR model (this may take a moment)...")
        # Suppress the specific FutureWarning about 'feature_extractor'
        warnings.filterwarnings(
            "ignore", 
            message="The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor` instead."
        )

        # Define model paths
        encoder_model = 'google/vit-base-patch16-224-in21k'
        decoder_model = 'surajp/RoBERTa-hindi-guj-san'
        # IMPORTANT: Make sure this path points to your fine-tuned model directory
        trained_model_path = './trained_trocr_model' 

        # Initialize and load the processor and model
        PROCESSOR = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        MODEL = VisionEncoderDecoderModel.from_pretrained(trained_model_path)
        print("Model loaded successfully.")

def perform_ocr_with_trocr(image_path: str) -> str:
    """
    Performs OCR on an image file using the loaded TrOCR model.
    """
    print("Step 2: Performing OCR with TrOCR model...")
    image = Image.open(image_path).convert('RGB')

    # Process the image and generate text
    pixel_values = PROCESSOR(image, return_tensors='pt').pixel_values
    generated_ids = MODEL.generate(pixel_values)
    generated_text = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("OCR finished.")
    print("--- Raw Text from Model ---")
    print(generated_text)
    print("---------------------------\n")
    return generated_text

# --- Your Parsing and Cleaning Functions (Copied from your second script) ---

def clean_and_format_date(text: str) -> str:
    """Cleans and formats a date string from OCR into DD/MM/YYYY."""
    if not text:
        return None
    digits = re.sub(r'\D', '', text)
    if len(digits) >= 8:
        day = digits[0:2]
        month = digits[2:4]
        year = digits[4:8]
        # This is a hardcoded correction, be mindful of this for other documents
        if year == "2001":
            year = "2025"
        return f"{day}/{month}/{year}"
    return text

def clean_digits(text: str) -> str:
    """A general-purpose cleaner for numeric fields."""
    if not text:
        return None
    hindi_to_english_digits = str.maketrans('१२३४५६७८९०oO!l', '12345678900011')
    cleaned_text = text.translate(hindi_to_english_digits)
    cleaned_text = re.sub(r'[^\d\/]', '', cleaned_text)
    return cleaned_text

def parse_entities(text: str) -> dict:
    """
    Parses entities from the OCR text using regular expressions.
    """
    print("Step 3: Parsing entities from the text...")
    
    # NOTE: These patterns were designed for easyocr's output.
    # You will likely need to adjust them based on the TrOCR model's output format.
    entity_patterns = {
        'GRAM_PANCHAYAT':   re.compile(r'ग्राम प(?:चायत|चायत्त)\s*[\.:\s]*([^\s\n]+)'),
        'JANPAD_PANCHAYAT': re.compile(r'जनपद प(?:चायत|चायत्त)\s*[\.:\s]*([^\s\n]+)'),
        'TEHSIL':           re.compile(r'तहसील\s*[\.:\s]*([^\s\n]+)'),
        'DISTRICT':         re.compile(r'जिला\s*[\.:\s]*([^\s\n\(\.]+)'),
        'SERIAL_NO':        re.compile(r'(?:क़नाऊ|क्रमांक)\s*.*?([\d.\/!]+\d{4})'),
        'DATE':             re.compile(r'दिनांक(.*)|दिनाक(.*)'),
        'HOLDER_NAME':      re.compile(r'(?:श्री / श्रोमति|श्री / श्रीमति)\.\s*(.*?)\s*(?:पिता , पाते|पिता / पति)'),
        'FATHER_NAME':      re.compile(r'(?:पिता , पाते|पिता / पति श्री)\s*(.*?)\s*जाते'),
        'KHASRA_NO':        re.compile(r'खसरा नं\.\s*([^\s\n]+)'),
        'TOTAL_AREA_SQFT':  re.compile(r'कुल क्षेत्रफल\s*.*?([^\s]+)'),
        'BOUNDARY_EAST':    re.compile(r'पूव मे : (.*)'),
        'BOUNDARY_WEST':    re.compile(r'चम में\s*(.*)'),
        'BOUNDARY_NORTH':   re.compile(r'उत्तर में\s*(.*)'),
        'BOUNDARY_SOUTH':   re.compile(r'दक्षेण में\s*(.*)'),
    }

    extracted_data = {}
    for entity, pattern in entity_patterns.items():
        match = pattern.search(text)
        if match:
            captured_value = next((g for g in match.groups() if g is not None), None)
            extracted_data[entity] = captured_value.strip().replace('\n', ' ') if captured_value else None
        else:
            extracted_data[entity] = None
            
    # --- Two-Stage Parsing & Final Cleaning ---
    print("Applying post-processing and corrections...")

    if extracted_data.get('SERIAL_NO'):
        # Hardcoded correction
        extracted_data['SERIAL_NO'] = '125/2025'

    if extracted_data.get('DATE'):
        value_match = re.search(r'([\d\.\/!]+)', extracted_data['DATE'])
        if value_match:
            extracted_data['DATE'] = clean_and_format_date(value_match.group(1))
    
    # Hardcoded corrections for known OCR failures
    khasra_ocr = clean_digits(extracted_data.get('KHASRA_NO', ''))
    if khasra_ocr == '2':
        extracted_data['KHASRA_NO'] = '152/1'
    
    area_ocr = clean_digits(extracted_data.get('TOTAL_AREA_SQFT', ''))
    if area_ocr == '40':
        extracted_data['TOTAL_AREA_SQFT'] = '2400'

    print("Entity parsing finished.")
    return extracted_data

def main():
    """Main function to run the full OCR-to-JSON pipeline."""
    image_path = 'test2.png'
    
    # 1. Load the AI Model (only happens once)
    load_model()
    
    # 2. Get raw text from the image using the model
    raw_text = perform_ocr_with_trocr(image_path)
    
    # 3. Parse the raw text into structured data
    structured_data = parse_entities(raw_text)
    
    # 4. Print the final result
    print("\n--- ✅ FINAL JSON OUTPUT ---")
    print(json.dumps(structured_data, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()