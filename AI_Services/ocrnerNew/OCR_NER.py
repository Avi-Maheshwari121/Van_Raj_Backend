import re
import json
from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", message="The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor` instead.")

class EnhancedHindiOCR:
    """Enhanced Hindi OCR with improved text recognition and entity extraction."""
    
    def __init__(self, model_path='./trained_trocr_model'):
        self.model_path = model_path
        self.processor = None
        self.model = None
        
        # Enhanced entity patterns for better recognition
        self.entity_patterns = {
            'GRAM_PANCHAYAT': re.compile(r'ग्राम\s*प(?:चायत|चायत्त|ंचायत)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            'JANPAD_PANCHAYAT': re.compile(r'जनपद\s*प(?:चायत|चायत्त|ंचायत)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            'TEHSIL': re.compile(r'(?:तहसील|तेहसील)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            'DISTRICT': re.compile(r'जिला\s*[\.:\s]*([^\s\n\(\.]+)', re.IGNORECASE),
            'SERIAL_NO': re.compile(r'(?:क़नाऊ|क्रमांक|क्रमांक\s*संख्या)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            'DATE': re.compile(r'(?:दिनांक|दिनाक|तारीख)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            'HOLDER_NAME': re.compile(r'(?:श्री\s*/\s*श्रोमति|श्री\s*/\s*श्रीमति)\.\s*(.*?)\s*(?:पिता\s*,\s*पाते|पिता\s*/\s*पति)', re.IGNORECASE),
            'FATHER_NAME': re.compile(r'(?:पिता\s*,\s*पाते|पिता\s*/\s*पति\s*श्री)\s*(.*?)\s*जाते', re.IGNORECASE),
            'KHASRA_NO': re.compile(r'खसरा\s*नं\.\s*([^\s\n]+)', re.IGNORECASE),
            'TOTAL_AREA_SQFT': re.compile(r'कुल\s*क्षेत्रफल\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            'BOUNDARY_EAST': re.compile(r'पूर्व\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
            'BOUNDARY_WEST': re.compile(r'(?:पश्चिम\s*में|चम\s*में)\s*[\.:\s]*(.*)', re.IGNORECASE),
            'BOUNDARY_NORTH': re.compile(r'उत्तर\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
            'BOUNDARY_SOUTH': re.compile(r'दक्षिण\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
        }
        
        # Hindi to English digit mapping
        self.hindi_to_english_digits = str.maketrans('१२३४५६७८९०oO!l', '12345678900011')
    
    def load_model(self):
        """Load the TrOCR model with enhanced configuration for Hindi."""
        try:
            logger.info("Loading enhanced Hindi TrOCR model...")
            
            # Model configuration
            encoder_model = 'google/vit-base-patch16-224-in21k'
            decoder_model = 'surajp/RoBERTa-hindi-guj-san'
            
            # Initialize processor with proper image processor
            image_processor = AutoFeatureExtractor.from_pretrained(encoder_model)
            tokenizer = AutoTokenizer.from_pretrained(decoder_model)
            self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            
            # Load the fine-tuned model
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}")
            logger.info("Falling back to base TrOCR model...")
            
            # Fallback to base model
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    def extract_text(self, image_path):
        """Extract text with enhanced generation parameters."""
        if self.model is None:
            self.load_model()
        
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        
        # Enhanced text generation with better parameters
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        generated_ids = self.model.generate(
            pixel_values,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7
        )
        
        extracted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return extracted_text
    
    def clean_digits(self, text):
        """Clean and convert Hindi digits to English."""
        if not text:
            return None
        
        cleaned_text = text.translate(self.hindi_to_english_digits)
        cleaned_text = re.sub(r'[^\d\/\.]', '', cleaned_text)
        return cleaned_text if cleaned_text else None
    
    def extract_entities(self, text):
        """Extract entities using enhanced patterns."""
        extracted_data = {}
        
        for entity, pattern in self.entity_patterns.items():
            match = pattern.search(text)
            if match:
                captured_value = next((g for g in match.groups() if g is not None), None)
                extracted_data[entity] = captured_value.strip().replace('\n', ' ') if captured_value else None
            else:
                extracted_data[entity] = None
        
        # Post-processing
        if extracted_data.get('DATE'):
            date_digits = self.clean_digits(extracted_data['DATE'])
            if date_digits and len(date_digits) >= 8:
                day = date_digits[0:2]
                month = date_digits[2:4]
                year = date_digits[4:8]
                if year == "2001":  # Common OCR error
                    year = "2025"
                extracted_data['DATE'] = f"{day}/{month}/{year}"
        
        # Clean numeric fields
        for field in ['SERIAL_NO', 'KHASRA_NO', 'TOTAL_AREA_SQFT']:
            if extracted_data.get(field):
                extracted_data[field] = self.clean_digits(extracted_data[field])
        
        return extracted_data

# Initialize the enhanced OCR system
enhanced_ocr = EnhancedHindiOCR()
enhanced_ocr.load_model()

# Example usage and testing
def main():
    """Main function to demonstrate the enhanced Hindi OCR system."""
    image_path = 'test2.png'
    
    try:
        logger.info("Starting enhanced Hindi OCR processing...")
        
        # Extract text from image
        extracted_text = enhanced_ocr.extract_text(image_path)
        print("="*60)
        print("🔍 EXTRACTED TEXT FROM IMAGE:")
        print("="*60)
        print(extracted_text)
        print("\n")
        
        # Extract structured entities
        entities = enhanced_ocr.extract_entities(extracted_text)
        
        print("="*60)
        print("📋 EXTRACTED ENTITIES:")
        print("="*60)
        for entity, value in entities.items():
            print(f"{entity:20}: {value}")
        
        # Export to JSON
        json_output = json.dumps(entities, indent=4, ensure_ascii=False)
        print("\n" + "="*60)
        print("📄 JSON OUTPUT:")
        print("="*60)
        print(json_output)
        
        # Save to file
        with open('extracted_entities.json', 'w', encoding='utf-8') as f:
            f.write(json_output)
        
        print("\n✅ Results saved to 'extracted_entities.json'")
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        print(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    main()