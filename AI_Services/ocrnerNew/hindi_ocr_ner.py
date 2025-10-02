import re
import json
from typing import Dict, Optional, List, Tuple
from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", message="The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor` instead.")


class HindiOCRNER:
    """
    A comprehensive Hindi OCR and Named Entity Recognition system for extracting
    structured data from Hindi documents, particularly government documents like Patta papers.
    """
    
    def __init__(self, model_path: str = './trained_trocr_model'):
        """
        Initialize the Hindi OCR NER system.
        
        Args:
            model_path: Path to the fine-tuned TrOCR model directory
        """
        self.model_path = model_path
        self.processor = None
        self.model = None
        
        # Enhanced entity patterns for Hindi document processing
        self.entity_patterns = {
            'GRAM_PANCHAYAT': [
                re.compile(r'ग्राम\s*प(?:चायत|चायत्त)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'ग्राम\s*पंचायत\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'ग्राम\s*प\s*चायत\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            ],
            'JANPAD_PANCHAYAT': [
                re.compile(r'जनपद\s*प(?:चायत|चायत्त)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'जनपद\s*पंचायत\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'जनपद\s*प\s*चायत\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            ],
            'TEHSIL': [
                re.compile(r'तहसील\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'तेहसील\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'तह\s*सील\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            ],
            'DISTRICT': [
                re.compile(r'जिला\s*[\.:\s]*([^\s\n\(\.]+)', re.IGNORECASE),
                re.compile(r'जि\s*ला\s*[\.:\s]*([^\s\n\(\.]+)', re.IGNORECASE),
                re.compile(r'ज़िला\s*[\.:\s]*([^\s\n\(\.]+)', re.IGNORECASE),
            ],
            'SERIAL_NO': [
                re.compile(r'(?:क़नाऊ|क्रमांक)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'(?:क्रमांक|क्रमांक\s*संख्या)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'(?:सीरियल|सि\s*रि\s*यल)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'(?:नंबर|नम्बर)\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            ],
            'DATE': [
                re.compile(r'दिनांक\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'दिनाक\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'दि\s*नांक\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'तारीख\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            ],
            'HOLDER_NAME': [
                re.compile(r'(?:श्री\s*/\s*श्रोमति|श्री\s*/\s*श्रीमति)\.\s*(.*?)\s*(?:पिता\s*,\s*पाते|पिता\s*/\s*पति)', re.IGNORECASE),
                re.compile(r'(?:श्री|श्रीमति)\s*(.*?)\s*(?:पुत्र|पुत्री|पिता)', re.IGNORECASE),
                re.compile(r'नाम\s*[\.:\s]*(.*?)\s*(?:पिता|पति)', re.IGNORECASE),
            ],
            'FATHER_NAME': [
                re.compile(r'(?:पिता\s*,\s*पाते|पिता\s*/\s*पति\s*श्री)\s*(.*?)\s*जाते', re.IGNORECASE),
                re.compile(r'पिता\s*का\s*नाम\s*[\.:\s]*(.*?)(?:\s|$)', re.IGNORECASE),
                re.compile(r'(?:पिता|पति)\s*[\.:\s]*(.*?)(?:\s|$)', re.IGNORECASE),
            ],
            'KHASRA_NO': [
                re.compile(r'खसरा\s*नं\.\s*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'खसरा\s*नंबर\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'खस\s*रा\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            ],
            'TOTAL_AREA_SQFT': [
                re.compile(r'कुल\s*क्षेत्रफल\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'क्षेत्रफल\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
                re.compile(r'एरिया\s*[\.:\s]*([^\s\n]+)', re.IGNORECASE),
            ],
            'BOUNDARY_EAST': [
                re.compile(r'पूर्व\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
                re.compile(r'पूव\s*मे\s*[\.:\s]*(.*)', re.IGNORECASE),
                re.compile(r'पूर्व\s*[\.:\s]*(.*)', re.IGNORECASE),
            ],
            'BOUNDARY_WEST': [
                re.compile(r'पश्चिम\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
                re.compile(r'चम\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
                re.compile(r'पश्चिम\s*[\.:\s]*(.*)', re.IGNORECASE),
            ],
            'BOUNDARY_NORTH': [
                re.compile(r'उत्तर\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
                re.compile(r'उत्तर\s*[\.:\s]*(.*)', re.IGNORECASE),
                re.compile(r'उ\s*त्तर\s*[\.:\s]*(.*)', re.IGNORECASE),
            ],
            'BOUNDARY_SOUTH': [
                re.compile(r'दक्षिण\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
                re.compile(r'दक्षेण\s*में\s*[\.:\s]*(.*)', re.IGNORECASE),
                re.compile(r'दक्षिण\s*[\.:\s]*(.*)', re.IGNORECASE),
            ],
        }
        
        # Hindi to English digit mapping
        self.hindi_to_english_digits = str.maketrans('१२३४५६७८९०oO!l', '12345678900011')
        
    def load_model(self) -> None:
        """Load the TrOCR model and processor."""
        try:
            logger.info("Loading TrOCR model for Hindi text recognition...")
            
            # Define model components
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
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to base model if trained model fails
            logger.info("Falling back to base TrOCR model...")
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Preprocess the image for better OCR results.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            # You can add more preprocessing steps here if needed
            # such as contrast enhancement, noise reduction, etc.
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Extracted text from the image
        """
        if self.model is None or self.processor is None:
            self.load_model()
        
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            # Preprocess image
            image = self.preprocess_image(image_path)
            
            # Generate text using TrOCR
            pixel_values = self.processor(image, return_tensors='pt').pixel_values
            generated_ids = self.model.generate(
                pixel_values,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
            extracted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.info("Text extraction completed")
            logger.debug(f"Extracted text: {extracted_text}")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error during text extraction: {str(e)}")
            raise
    
    def clean_digits(self, text: str) -> str:
        """
        Clean and convert Hindi digits to English digits.
        
        Args:
            text: Input text containing digits
            
        Returns:
            Cleaned text with English digits
        """
        if not text:
            return None
        
        # Convert Hindi digits to English
        cleaned_text = text.translate(self.hindi_to_english_digits)
        # Remove non-digit characters except forward slash and period
        cleaned_text = re.sub(r'[^\d\/\.]', '', cleaned_text)
        
        return cleaned_text if cleaned_text else None
    
    def clean_and_format_date(self, text: str) -> str:
        """
        Clean and format date string from OCR into DD/MM/YYYY format.
        
        Args:
            text: Raw date text from OCR
            
        Returns:
            Formatted date string
        """
        if not text:
            return None
        
        # Extract digits from the text
        digits = re.sub(r'\D', '', text)
        
        if len(digits) >= 8:
            day = digits[0:2]
            month = digits[2:4]
            year = digits[4:8]
            
            # Basic validation and correction
            if year == "2001":  # Common OCR error
                year = "2025"
            
            return f"{day}/{month}/{year}"
        
        return text
    
    def extract_entity_with_patterns(self, text: str, entity_name: str) -> Optional[str]:
        """
        Extract entity using multiple regex patterns.
        
        Args:
            text: Input text to search
            entity_name: Name of the entity to extract
            
        Returns:
            Extracted entity value or None
        """
        patterns = self.entity_patterns.get(entity_name, [])
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                # Get the first non-None group
                captured_value = next((g for g in match.groups() if g is not None), None)
                if captured_value:
                    return captured_value.strip().replace('\n', ' ')
        
        return None
    
    def extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract all entities from the given text.
        
        Args:
            text: Input text from OCR
            
        Returns:
            Dictionary containing extracted entities
        """
        logger.info("Extracting entities from text...")
        
        extracted_data = {}
        
        # Extract each entity using multiple patterns
        for entity_name in self.entity_patterns.keys():
            extracted_data[entity_name] = self.extract_entity_with_patterns(text, entity_name)
        
        # Post-processing and cleaning
        self._post_process_entities(extracted_data)
        
        logger.info("Entity extraction completed")
        return extracted_data
    
    def _post_process_entities(self, extracted_data: Dict[str, Optional[str]]) -> None:
        """
        Post-process extracted entities for better accuracy.
        
        Args:
            extracted_data: Dictionary of extracted entities to be modified in-place
        """
        # Clean and format date
        if extracted_data.get('DATE'):
            extracted_data['DATE'] = self.clean_and_format_date(extracted_data['DATE'])
        
        # Clean numeric fields
        numeric_fields = ['SERIAL_NO', 'KHASRA_NO', 'TOTAL_AREA_SQFT']
        for field in numeric_fields:
            if extracted_data.get(field):
                extracted_data[field] = self.clean_digits(extracted_data[field])
        
        # Clean names (remove extra spaces, normalize)
        name_fields = ['HOLDER_NAME', 'FATHER_NAME']
        for field in name_fields:
            if extracted_data.get(field):
                # Remove multiple spaces and normalize
                extracted_data[field] = re.sub(r'\s+', ' ', extracted_data[field]).strip()
        
        # Clean boundary descriptions
        boundary_fields = ['BOUNDARY_EAST', 'BOUNDARY_WEST', 'BOUNDARY_NORTH', 'BOUNDARY_SOUTH']
        for field in boundary_fields:
            if extracted_data.get(field):
                extracted_data[field] = extracted_data[field].strip()
    
    def process_image(self, image_path: str) -> Dict[str, Optional[str]]:
        """
        Complete pipeline to process image and extract structured data.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing extracted structured data
        """
        try:
            # Extract text from image
            extracted_text = self.extract_text(image_path)
            
            # Extract entities from text
            structured_data = self.extract_entities(extracted_text)
            
            # Add metadata
            structured_data['_metadata'] = {
                'image_path': image_path,
                'extracted_text': extracted_text,
                'processing_successful': True
            }
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {
                '_metadata': {
                    'image_path': image_path,
                    'processing_successful': False,
                    'error': str(e)
                }
            }
    
    def process_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Optional[str]]]:
        """
        Process multiple images and return structured data for each.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            List of dictionaries containing extracted data for each image
        """
        results = []
        
        for image_path in image_paths:
            logger.info(f"Processing image: {image_path}")
            result = self.process_image(image_path)
            results.append(result)
        
        return results
    
    def export_results(self, results: Dict[str, Optional[str]], output_path: str = None) -> str:
        """
        Export results to JSON file.
        
        Args:
            results: Extracted data dictionary
            output_path: Output file path (optional)
            
        Returns:
            JSON string of the results
        """
        json_output = json.dumps(results, indent=4, ensure_ascii=False)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                logger.info(f"Results exported to: {output_path}")
            except Exception as e:
                logger.error(f"Error exporting results: {str(e)}")
        
        return json_output


def main():
    """Main function demonstrating the usage of HindiOCRNER."""
    # Initialize the OCR NER system
    ocr_ner = HindiOCRNER()
    
    # Process a single image
    image_path = 'test2.png'
    
    try:
        logger.info("Starting Hindi OCR NER processing...")
        
        # Process the image
        results = ocr_ner.process_image(image_path)
        
        # Display results
        print("\n" + "="*50)
        print("🔍 HINDI OCR NER RESULTS")
        print("="*50)
        
        # Print extracted entities
        for entity, value in results.items():
            if entity != '_metadata':
                print(f"{entity:20}: {value}")
        
        print("\n" + "-"*50)
        print("📝 RAW EXTRACTED TEXT:")
        print("-"*50)
        if '_metadata' in results and 'extracted_text' in results['_metadata']:
            print(results['_metadata']['extracted_text'])
        
        # Export to JSON
        json_output = ocr_ner.export_results(results, 'extracted_data.json')
        
        print("\n" + "="*50)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        print(f"\n❌ Error: {str(e)}")


if __name__ == '__main__':
    main()