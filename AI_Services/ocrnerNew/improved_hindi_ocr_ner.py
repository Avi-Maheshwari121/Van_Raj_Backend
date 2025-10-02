import re
import json
from typing import Dict, Optional, List, Tuple
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoFeatureExtractor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
import warnings
import logging
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", message="The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor` instead.")


class ImprovedHindiOCRNER:
    """
    Improved Hindi OCR and Named Entity Recognition system specifically optimized
    for Patta documents and similar government papers with enhanced pattern matching
    and image preprocessing.
    """
    
    def __init__(self, model_path: str = './trained_trocr_model'):
        """Initialize the improved Hindi OCR NER system."""
        self.model_path = model_path
        self.processor = None
        self.model = None
        
        # Enhanced and expanded entity patterns based on actual Patta document analysis
        self.entity_patterns = {
            'GRAM_PANCHAYAT': [
                # Standard patterns
                re.compile(r'ग्राम\s*प(?:चायत|चायत्त|ंचायत)\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'ग्राम\s*पंचायत\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'ग्राम\s*प\s*चायत\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                # From document structure
                re.compile(r'ग्राम\s*([^\s\n,\.]+)\s*जनपद', re.IGNORECASE),
                re.compile(r'कार्यालय\s*ग्राम\s*पंचायत[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
            ],
            'JANPAD_PANCHAYAT': [
                re.compile(r'जनपद\s*प(?:चायत|चायत्त|ंचायत)\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'जनपद\s*पंचायत\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'जनपद\s*([^\s\n,\.]+)\s*तहसील', re.IGNORECASE),
                re.compile(r'जनपद\s*पंचायत[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
            ],
            'TEHSIL': [
                re.compile(r'तहसील\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'तेहसील\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'तह\s*सील\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'तहसील[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
            ],
            'DISTRICT': [
                re.compile(r'जिला\s*[\.:\-\s]*([^\s\n,\(\.]+)', re.IGNORECASE),
                re.compile(r'जि\s*ला\s*[\.:\-\s]*([^\s\n,\(\.]+)', re.IGNORECASE),
                re.compile(r'ज़िला\s*[\.:\-\s]*([^\s\n,\(\.]+)', re.IGNORECASE),
                re.compile(r'जिला[\.:\-\s]*([^\s\n,\(\.]+)', re.IGNORECASE),
            ],
            'SERIAL_NO': [
                # Standard patterns
                re.compile(r'(?:क्रमांक|क्रमांक\s*संख्या)\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'क्रमांक[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                # Document specific - serial number format 125/2025
                re.compile(r'क्रमांक\s*[\.:\-\s]*(\d+/\d+)', re.IGNORECASE),
                re.compile(r'(\d{2,3}/\d{4})', re.IGNORECASE),  # Direct number pattern
                # From document structure
                re.compile(r'[^\w](\d+/20\d{2})[^\w]'),
            ],
            'DATE': [
                re.compile(r'दिनांक\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'दिनाक\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'दि\s*नांक\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'तारीख\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                # Date format patterns (DD/MM/YYYY)
                re.compile(r'(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE),
                re.compile(r'(\d{1,2}\.\d{1,2}\.\d{4})', re.IGNORECASE),
            ],
            'HOLDER_NAME': [
                # Enhanced patterns for name extraction
                re.compile(r'(?:श्री\s*/\s*श्रीमति|श्री\s*/\s*श्रोमति)\s*[\.:\-\s]*([^\.]+?)(?:\s*पिता|$)', re.IGNORECASE),
                re.compile(r'(?:श्री|श्रीमति)\s*([^\.]+?)\s*(?:पुत्र|पुत्री|पिता)', re.IGNORECASE),
                re.compile(r'नाम\s*[\.:\-\s]*([^\.]+?)\s*(?:पिता|पति)', re.IGNORECASE),
                # From document - specific name pattern
                re.compile(r'श्री\s*/\s*श्रीमति[\.:\-\s]*([^\.]+?)(?:\s*पिता|\s*पति)', re.IGNORECASE),
                # Extract names that appear before "पिता" or "पति"
                re.compile(r'([^\n\.]+?)\s*(?:पिता|पति)\s*श्री', re.IGNORECASE),
            ],
            'FATHER_NAME': [
                re.compile(r'(?:पिता\s*[,/]\s*पति|पिता\s*/\s*पति)\s*श्री\s*([^\.]+?)(?:\s*जाति|$)', re.IGNORECASE),
                re.compile(r'पिता\s*का\s*नाम\s*[\.:\-\s]*श्री\s*([^\.]+?)(?:\s|$)', re.IGNORECASE),
                re.compile(r'पिता\s*श्री\s*([^\.]+?)(?:\s*जाति|$)', re.IGNORECASE),
                re.compile(r'पति\s*श्री\s*([^\.]+?)(?:\s*जाति|$)', re.IGNORECASE),
                # From document structure
                re.compile(r'पिता\s*[,/]\s*पति\s*श्री\s*([^\s\n\.]+)', re.IGNORECASE),
            ],
            'CASTE': [
                # Additional pattern for caste extraction
                re.compile(r'जाति\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'जाति[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
            ],
            'KHASRA_NO': [
                re.compile(r'खसरा\s*नं\.\s*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'खसरा\s*नंबर\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'खस\s*रा\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'खसरा\s*([^\s\n,]+)', re.IGNORECASE),
            ],
            'TOTAL_AREA_SQFT': [
                re.compile(r'कुल\s*क्षेत्रफल\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'क्षेत्रफल\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'एरिया\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
                # From document - area mentioned
                re.compile(r'(\d+(?:\.\d+)?)\s*(?:वर्ग\s*फुट|sq\s*ft)', re.IGNORECASE),
            ],
            'REVENUE_VILLAGE': [
                # Pattern for revenue village
                re.compile(r'राजस्व\s*ग्राम\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
                re.compile(r'राजस्व\s*गांव\s*[\.:\-\s]*([^\s\n,\.]+)', re.IGNORECASE),
            ],
            'SURVEY_NO': [
                # Survey number patterns
                re.compile(r'सर्वे\s*नं\.\s*([^\s\n,]+)', re.IGNORECASE),
                re.compile(r'सर्वे\s*संख्या\s*[\.:\-\s]*([^\s\n,]+)', re.IGNORECASE),
            ],
            'BOUNDARY_EAST': [
                re.compile(r'पूर्व\s*में\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'पूव\s*मे\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'पूर्व\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'पूर्व\s*में\s*[:।]\s*([^,\n]+)', re.IGNORECASE),
            ],
            'BOUNDARY_WEST': [
                re.compile(r'पश्चिम\s*में\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'पच्छिम\s*में\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'पश्चिम\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
            ],
            'BOUNDARY_NORTH': [
                re.compile(r'उत्तर\s*में\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'उत्तर\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'उ\s*त्तर\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
            ],
            'BOUNDARY_SOUTH': [
                re.compile(r'दक्षिण\s*में\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'दक्षेण\s*में\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
                re.compile(r'दक्षिण\s*[\.:\-\s]*([^,\n]+)', re.IGNORECASE),
            ],
        }
        
        # Enhanced digit mapping including more variations
        self.hindi_to_english_digits = str.maketrans('१२३४५६७८९०', '1234567890')
        
        # Common OCR corrections for Hindi text
        self.text_corrections = {
            'ग्राम पचायत': 'ग्राम पंचायत',
            'जनपद पचायत': 'जनपद पंचायत',
            'तेहसील': 'तहसील',
            'दिनाक': 'दिनांक',
            'क्रमाक': 'क्रमांक',
            'खसरा न': 'खसरा नं',
        }
        
    def load_model(self) -> None:
        """Load the TrOCR model with enhanced error handling."""
        try:
            logger.info("Loading enhanced Hindi TrOCR model...")
            
            # Model configuration
            encoder_model = 'google/vit-base-patch16-224-in21k'
            decoder_model = 'surajp/RoBERTa-hindi-guj-san'
            
            # Initialize processor
            image_processor = AutoFeatureExtractor.from_pretrained(encoder_model)
            tokenizer = AutoTokenizer.from_pretrained(decoder_model)
            self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            
            # Try to load fine-tuned model
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            logger.info("Fine-tuned model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}")
            logger.info("Falling back to base TrOCR model...")
            
            try:
                # Fallback to base model
                self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                logger.info("Base model loaded successfully!")
            except Exception as e2:
                logger.error(f"Error loading base model: {str(e2)}")
                raise e2
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Enhanced image preprocessing for better OCR results.
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Apply Gaussian blur to reduce noise
            img_blurred = cv2.GaussianBlur(img_bgr, (1, 1), 0)
            
            # Enhance contrast
            lab = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            
            # Merge channels and convert back
            limg = cv2.merge((cl, a, b))
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Convert back to RGB
            enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL Image
            enhanced_pil = Image.fromarray(enhanced_img_rgb)
            
            # Additional PIL enhancements
            enhancer = ImageEnhance.Sharpness(enhanced_pil)
            enhanced_pil = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Contrast(enhanced_pil)
            enhanced_pil = enhancer.enhance(1.1)
            
            return enhanced_pil
            
        except Exception as e:
            logger.warning(f"Error in image preprocessing: {str(e)}, using original image")
            return Image.open(image_path).convert('RGB')
    
    def correct_text(self, text: str) -> str:
        """Apply common OCR corrections to Hindi text."""
        corrected_text = text
        for wrong, correct in self.text_corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)
        return corrected_text
    
    def extract_text(self, image_path: str) -> str:
        """Enhanced text extraction with preprocessing."""
        if self.model is None or self.processor is None:
            self.load_model()
        
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            # Preprocess image
            image = self.preprocess_image(image_path)
            
            # Generate text with enhanced parameters
            pixel_values = self.processor(image, return_tensors='pt').pixel_values
            generated_ids = self.model.generate(
                pixel_values,
                max_length=1024,  # Increased for longer documents
                num_beams=5,      # More beams for better quality
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=True,
                temperature=0.6,
                repetition_penalty=1.2
            )
            
            extracted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Apply text corrections
            corrected_text = self.correct_text(extracted_text)
            
            logger.info("Text extraction completed")
            logger.debug(f"Extracted text: {corrected_text[:500]}...")
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error during text extraction: {str(e)}")
            raise
    
    def clean_digits(self, text: str) -> str:
        """Enhanced digit cleaning with better Hindi support."""
        if not text:
            return None
        
        # Convert Hindi digits to English
        cleaned_text = text.translate(self.hindi_to_english_digits)
        
        # Handle common OCR mistakes
        cleaned_text = cleaned_text.replace('o', '0').replace('O', '0')
        cleaned_text = cleaned_text.replace('l', '1').replace('I', '1')
        
        # Keep digits, forward slash, period, and hyphen
        cleaned_text = re.sub(r'[^\d\/\.\-]', '', cleaned_text)
        
        return cleaned_text if cleaned_text else None
    
    def clean_and_format_date(self, text: str) -> str:
        """Enhanced date formatting with better recognition."""
        if not text:
            return None
        
        # Convert Hindi digits first
        text = text.translate(self.hindi_to_english_digits)
        
        # Try to extract date pattern
        date_patterns = [
            r'(\d{1,2})[/\.\-](\d{1,2})[/\.\-](\d{4})',
            r'(\d{1,2})[/\.\-](\d{1,2})[/\.\-](\d{2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                day, month, year = match.groups()
                
                # Handle 2-digit years
                if len(year) == 2:
                    if int(year) < 50:
                        year = '20' + year
                    else:
                        year = '19' + year
                
                return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        
        # If no pattern matched, try to extract just digits
        digits = re.sub(r'\D', '', text)
        if len(digits) >= 6:
            if len(digits) == 6:  # DDMMYY
                day = digits[0:2]
                month = digits[2:4]
                year = '20' + digits[4:6]
            elif len(digits) == 8:  # DDMMYYYY
                day = digits[0:2]
                month = digits[2:4]
                year = digits[4:8]
            else:
                return text
            
            return f"{day}/{month}/{year}"
        
        return text
    
    def extract_entity_with_patterns(self, text: str, entity_name: str) -> Optional[str]:
        """Enhanced entity extraction with better pattern matching."""
        patterns = self.entity_patterns.get(entity_name, [])
        
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                # Handle different types of matches
                if isinstance(matches[0], tuple):
                    # Multiple groups in regex
                    for match in matches:
                        for group in match:
                            if group and group.strip():
                                return group.strip().replace('\n', ' ')
                else:
                    # Single group or string match
                    for match in matches:
                        if match and match.strip():
                            return match.strip().replace('\n', ' ')
        
        return None
    
    def extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """Enhanced entity extraction with better processing."""
        logger.info("Extracting entities from text...")
        
        extracted_data = {}
        
        # Extract each entity using multiple patterns
        for entity_name in self.entity_patterns.keys():
            extracted_data[entity_name] = self.extract_entity_with_patterns(text, entity_name)
        
        # Enhanced post-processing
        self._enhanced_post_process_entities(extracted_data, text)
        
        logger.info("Entity extraction completed")
        return extracted_data
    
    def _enhanced_post_process_entities(self, extracted_data: Dict[str, Optional[str]], original_text: str) -> None:
        """Enhanced post-processing with better validation."""
        
        # Clean and format date
        if extracted_data.get('DATE'):
            extracted_data['DATE'] = self.clean_and_format_date(extracted_data['DATE'])
        
        # Clean numeric fields
        numeric_fields = ['SERIAL_NO', 'KHASRA_NO', 'TOTAL_AREA_SQFT', 'SURVEY_NO']
        for field in numeric_fields:
            if extracted_data.get(field):
                extracted_data[field] = self.clean_digits(extracted_data[field])
        
        # Clean and validate names
        name_fields = ['HOLDER_NAME', 'FATHER_NAME']
        for field in name_fields:
            if extracted_data.get(field):
                name = extracted_data[field]
                # Remove extra spaces and clean
                name = re.sub(r'\s+', ' ', name).strip()
                # Remove trailing punctuation
                name = re.sub(r'[\.,:;]+$', '', name)
                # Ensure proper capitalization
                extracted_data[field] = name.title() if name else None
        
        # Clean administrative divisions
        admin_fields = ['GRAM_PANCHAYAT', 'JANPAD_PANCHAYAT', 'TEHSIL', 'DISTRICT', 'REVENUE_VILLAGE']
        for field in admin_fields:
            if extracted_data.get(field):
                value = extracted_data[field]
                # Clean and format
                value = re.sub(r'\s+', ' ', value).strip()
                value = re.sub(r'[\.,:;]+$', '', value)
                extracted_data[field] = value if value else None
        
        # Special handling for boundaries
        boundary_fields = ['BOUNDARY_EAST', 'BOUNDARY_WEST', 'BOUNDARY_NORTH', 'BOUNDARY_SOUTH']
        for field in boundary_fields:
            if extracted_data.get(field):
                boundary = extracted_data[field]
                # Clean boundary description
                boundary = re.sub(r'\s+', ' ', boundary).strip()
                boundary = re.sub(r'^[:।\-\s]+', '', boundary)  # Remove leading punctuation
                boundary = re.sub(r'[\.,:;]+$', '', boundary)   # Remove trailing punctuation
                extracted_data[field] = boundary if boundary else None
    
    def process_image(self, image_path: str) -> Dict[str, Optional[str]]:
        """Complete enhanced pipeline for image processing."""
        try:
            # Extract text from image
            extracted_text = self.extract_text(image_path)
            
            # Extract entities from text
            structured_data = self.extract_entities(extracted_text)
            
            # Add metadata
            structured_data['_metadata'] = {
                'image_path': image_path,
                'extracted_text': extracted_text,
                'processing_successful': True,
                'extracted_entities_count': sum(1 for v in structured_data.values() if v is not None)
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
    
    def export_results(self, results: Dict[str, Optional[str]], output_path: str = None) -> str:
        """Export results with enhanced formatting."""
        # Create a clean version for export (without metadata for main display)
        clean_results = {k: v for k, v in results.items() if k != '_metadata' and v is not None}
        
        export_data = {
            'extracted_entities': clean_results,
            'metadata': results.get('_metadata', {}),
            'summary': {
                'total_entities_found': len(clean_results),
                'extraction_successful': results.get('_metadata', {}).get('processing_successful', False)
            }
        }
        
        json_output = json.dumps(export_data, indent=4, ensure_ascii=False)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                logger.info(f"Results exported to: {output_path}")
            except Exception as e:
                logger.error(f"Error exporting results: {str(e)}")
        
        return json_output


def main():
    """Enhanced main function for testing improved OCR."""
    # Initialize the improved OCR NER system
    ocr_ner = ImprovedHindiOCRNER()
    
    # Test with available images
    test_images = ['test2.png', 'test_patta.png', 'test_patta_new.png']
    
    for image_path in test_images:
        try:
            if not os.path.exists(image_path):
                continue
                
            print(f"\n{'='*60}")
            print(f"🔍 PROCESSING: {image_path}")
            print(f"{'='*60}")
            
            # Process the image
            results = ocr_ner.process_image(image_path)
            
            if results.get('_metadata', {}).get('processing_successful', False):
                print("✅ Processing successful!")
                
                # Display extracted entities
                print(f"\n📋 EXTRACTED ENTITIES:")
                print(f"{'-'*40}")
                entity_count = 0
                for entity, value in results.items():
                    if entity != '_metadata' and value:
                        print(f"{entity:20}: {value}")
                        entity_count += 1
                
                print(f"\n📊 SUMMARY: {entity_count} entities extracted")
                
                # Show partial raw text
                raw_text = results.get('_metadata', {}).get('extracted_text', '')
                if raw_text:
                    print(f"\n📝 RAW TEXT (first 300 chars):")
                    print(f"{'-'*40}")
                    print(raw_text[:300] + "..." if len(raw_text) > 300 else raw_text)
                
                # Export results
                output_file = f"improved_results_{image_path.replace('.png', '.json')}"
                ocr_ner.export_results(results, output_file)
                print(f"\n💾 Results saved to: {output_file}")
                
            else:
                print("❌ Processing failed!")
                error = results.get('_metadata', {}).get('error', 'Unknown error')
                print(f"Error: {error}")
            
            break  # Process only the first available image for demo
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("🎉 IMPROVED HINDI OCR NER TESTING COMPLETED")
    print(f"{'='*60}")


if __name__ == '__main__':
    import os
    main()