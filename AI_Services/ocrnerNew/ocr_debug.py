"""
Simple OCR tester to debug text extraction issues
"""
import easyocr
from PIL import Image
import cv2
import numpy as np

def test_easyocr(image_path):
    """Test with EasyOCR which has better Hindi support."""
    print(f"🔍 Testing EasyOCR with {image_path}")
    print("-" * 40)
    
    try:
        # Initialize EasyOCR with Hindi and English
        reader = easyocr.Reader(['hi', 'en'])  # Hindi and English
        
        # Read text from image
        results = reader.readtext(image_path)
        
        print("✅ EasyOCR Results:")
        extracted_text = ""
        for (bbox, text, confidence) in results:
            print(f"  Text: {text}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Bbox: {bbox}")
            print("-" * 20)
            extracted_text += text + " "
        
        print(f"\n📝 Combined Text: {extracted_text}")
        return extracted_text
        
    except Exception as e:
        print(f"❌ EasyOCR Error: {e}")
        return None

def preprocess_image_for_ocr(image_path):
    """Preprocess image for better OCR."""
    print(f"🔧 Preprocessing image: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((1,1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel, iterations=1)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Save preprocessed image
    preprocessed_path = image_path.replace('.png', '_preprocessed.png')
    cv2.imwrite(preprocessed_path, enhanced)
    
    print(f"💾 Preprocessed image saved as: {preprocessed_path}")
    return preprocessed_path

def main():
    """Test OCR with different approaches."""
    
    # Install easyocr if not available
    try:
        import easyocr
    except ImportError:
        print("📦 Installing EasyOCR...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'easyocr'])
        import easyocr
    
    # Test images
    test_images = ['test2.png', 'test_patta.png', 'test_patta_new.png']
    
    for image_path in test_images:
        try:
            print(f"\n{'='*60}")
            print(f"🔍 TESTING: {image_path}")
            print(f"{'='*60}")
            
            # Test with original image
            text = test_easyocr(image_path)
            
            if text and text.strip() != "":
                print(f"\n✅ Successfully extracted text with EasyOCR!")
                
                # Test basic pattern matching
                print(f"\n🔍 Testing basic pattern matching:")
                
                import re
                
                # Test for numbers
                numbers = re.findall(r'\d+', text)
                if numbers:
                    print(f"  Numbers found: {numbers}")
                
                # Test for common Hindi words
                hindi_words = ['ग्राम', 'पंचायत', 'जनपद', 'तहसील', 'जिला', 'क्रमांक', 'दिनांक']
                for word in hindi_words:
                    if word in text:
                        print(f"  Found Hindi word: {word}")
                
                break
            else:
                print(f"❌ No text extracted from {image_path}")
                
                # Try with preprocessed image
                print(f"\n🔧 Trying with preprocessed image...")
                preprocessed = preprocess_image_for_ocr(image_path)
                preprocessed_text = test_easyocr(preprocessed)
                
                if preprocessed_text and preprocessed_text.strip():
                    print(f"✅ Preprocessed image worked better!")
                    break
                    
        except Exception as e:
            print(f"❌ Error testing {image_path}: {e}")
            continue

if __name__ == '__main__':
    main()