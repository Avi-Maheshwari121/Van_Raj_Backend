# Test script specifically for the new Patta document
from improved_hindi_ocr_ner import ImprovedHindiOCRNER
import json

def test_patta_document():
    """Test the improved system with the actual Patta document."""
    print("🔍 Testing Improved Hindi OCR NER with Patta Document")
    print("=" * 60)
    
    # Initialize the improved OCR system
    ocr_ner = ImprovedHindiOCRNER()
    
    # Test images
    test_images = ['test2.png', 'test_patta.png', 'test_patta_new.png']
    
    for image_path in test_images:
        try:
            print(f"\n📷 Processing: {image_path}")
            print("-" * 40)
            
            # Process the image
            results = ocr_ner.process_image(image_path)
            
            if results.get('_metadata', {}).get('processing_successful', False):
                print("✅ Processing successful!")
                
                # Display all extracted entities
                print(f"\n📋 EXTRACTED ENTITIES:")
                entities_found = 0
                for entity, value in results.items():
                    if entity != '_metadata':
                        if value:
                            print(f"  {entity:20}: {value}")
                            entities_found += 1
                        else:
                            print(f"  {entity:20}: [NOT FOUND]")
                
                print(f"\n📊 Total entities found: {entities_found}")
                
                # Show raw text for analysis
                raw_text = results.get('_metadata', {}).get('extracted_text', '')
                if raw_text:
                    print(f"\n📝 Raw extracted text:")
                    print("-" * 40)
                    print(raw_text)
                
                # Save detailed results
                with open(f'patta_test_{image_path.replace(".png", ".json")}', 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"\n💾 Detailed results saved to: patta_test_{image_path.replace('.png', '.json')}")
                
            else:
                print("❌ Processing failed!")
                error = results.get('_metadata', {}).get('error', 'Unknown error')
                print(f"Error: {error}")
            
            print("\n" + "="*60)
            break  # Test first available image
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue

if __name__ == '__main__':
    test_patta_document()