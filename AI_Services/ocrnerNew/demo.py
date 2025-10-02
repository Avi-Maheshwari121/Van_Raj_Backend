"""
Hindi OCR NER Demo Script
=========================

This script demonstrates the complete Hindi OCR and Named Entity Recognition pipeline
for extracting structured data from Hindi government documents like Patta papers.

Features:
- Enhanced Hindi text recognition using TrOCR
- Multi-pattern entity extraction with regex
- Data cleaning and validation
- JSON export functionality
- Batch processing support
"""

import os
import sys
from typing import List
import json
from hindi_ocr_ner import HindiOCRNER

def test_single_image():
    """Test OCR NER on a single image."""
    print("🔍 Testing Single Image Processing")
    print("=" * 50)
    
    # Initialize the OCR NER system
    ocr_ner = HindiOCRNER()
    
    # Test images available in the directory
    test_images = ['test2.png', 'test_patta.png', 'test_patta_new.png']
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n📷 Processing: {image_path}")
            print("-" * 30)
            
            # Process the image
            results = ocr_ner.process_image(image_path)
            
            # Display results
            if results.get('_metadata', {}).get('processing_successful', False):
                print("✅ Processing successful!")
                
                # Show extracted entities
                print("\n📋 Extracted Entities:")
                for entity, value in results.items():
                    if entity != '_metadata' and value:
                        print(f"  {entity:20}: {value}")
                
                # Show raw text (truncated)
                raw_text = results.get('_metadata', {}).get('extracted_text', '')
                if raw_text:
                    print(f"\n📝 Raw Text (first 200 chars): {raw_text[:200]}...")
            else:
                print("❌ Processing failed!")
                error = results.get('_metadata', {}).get('error', 'Unknown error')
                print(f"Error: {error}")
            
            print("\n" + "="*50)
            break
    else:
        print("❌ No test images found!")

def test_batch_processing():
    """Test batch processing of multiple images."""
    print("\n🔄 Testing Batch Processing")
    print("=" * 50)
    
    # Initialize the OCR NER system
    ocr_ner = HindiOCRNER()
    
    # Get all available test images
    test_images = []
    for img in ['test2.png', 'test_patta.png', 'test_patta_new.png']:
        if os.path.exists(img):
            test_images.append(img)
    
    if not test_images:
        print("❌ No test images found for batch processing!")
        return
    
    print(f"📷 Processing {len(test_images)} images...")
    
    # Process all images
    results = ocr_ner.process_multiple_images(test_images)
    
    # Summary statistics
    successful = sum(1 for r in results if r.get('_metadata', {}).get('processing_successful', False))
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"  Total images: {len(test_images)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(test_images) - successful}")
    
    # Export batch results
    batch_output = {
        'summary': {
            'total_images': len(test_images),
            'successful_count': successful,
            'failed_count': len(test_images) - successful
        },
        'results': results
    }
    
    with open('batch_results.json', 'w', encoding='utf-8') as f:
        json.dump(batch_output, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Batch results saved to 'batch_results.json'")

def test_entity_patterns():
    """Test individual entity extraction patterns."""
    print("\n🧪 Testing Entity Extraction Patterns")
    print("=" * 50)
    
    # Sample Hindi text for testing patterns
    sample_texts = [
        "ग्राम पंचायत: रामपुर",
        "जनपद पचायत धमधा",
        "तहसील: बिलासपुर",
        "जिला रायपुर",
        "क्रमांक 125/2025",
        "दिनांक 15/03/2025",
        "श्री / श्रीमति. राम कुमार पिता / पति",
        "खसरा नं. 152/1",
        "कुल क्षेत्रफल 2400",
        "पूर्व में: सड़क",
        "पश्चिम में: खेत"
    ]
    
    ocr_ner = HindiOCRNER()
    
    print("🔍 Testing individual patterns:")
    for text in sample_texts:
        print(f"\n📝 Text: {text}")
        entities = ocr_ner.extract_entities(text)
        found_entities = {k: v for k, v in entities.items() if v is not None}
        if found_entities:
            for entity, value in found_entities.items():
                print(f"  ✅ {entity}: {value}")
        else:
            print("  ❌ No entities found")

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n📋 Generating Test Report")
    print("=" * 50)
    
    report = {
        "test_summary": {
            "timestamp": "2025-09-30",
            "system": "Hindi OCR NER",
            "version": "1.0"
        },
        "entity_patterns_tested": list(HindiOCRNER().entity_patterns.keys()),
        "test_images": [],
        "features_tested": [
            "Text extraction with TrOCR",
            "Multi-pattern entity extraction",
            "Hindi digit conversion",
            "Date formatting",
            "Data cleaning and validation",
            "JSON export",
            "Batch processing"
        ]
    }
    
    # Check available test images
    for img in ['test2.png', 'test_patta.png', 'test_patta_new.png']:
        if os.path.exists(img):
            report["test_images"].append({
                "filename": img,
                "status": "available"
            })
        else:
            report["test_images"].append({
                "filename": img,
                "status": "not_found"
            })
    
    # Save report
    with open('test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("📄 Test report generated: 'test_report.json'")

def main():
    """Main demo function."""
    print("🚀 Hindi OCR NER Demo")
    print("=" * 60)
    print("This demo showcases Hindi text recognition and entity extraction")
    print("for government documents like Patta papers.")
    print("=" * 60)
    
    try:
        # Test single image processing
        test_single_image()
        
        # Test batch processing
        test_batch_processing()
        
        # Test entity patterns
        test_entity_patterns()
        
        # Generate test report
        generate_test_report()
        
        print("\n🎉 Demo completed successfully!")
        print("\nGenerated files:")
        print("  📄 extracted_data.json - Single image results")
        print("  📄 batch_results.json - Batch processing results")
        print("  📄 test_report.json - Comprehensive test report")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()