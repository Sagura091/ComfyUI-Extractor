"""
Enhanced PDF Processing with Revolutionary OCR
Integrates advanced OCR capabilities for superior text extraction,
table recognition, and mathematical formula processing.
"""

import pdfplumber
import fitz  # PyMuPDF for better image extraction
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any
import json
import logging
import io

from .revolutionary_ocr import create_revolutionary_ocr, OCRResult
from .caption_api import auto_caption
from .settings import OUT_DIR

# Setup logging
logger = logging.getLogger(__name__)

class EnhancedPDFProcessor:
    """Enhanced PDF processor with revolutionary OCR capabilities"""
    
    def __init__(self, ocr_config: Dict[str, Any] = None):
        # Initialize revolutionary OCR
        self.ocr = create_revolutionary_ocr(ocr_config or {
            'languages': ['en'],  # Can be expanded based on document language detection
            'enable_table_recognition': True,
            'enable_formula_recognition': True,
            'enable_handwriting': True,
            'confidence_threshold': 0.7
        })
        
        self.logger = logging.getLogger(__name__)
    
    def process_pdf_advanced(self, path: Path, push_record_func) -> Dict[str, Any]:
        """
        Process PDF with advanced OCR and layout understanding
        """
        doc_id = path.stem
        processing_results = {
            'document_id': doc_id,
            'total_pages': 0,
            'processed_pages': 0,
            'text_regions': [],
            'tables': [],
            'formulas': [],
            'images': [],
            'errors': []
        }
        
        try:
            # Use PyMuPDF for better image extraction and pdfplumber for text
            pdf_doc = fitz.open(path)
            processing_results['total_pages'] = len(pdf_doc)
            
            with pdfplumber.open(path) as plumber_pdf:
                for page_num in range(len(pdf_doc)):
                    try:
                        page_results = self._process_page_advanced(
                            pdf_doc[page_num], 
                            plumber_pdf.pages[page_num], 
                            doc_id, 
                            page_num + 1,
                            push_record_func
                        )
                        
                        # Aggregate results
                        processing_results['text_regions'].extend(page_results.get('text_regions', []))
                        processing_results['tables'].extend(page_results.get('tables', []))
                        processing_results['formulas'].extend(page_results.get('formulas', []))
                        processing_results['images'].extend(page_results.get('images', []))
                        processing_results['processed_pages'] += 1
                        
                    except Exception as e:
                        error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                        self.logger.error(error_msg)
                        processing_results['errors'].append(error_msg)
            
            pdf_doc.close()
            
        except Exception as e:
            error_msg = f"Error processing PDF {path}: {str(e)}"
            self.logger.error(error_msg)
            processing_results['errors'].append(error_msg)
        
        return processing_results
    
    def _process_page_advanced(self, pymupdf_page, plumber_page, doc_id: str, page_num: int, push_record_func) -> Dict[str, Any]:
        """Process a single PDF page with advanced OCR"""
        
        page_results = {
            'text_regions': [],
            'tables': [],
            'formulas': [],
            'images': []
        }
        
        # Create output directory
        out_dir = OUT_DIR / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get page as image for OCR processing
        page_image = self._get_page_as_image(pymupdf_page)
        
        # Process with revolutionary OCR
        ocr_results = self.ocr.process_document_page(page_image)
        
        # Process regular images first (from original PDF)
        image_idx = 0
        for img_dict in plumber_page.images:
            try:
                image_path = self._extract_and_save_image(plumber_page, img_dict, out_dir, page_num, image_idx)
                if image_path:
                    # Get caption using multiple methods
                    caption = self._get_best_caption(plumber_page, img_dict, image_path)
                    
                    # Push to record system
                    push_record_func(
                        doc_id, 
                        f"p{page_num}_img{image_idx}", 
                        caption, 
                        image_path,
                        raw_text=plumber_page.extract_text() or ""
                    )
                    
                    page_results['images'].append({
                        'path': str(image_path),
                        'caption': caption,
                        'bbox': (img_dict["x0"], img_dict["top"], img_dict["x1"], img_dict["bottom"]),
                        'page': page_num
                    })
                    
                    image_idx += 1
            except Exception as e:
                self.logger.warning(f"Failed to process image {image_idx} on page {page_num}: {e}")
        
        # Process OCR-detected text regions
        for idx, text_result in enumerate(ocr_results.get('text_regions', [])):
            if text_result.confidence > 0.5 and text_result.text.strip():
                part_id = f"p{page_num}_text{idx}"
                
                push_record_func(
                    doc_id,
                    part_id,
                    text_result.text,
                    None,  # No image path for text
                    raw_text=text_result.text
                )
                
                page_results['text_regions'].append({
                    'text': text_result.text,
                    'confidence': text_result.confidence,
                    'bbox': text_result.bbox,
                    'language': text_result.language,
                    'page': page_num
                })
        
        # Process OCR-detected tables
        for idx, table_result in enumerate(ocr_results.get('table_regions', [])):
            try:
                # Save table as image
                table_image_path = self._save_table_image(page_image, table_result, out_dir, page_num, idx)
                
                # Create table description
                table_description = self._describe_table(table_result)
                
                push_record_func(
                    doc_id,
                    f"p{page_num}_table{idx}",
                    table_description,
                    table_image_path,
                    raw_text=json.dumps(table_result.get('data', []))
                )
                
                page_results['tables'].append({
                    'data': table_result.get('data', []),
                    'description': table_description,
                    'image_path': str(table_image_path),
                    'bbox': table_result.get('bbox'),
                    'page': page_num
                })
            except Exception as e:
                self.logger.warning(f"Failed to process table {idx} on page {page_num}: {e}")
        
        # Process mathematical formulas
        for idx, formula_result in enumerate(ocr_results.get('formula_regions', [])):
            try:
                # Save formula as image
                formula_image_path = self._save_formula_image(page_image, formula_result, out_dir, page_num, idx)
                
                # Create formula description
                latex_formula = formula_result.metadata.get('latex', formula_result.text)
                description = f"Mathematical formula: {latex_formula}"
                
                push_record_func(
                    doc_id,
                    f"p{page_num}_formula{idx}",
                    description,
                    formula_image_path,
                    raw_text=formula_result.text
                )
                
                page_results['formulas'].append({
                    'text': formula_result.text,
                    'latex': latex_formula,
                    'description': description,
                    'image_path': str(formula_image_path),
                    'bbox': formula_result.bbox,
                    'page': page_num
                })
            except Exception as e:
                self.logger.warning(f"Failed to process formula {idx} on page {page_num}: {e}")
        
        # Process handwriting regions
        for idx, handwriting_result in enumerate(ocr_results.get('handwriting_regions', [])):
            if handwriting_result.confidence > 0.3 and handwriting_result.text.strip():
                push_record_func(
                    doc_id,
                    f"p{page_num}_handwriting{idx}",
                    f"Handwritten text: {handwriting_result.text}",
                    None,
                    raw_text=handwriting_result.text
                )
                
                page_results['text_regions'].append({
                    'text': handwriting_result.text,
                    'confidence': handwriting_result.confidence,
                    'bbox': handwriting_result.bbox,
                    'type': 'handwriting',
                    'page': page_num
                })
        
        return page_results
    
    def _get_page_as_image(self, pymupdf_page, dpi: int = 300) -> np.ndarray:
        """Convert PDF page to high-resolution image for OCR"""
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 300 DPI transformation matrix
        pix = pymupdf_page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(img_data))
        return np.array(image)
    
    def _extract_and_save_image(self, page, img_dict: Dict, out_dir: Path, page_num: int, img_idx: int) -> Path:
        """Extract and save image from PDF page with enhanced quality"""
        try:
            # Use pdfplumber's crop method for better quality
            bbox = (img_dict["x0"], img_dict["top"], img_dict["x1"], img_dict["bottom"])
            img = page.crop(bbox).to_image(resolution=300).original
            
            # Save with high quality
            img_path = out_dir / f"p{page_num}_img{img_idx}.png"
            img.save(img_path, "PNG", optimize=True)
            
            return img_path
        except Exception as e:
            self.logger.warning(f"Failed to extract image: {e}")
            return None
    
    def _save_table_image(self, page_image: np.ndarray, table_result: Dict, out_dir: Path, page_num: int, table_idx: int) -> Path:
        """Save table region as image"""
        bbox = table_result.get('bbox', (0, 0, 100, 100))
        x1, y1, x2, y2 = bbox
        
        # Crop table region
        table_img = page_image[y1:y2, x1:x2]
        
        # Save as PNG
        table_path = out_dir / f"p{page_num}_table{table_idx}.png"
        Image.fromarray(table_img).save(table_path, "PNG")
        
        return table_path
    
    def _save_formula_image(self, page_image: np.ndarray, formula_result: OCRResult, out_dir: Path, page_num: int, formula_idx: int) -> Path:
        """Save formula region as image"""
        x1, y1, x2, y2 = formula_result.bbox
        
        # Crop formula region with some padding
        padding = 10
        formula_img = page_image[max(0, y1-padding):y2+padding, max(0, x1-padding):x2+padding]
        
        # Save as PNG
        formula_path = out_dir / f"p{page_num}_formula{formula_idx}.png"
        Image.fromarray(formula_img).save(formula_path, "PNG")
        
        return formula_path
    
    def _get_best_caption(self, page, img_dict: Dict, image_path: Path) -> str:
        """Get the best caption using multiple methods"""
        # Method 1: Nearby text (existing method)
        nearby_caption = self._get_nearby_caption(page, img_dict)
        if nearby_caption and len(nearby_caption.strip()) > 5:
            return nearby_caption
        
        # Method 2: AI-generated caption
        ai_caption = auto_caption(str(image_path))
        if ai_caption and len(ai_caption.strip()) > 5:
            return ai_caption
        
        # Method 3: OCR on the image itself
        try:
            img_array = np.array(Image.open(image_path))
            ocr_result = self.ocr.extract_text_multilingual(img_array)
            if ocr_result.confidence > 0.7 and len(ocr_result.text.strip()) > 5:
                return f"Image containing text: {ocr_result.text[:100]}"
        except:
            pass
        
        # Fallback
        return f"Image extracted from page (no description available)"
    
    def _get_nearby_caption(self, page, img_dict: Dict, max_distance: int = 60) -> str:
        """Find text near the image that might be a caption"""
        y_bottom = img_dict["bottom"]
        words = page.extract_words()
        
        # Look for text below the image
        nearby_words = [w for w in words if 0 < w["top"] - y_bottom < max_distance]
        
        if nearby_words:
            nearby_words = sorted(nearby_words, key=lambda w: w["x0"])
            return " ".join(w["text"] for w in nearby_words)
        
        return ""
    
    def _describe_table(self, table_result: Dict) -> str:
        """Create a description for a table"""
        data = table_result.get('data', [])
        if not data:
            return "Empty table detected"
        
        rows = len(data)
        cols = len(data[0]) if data else 0
        
        # Try to create a meaningful description
        description = f"Table with {rows} rows and {cols} columns"
        
        # Add header information if available
        if rows > 0 and cols > 0:
            header_row = data[0]
            non_empty_headers = [cell for cell in header_row if cell.strip()]
            if non_empty_headers:
                description += f". Headers: {', '.join(non_empty_headers[:3])}"
                if len(non_empty_headers) > 3:
                    description += "..."
        
        return description


# Backward compatibility function that integrates with existing code
def process_pdf(path: Path, push_record):
    """
    Enhanced PDF processing function with revolutionary OCR
    Maintains compatibility with existing interface while adding advanced features
    """
    try:
        # Initialize enhanced processor
        processor = EnhancedPDFProcessor()
        
        # Process with advanced capabilities
        results = processor.process_pdf_advanced(path, push_record)
        
        # Log processing summary
        logger.info(f"Processed PDF {path.name}: "
                   f"{results['processed_pages']}/{results['total_pages']} pages, "
                   f"{len(results['text_regions'])} text regions, "
                   f"{len(results['tables'])} tables, "
                   f"{len(results['formulas'])} formulas, "
                   f"{len(results['images'])} images")
        
        if results['errors']:
            logger.warning(f"Errors during processing: {results['errors']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process PDF {path}: {e}")
        # Fallback to basic processing
        return _process_pdf_basic(path, push_record)


def _process_pdf_basic(path: Path, push_record):
    """Fallback to basic PDF processing if advanced processing fails"""
    doc_id = path.stem
    try:
        with pdfplumber.open(path) as pdf:
            for pnum, page in enumerate(pdf.pages, 1):
                for idx, img in enumerate(page.images):
                    out_dir = (OUT_DIR / doc_id)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Basic image extraction
                    bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                    page_img = page.crop(bbox).to_image(resolution=300).original
                    img_path = out_dir / f"p{pnum}_{idx}.png"
                    page_img.save(img_path, "PNG")
                    
                    # Basic caption
                    caption = auto_caption(str(img_path)) or f"Image from page {pnum}"
                    
                    push_record(doc_id, f"p{pnum}_{idx}", caption, img_path,
                               raw_text=page.extract_text() or "")
    except Exception as e:
        logger.error(f"Basic PDF processing also failed: {e}")
        raise
