"""
Revolutionary OCR & Text Recognition Module
Next-generation OCR with multi-language support, advanced layout understanding,
mathematical formula extraction, and form field detection.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# Advanced OCR Libraries
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
    print("Warning: PaddleOCR not installed. Install with: pip install paddleocr")

try:
    import easyocr
except ImportError:
    easyocr = None
    print("Warning: EasyOCR not installed. Install with: pip install easyocr")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    TrOCRProcessor = VisionEncoderDecoderModel = None
    print("Warning: Transformers not installed for TrOCR")

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Mathematical formula recognition
try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ImportError:
    sympy = parse_latex = None

@dataclass
class OCRResult:
    """Enhanced OCR result with confidence and metadata"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    language: str
    text_type: str  # 'text', 'formula', 'table', 'form_field', 'handwriting'
    metadata: Dict[str, Any]

@dataclass
class DocumentLayout:
    """Document layout analysis result"""
    text_regions: List[Dict]
    table_regions: List[Dict]
    image_regions: List[Dict]
    formula_regions: List[Dict]
    form_fields: List[Dict]
    reading_order: List[int]
    confidence: float

class RevolutionaryOCR:
    """
    Revolutionary OCR system with advanced capabilities:
    - Multi-language support (80+ languages)
    - Mathematical formula extraction
    - Table structure recognition
    - Handwriting recognition
    - Form field detection
    - Layout understanding
    """
    
    def __init__(self, 
                 languages: List[str] = ['en'],
                 enable_table_recognition: bool = True,
                 enable_formula_recognition: bool = True,
                 enable_handwriting: bool = True,
                 confidence_threshold: float = 0.7):
        
        self.languages = languages
        self.enable_table_recognition = enable_table_recognition
        self.enable_formula_recognition = enable_formula_recognition
        self.enable_handwriting = enable_handwriting
        self.confidence_threshold = confidence_threshold
        
        # Initialize OCR engines
        self._init_ocr_engines()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _init_ocr_engines(self):
        """Initialize multiple OCR engines for ensemble processing"""
        self.ocr_engines = {}
        
        # PaddleOCR - Best multilingual support
        if PaddleOCR:
            try:
                self.ocr_engines['paddle'] = PaddleOCR(
                    use_angle_cls=True,
                    lang='en' if 'en' in self.languages else self.languages[0],
                    show_log=False,
                    use_gpu=True
                )
                self.logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize PaddleOCR: {e}")
        
        # EasyOCR - Good for handwriting
        if easyocr and self.enable_handwriting:
            try:
                self.ocr_engines['easy'] = easyocr.Reader(
                    self.languages,
                    gpu=True
                )
                self.logger.info("EasyOCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EasyOCR: {e}")
        
        # TrOCR for handwriting recognition
        if TrOCRProcessor and self.enable_handwriting:
            try:
                self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                self.logger.info("TrOCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TrOCR: {e}")
        
        # Tesseract as fallback
        if pytesseract:
            self.ocr_engines['tesseract'] = pytesseract
            self.logger.info("Tesseract available as fallback")
    
    def analyze_layout(self, image: np.ndarray) -> DocumentLayout:
        """
        Advanced layout analysis to identify different content regions
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect text regions using MSER (Maximally Stable Extremal Regions)
        text_regions = self._detect_text_regions(gray)
        
        # Detect table regions
        table_regions = self._detect_table_regions(gray) if self.enable_table_recognition else []
        
        # Detect mathematical formulas
        formula_regions = self._detect_formula_regions(gray) if self.enable_formula_recognition else []
        
        # Detect form fields
        form_fields = self._detect_form_fields(gray)
        
        # Detect image regions
        image_regions = self._detect_image_regions(gray)
        
        # Determine reading order
        reading_order = self._determine_reading_order(text_regions + table_regions + formula_regions)
        
        return DocumentLayout(
            text_regions=text_regions,
            table_regions=table_regions,
            image_regions=image_regions,
            formula_regions=formula_regions,
            form_fields=form_fields,
            reading_order=reading_order,
            confidence=0.85  # Placeholder confidence
        )
    
    def _detect_text_regions(self, gray: np.ndarray) -> List[Dict]:
        """Detect text regions using computer vision techniques"""
        # MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w > 20 and h > 10 and w/h < 10:  # Filter reasonable text regions
                text_regions.append({
                    'bbox': (x, y, x+w, y+h),
                    'type': 'text',
                    'confidence': 0.8
                })
        
        return text_regions
    
    def _detect_table_regions(self, gray: np.ndarray) -> List[Dict]:
        """Detect table regions using line detection"""
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines to find table structures
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours of potential tables
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 50:  # Minimum table size
                table_regions.append({
                    'bbox': (x, y, x+w, y+h),
                    'type': 'table',
                    'confidence': 0.75
                })
        
        return table_regions
    
    def _detect_formula_regions(self, gray: np.ndarray) -> List[Dict]:
        """Detect mathematical formula regions"""
        # Look for mathematical symbols and patterns
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        
        # Detect special characters common in formulas
        formula_patterns = [
            r'[∑∏∫∂∇√±×÷≤≥≠∞πθφλμσΔΩαβγδε]',
            r'[\^_{}()[\]]',  # Superscript/subscript indicators
            r'[0-9]+\s*[+\-*/=]\s*[0-9]+',  # Mathematical expressions
        ]
        
        # This would be enhanced with actual formula detection algorithms
        formula_regions = []
        
        return formula_regions
    
    def _detect_form_fields(self, gray: np.ndarray) -> List[Dict]:
        """Detect form fields like checkboxes, text fields, etc."""
        # Detect rectangular regions that could be form fields
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        form_fields = []
        for contour in contours:
            # Check if contour is rectangular
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify field type based on dimensions
                if w > h * 3:  # Text field
                    field_type = 'text_field'
                elif abs(w - h) < 10:  # Checkbox
                    field_type = 'checkbox'
                else:
                    field_type = 'form_field'
                
                form_fields.append({
                    'bbox': (x, y, x+w, y+h),
                    'type': field_type,
                    'confidence': 0.7
                })
        
        return form_fields
    
    def _detect_image_regions(self, gray: np.ndarray) -> List[Dict]:
        """Detect image/figure regions"""
        # Find large connected components that aren't text
        # This is a simplified implementation
        
        image_regions = []
        return image_regions
    
    def _determine_reading_order(self, regions: List[Dict]) -> List[int]:
        """Determine the reading order of regions"""
        # Sort regions by y-coordinate first, then x-coordinate
        sorted_regions = sorted(enumerate(regions), 
                              key=lambda x: (x[1]['bbox'][1], x[1]['bbox'][0]))
        return [idx for idx, _ in sorted_regions]
    
    def extract_text_multilingual(self, image: np.ndarray, region: Dict = None) -> OCRResult:
        """
        Extract text using ensemble of multilingual OCR engines
        """
        if region:
            x1, y1, x2, y2 = region['bbox']
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        results = []
        
        # Try PaddleOCR first (best multilingual support)
        if 'paddle' in self.ocr_engines:
            try:
                paddle_result = self.ocr_engines['paddle'].ocr(roi, cls=True)
                if paddle_result and paddle_result[0]:
                    for line in paddle_result[0]:
                        bbox, (text, confidence) = line
                        results.append(OCRResult(
                            text=text,
                            confidence=confidence,
                            bbox=self._normalize_bbox(bbox, region),
                            language='auto',
                            text_type='text',
                            metadata={'engine': 'paddle'}
                        ))
            except Exception as e:
                self.logger.warning(f"PaddleOCR failed: {e}")
        
        # Try EasyOCR for additional validation
        if 'easy' in self.ocr_engines and self.enable_handwriting:
            try:
                easy_results = self.ocr_engines['easy'].readtext(roi)
                for bbox, text, confidence in easy_results:
                    if confidence > self.confidence_threshold:
                        results.append(OCRResult(
                            text=text,
                            confidence=confidence,
                            bbox=self._normalize_bbox(bbox, region),
                            language='auto',
                            text_type='text',
                            metadata={'engine': 'easy'}
                        ))
            except Exception as e:
                self.logger.warning(f"EasyOCR failed: {e}")
        
        # Return best result or ensemble result
        if results:
            return self._ensemble_results(results)
        else:
            return OCRResult("", 0.0, (0, 0, 0, 0), "unknown", "text", {})
    
    def extract_handwriting(self, image: np.ndarray, region: Dict = None) -> OCRResult:
        """
        Extract handwritten text using specialized models
        """
        if region:
            x1, y1, x2, y2 = region['bbox']
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        # Convert to PIL Image for TrOCR
        pil_image = Image.fromarray(roi)
        
        try:
            if hasattr(self, 'trocr_processor'):
                pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
                generated_ids = self.trocr_model.generate(pixel_values)
                generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                return OCRResult(
                    text=generated_text,
                    confidence=0.8,  # TrOCR doesn't provide confidence scores
                    bbox=region['bbox'] if region else (0, 0, roi.shape[1], roi.shape[0]),
                    language='en',
                    text_type='handwriting',
                    metadata={'engine': 'trocr'}
                )
        except Exception as e:
            self.logger.warning(f"TrOCR failed: {e}")
        
        # Fallback to EasyOCR for handwriting
        return self.extract_text_multilingual(roi, region)
    
    def extract_mathematical_formulas(self, image: np.ndarray, region: Dict = None) -> OCRResult:
        """
        Extract and convert mathematical formulas to LaTeX
        """
        if region:
            x1, y1, x2, y2 = region['bbox']
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        # First extract text using OCR
        text_result = self.extract_text_multilingual(roi, region)
        
        # Try to convert to LaTeX if it looks like a formula
        latex_formula = self._convert_to_latex(text_result.text)
        
        return OCRResult(
            text=latex_formula if latex_formula else text_result.text,
            confidence=text_result.confidence * 0.8,  # Lower confidence for formula conversion
            bbox=text_result.bbox,
            language=text_result.language,
            text_type='formula',
            metadata={
                'original_text': text_result.text,
                'latex': latex_formula,
                'engine': 'formula_converter'
            }
        )
    
    def extract_table_structure(self, image: np.ndarray, region: Dict) -> Dict:
        """
        Extract table structure and content
        """
        x1, y1, x2, y2 = region['bbox']
        table_roi = image[y1:y2, x1:x2]
        
        # Detect table grid
        grid = self._detect_table_grid(table_roi)
        
        # Extract text from each cell
        table_data = []
        for row in grid:
            row_data = []
            for cell_bbox in row:
                cell_roi = table_roi[cell_bbox[1]:cell_bbox[3], cell_bbox[0]:cell_bbox[2]]
                cell_text = self.extract_text_multilingual(cell_roi)
                row_data.append(cell_text.text)
            table_data.append(row_data)
        
        return {
            'type': 'table',
            'data': table_data,
            'structure': grid,
            'bbox': region['bbox']
        }
    
    def _detect_table_grid(self, table_image: np.ndarray) -> List[List[Tuple]]:
        """Detect table cell boundaries"""
        # Simplified table grid detection
        # In practice, this would use more sophisticated algorithms
        
        gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY) if len(table_image.shape) == 3 else table_image
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find line coordinates
        h_lines = self._find_line_coordinates(horizontal_lines, 'horizontal')
        v_lines = self._find_line_coordinates(vertical_lines, 'vertical')
        
        # Create grid from intersections
        grid = []
        for i in range(len(h_lines) - 1):
            row = []
            for j in range(len(v_lines) - 1):
                cell_bbox = (v_lines[j], h_lines[i], v_lines[j+1], h_lines[i+1])
                row.append(cell_bbox)
            grid.append(row)
        
        return grid
    
    def _find_line_coordinates(self, line_image: np.ndarray, direction: str) -> List[int]:
        """Find coordinates of detected lines"""
        # Simplified line coordinate detection
        if direction == 'horizontal':
            line_sums = np.sum(line_image, axis=1)
            threshold = np.max(line_sums) * 0.5
            lines = np.where(line_sums > threshold)[0]
        else:  # vertical
            line_sums = np.sum(line_image, axis=0)
            threshold = np.max(line_sums) * 0.5
            lines = np.where(line_sums > threshold)[0]
        
        # Group nearby lines and take the center
        grouped_lines = []
        if len(lines) > 0:
            current_group = [lines[0]]
            for line in lines[1:]:
                if line - current_group[-1] <= 5:  # Group lines within 5 pixels
                    current_group.append(line)
                else:
                    grouped_lines.append(int(np.mean(current_group)))
                    current_group = [line]
            grouped_lines.append(int(np.mean(current_group)))
        
        return sorted(grouped_lines)
    
    def _convert_to_latex(self, text: str) -> Optional[str]:
        """Convert mathematical text to LaTeX format"""
        if not text or not sympy:
            return None
        
        # Basic mathematical symbol conversion
        conversions = {
            'alpha': r'\alpha',
            'beta': r'\beta',
            'gamma': r'\gamma',
            'delta': r'\delta',
            'epsilon': r'\epsilon',
            'pi': r'\pi',
            'theta': r'\theta',
            'sigma': r'\sigma',
            'infinity': r'\infty',
            '+-': r'\pm',
            '<=': r'\leq',
            '>=': r'\geq',
            '!=': r'\neq',
            'sqrt': r'\sqrt',
            'sum': r'\sum',
            'integral': r'\int',
        }
        
        latex_text = text.lower()
        for word, latex in conversions.items():
            latex_text = latex_text.replace(word, latex)
        
        # Try to parse and validate the LaTeX
        try:
            if parse_latex:
                parsed = parse_latex(latex_text)
                return latex_text
        except:
            pass
        
        return None
    
    def _normalize_bbox(self, bbox: List, region: Dict = None) -> Tuple[int, int, int, int]:
        """Normalize bounding box coordinates"""
        if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
            x1, y1, x2, y2 = bbox
        else:
            # Handle different bbox formats
            try:
                points = np.array(bbox).reshape(-1, 2)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
            except:
                return (0, 0, 0, 0)
        
        # Adjust coordinates if this is a region within a larger image
        if region:
            region_x1, region_y1, _, _ = region['bbox']
            x1 += region_x1
            y1 += region_y1
            x2 += region_x1
            y2 += region_y1
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _ensemble_results(self, results: List[OCRResult]) -> OCRResult:
        """Combine results from multiple OCR engines"""
        if not results:
            return OCRResult("", 0.0, (0, 0, 0, 0), "unknown", "text", {})
        
        # For now, return the result with highest confidence
        # In practice, you might want more sophisticated ensemble methods
        best_result = max(results, key=lambda x: x.confidence)
        
        # Add ensemble metadata
        best_result.metadata['ensemble'] = True
        best_result.metadata['num_engines'] = len(results)
        best_result.metadata['alternative_results'] = [
            {'text': r.text, 'confidence': r.confidence, 'engine': r.metadata.get('engine')}
            for r in results if r != best_result
        ]
        
        return best_result
    
    def process_document_page(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a complete document page with all OCR capabilities
        """
        # Step 1: Layout analysis
        layout = self.analyze_layout(image)
        
        # Step 2: Process each region based on its type
        results = {
            'layout': layout,
            'text_regions': [],
            'table_regions': [],
            'formula_regions': [],
            'form_fields': [],
            'handwriting_regions': []
        }
        
        # Process text regions
        for region in layout.text_regions:
            if region.get('type') == 'handwriting':
                ocr_result = self.extract_handwriting(image, region)
                results['handwriting_regions'].append(ocr_result)
            else:
                ocr_result = self.extract_text_multilingual(image, region)
                results['text_regions'].append(ocr_result)
        
        # Process table regions
        if self.enable_table_recognition:
            for region in layout.table_regions:
                table_result = self.extract_table_structure(image, region)
                results['table_regions'].append(table_result)
        
        # Process formula regions
        if self.enable_formula_recognition:
            for region in layout.formula_regions:
                formula_result = self.extract_mathematical_formulas(image, region)
                results['formula_regions'].append(formula_result)
        
        # Process form fields
        for region in layout.form_fields:
            field_result = self.extract_text_multilingual(image, region)
            field_result.text_type = region['type']
            results['form_fields'].append(field_result)
        
        return results


# Factory function for easy initialization
def create_revolutionary_ocr(config: Dict[str, Any] = None) -> RevolutionaryOCR:
    """
    Factory function to create RevolutionaryOCR with configuration
    
    Args:
        config: Configuration dictionary with OCR settings
    
    Returns:
        Configured RevolutionaryOCR instance
    """
    if config is None:
        config = {}
    
    return RevolutionaryOCR(
        languages=config.get('languages', ['en']),
        enable_table_recognition=config.get('enable_table_recognition', True),
        enable_formula_recognition=config.get('enable_formula_recognition', True),
        enable_handwriting=config.get('enable_handwriting', True),
        confidence_threshold=config.get('confidence_threshold', 0.7)
    )


# Example usage and testing
if __name__ == "__main__":
    # Initialize the revolutionary OCR system
    ocr = create_revolutionary_ocr({
        'languages': ['en', 'zh', 'ja', 'fr', 'de', 'es'],
        'enable_table_recognition': True,
        'enable_formula_recognition': True,
        'enable_handwriting': True,
        'confidence_threshold': 0.7
    })
    
    # Example: Process a document page
    # image = cv2.imread('document_page.jpg')
    # results = ocr.process_document_page(image)
    # print(json.dumps(results, indent=2, default=str))
    
    print("Revolutionary OCR system initialized successfully!")
    print(f"Available OCR engines: {list(ocr.ocr_engines.keys())}")
