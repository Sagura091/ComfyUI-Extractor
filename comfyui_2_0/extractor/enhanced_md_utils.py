"""
Enhanced Markdown Processing with Revolutionary OCR
Extracts images with full document context, OCR text recognition,
code block analysis, and comprehensive markdown structure analysis.
"""

import markdown
from markdown.extensions import codehilite, tables, toc
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import Extension
import re
import json
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import base64
import requests
from urllib.parse import urljoin, urlparse
import io

from .revolutionary_ocr import create_revolutionary_ocr
from .caption_api import auto_caption
from .settings import OUT_DIR
from .ocr_config import get_revolutionary_ocr_config

# Setup logging
logger = logging.getLogger(__name__)

class ImageExtractorTreeProcessor(Treeprocessor):
    """Custom tree processor to extract images and their context from markdown"""
    
    def __init__(self, md, image_list):
        super().__init__(md)
        self.image_list = image_list
    
    def run(self, root):
        for element in root.iter():
            if element.tag == 'img':
                self.image_list.append({
                    'src': element.get('src', ''),
                    'alt': element.get('alt', ''),
                    'title': element.get('title', ''),
                    'parent_context': self._get_parent_context(element),
                    'element': element
                })
    
    def _get_parent_context(self, element):
        """Get context from parent elements"""
        context = {}
        
        # Find parent heading
        parent = element.getparent()
        while parent is not None:
            if parent.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                context['heading'] = parent.text or ''
                context['heading_level'] = parent.tag
                break
            parent = parent.getparent()
        
        # Find surrounding text
        if element.getparent() is not None:
            siblings = list(element.getparent())
            element_index = siblings.index(element)
            
            # Get previous sibling text
            if element_index > 0:
                prev_element = siblings[element_index - 1]
                if prev_element.text:
                    context['previous_text'] = prev_element.text[:200]
            
            # Get next sibling text  
            if element_index < len(siblings) - 1:
                next_element = siblings[element_index + 1]
                if next_element.text:
                    context['next_text'] = next_element.text[:200]
        
        return context

class ImageExtractorExtension(Extension):
    """Markdown extension to extract images during parsing"""
    
    def __init__(self, image_list, **kwargs):
        self.image_list = image_list
        super().__init__(**kwargs)
    
    def extendMarkdown(self, md):
        processor = ImageExtractorTreeProcessor(md, self.image_list)
        processor.priority = 0  # Run after other processors
        md.treeprocessors.register(processor, 'image_extractor', 0)

class EnhancedMarkdownProcessor:
    """Enhanced Markdown processor with Revolutionary OCR capabilities"""
    
    def __init__(self, ocr_config: Dict[str, Any] = None):
        # Initialize revolutionary OCR
        self.ocr = create_revolutionary_ocr(ocr_config or {
            'languages': ['en'],
            'enable_table_recognition': True,
            'enable_formula_recognition': True,
            'enable_handwriting': True,
            'confidence_threshold': 0.7
        })
        
        self.logger = logging.getLogger(__name__)
    
    def process_markdown_advanced(self, path: Path, push_record_func) -> Dict[str, Any]:
        """
        Process Markdown file with advanced OCR and content analysis
        """
        try:
            # Read markdown content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_id = path.stem
            
            processing_results = {
                'document_id': doc_id,
                'total_images': 0,
                'processed_images': 0,
                'images': [],
                'code_blocks': [],
                'tables': [],
                'formulas': [],
                'document_structure': {},
                'errors': []
            }
            
            # Extract comprehensive document structure
            doc_structure = self._extract_document_structure(content, path)
            processing_results['document_structure'] = doc_structure
            
            # Process images with context
            image_results = self._process_images_with_context(
                content, path, doc_structure, push_record_func
            )
            processing_results['images'] = image_results
            processing_results['total_images'] = len(image_results)
            processing_results['processed_images'] = len([img for img in image_results if 'path' in img])
            
            # Process code blocks
            code_results = self._process_code_blocks(
                content, doc_structure, doc_id, push_record_func
            )
            processing_results['code_blocks'] = code_results
            
            # Process markdown tables
            table_results = self._process_markdown_tables(
                content, doc_structure, doc_id, push_record_func
            )
            processing_results['tables'] = table_results
            
            # Process mathematical formulas (LaTeX/MathJax)
            formula_results = self._process_mathematical_content(
                content, doc_structure, doc_id, push_record_func
            )
            processing_results['formulas'] = formula_results
            
            # Create document overview record
            self._create_document_overview_record(
                doc_structure, processing_results, doc_id, path, push_record_func
            )
            
            return processing_results
            
        except Exception as e:
            error_msg = f"Error processing Markdown {path}: {str(e)}"
            self.logger.error(error_msg)
            return {'errors': [error_msg]}
    
    def _extract_document_structure(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Extract comprehensive document structure from markdown"""
        
        structure = {
            'title': '',
            'headers': [],
            'sections': {},
            'metadata': {},
            'links': [],
            'image_references': [],
            'code_blocks': [],
            'tables': [],
            'lists': [],
            'emphasis': [],
            'toc': [],
            'word_count': 0,
            'line_count': 0,
            'character_count': len(content)
        }
        
        lines = content.split('\n')
        structure['line_count'] = len(lines)
        structure['word_count'] = len(content.split())
        
        # Extract metadata (YAML front matter)
        if content.startswith('---\n'):
            end_idx = content.find('\n---\n', 4)
            if end_idx != -1:
                yaml_content = content[4:end_idx]
                structure['metadata']['front_matter'] = yaml_content
                content_without_metadata = content[end_idx + 5:]
            else:
                content_without_metadata = content
        else:
            content_without_metadata = content
        
        # Extract title (first H1 or from metadata)
        title_match = re.search(r'^#\s+(.+)$', content_without_metadata, re.MULTILINE)
        if title_match:
            structure['title'] = title_match.group(1).strip()
        
        # Extract all headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(header_pattern, content_without_metadata, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            line_num = content_without_metadata[:match.start()].count('\n') + 1
            
            structure['headers'].append({
                'level': level,
                'text': text,
                'line': line_num,
                'anchor': self._create_anchor(text)
            })
        
        # Build table of contents
        structure['toc'] = self._build_toc(structure['headers'])
        
        # Extract sections based on headers
        structure['sections'] = self._extract_sections(content_without_metadata, structure['headers'])
        
        # Extract links
        link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            structure['links'].append({
                'text': match.group(1),
                'url': match.group(2),
                'line': content[:match.start()].count('\n') + 1
            })
        
        # Extract image references
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)(?:\s+"([^"]*)")?\)'
        for match in re.finditer(img_pattern, content):
            structure['image_references'].append({
                'alt': match.group(1),
                'src': match.group(2),
                'title': match.group(3) or '',
                'line': content[:match.start()].count('\n') + 1
            })
        
        # Extract code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            structure['code_blocks'].append({
                'language': match.group(1) or 'text',
                'code': match.group(2),
                'line': content[:match.start()].count('\n') + 1
            })
        
        # Extract tables
        table_pattern = r'(\|.+\|\n)+(\|[\s:|-]+\|\n)+(\|.+\|\n)+'
        for match in re.finditer(table_pattern, content, re.MULTILINE):
            table_content = match.group(0)
            structure['tables'].append({
                'content': table_content,
                'line': content[:match.start()].count('\n') + 1,
                'parsed': self._parse_markdown_table(table_content)
            })
        
        # Extract lists
        list_pattern = r'^(\s*)([-*+]|\d+\.)\s+(.+)$'
        for match in re.finditer(list_pattern, content, re.MULTILINE):
            structure['lists'].append({
                'indent': len(match.group(1)),
                'marker': match.group(2),
                'text': match.group(3),
                'line': content[:match.start()].count('\n') + 1
            })
        
        # Extract emphasis (bold, italic)
        emphasis_patterns = [
            (r'\*\*([^*]+)\*\*', 'bold'),
            (r'__([^_]+)__', 'bold'),
            (r'\*([^*]+)\*', 'italic'),
            (r'_([^_]+)_', 'italic'),
            (r'`([^`]+)`', 'code')
        ]
        
        for pattern, style in emphasis_patterns:
            for match in re.finditer(pattern, content):
                structure['emphasis'].append({
                    'style': style,
                    'text': match.group(1),
                    'line': content[:match.start()].count('\n') + 1
                })
        
        return structure
    
    def _process_images_with_context(self, content: str, file_path: Path, 
                                   doc_structure: Dict, push_record_func) -> List[Dict[str, Any]]:
        """Process images with comprehensive context analysis"""
        
        images = []
        doc_id = file_path.stem
        out_dir = OUT_DIR / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image reference
        for img_idx, img_ref in enumerate(doc_structure['image_references']):
            try:
                img_result = self._process_single_image(
                    img_ref, img_idx, content, file_path, doc_structure, 
                    out_dir, doc_id, push_record_func
                )
                if img_result:
                    images.append(img_result)
                    
            except Exception as e:
                error_msg = f"Failed to process image {img_idx}: {e}"
                self.logger.warning(error_msg)
                images.append({'error': error_msg, 'index': img_idx})
        
        return images
    
    def _process_single_image(self, img_ref: Dict, img_idx: int, content: str, 
                            file_path: Path, doc_structure: Dict, out_dir: Path, 
                            doc_id: str, push_record_func) -> Optional[Dict[str, Any]]:
        """Process a single image with comprehensive context"""
        
        # Download/copy image
        img_path = self._get_image_file(img_ref['src'], file_path, out_dir, img_idx)
        if not img_path or not img_path.exists():
            return None
        
        # Get comprehensive context
        context_info = self._get_comprehensive_image_context(
            img_ref, content, doc_structure, img_idx
        )
        
        # Generate caption using multiple methods
        caption = self._get_comprehensive_caption(
            img_ref, img_path, context_info, img_idx
        )
        
        # Perform OCR on the image
        img_ocr_results = self._ocr_analyze_image(img_path)
        if img_ocr_results:
            context_info['image_ocr_text'] = img_ocr_results.get('text', '')
            context_info['image_contains_text'] = bool(img_ocr_results.get('text', '').strip())
            context_info['image_has_tables'] = img_ocr_results.get('has_tables', False)
            context_info['image_has_formulas'] = img_ocr_results.get('has_formulas', False)
        
        # Create enhanced description
        enhanced_description = self._create_enhanced_description(caption, context_info)
        
        # Get full page context where image appears
        page_context = self._get_full_page_context(img_ref, content, doc_structure)
        
        # Push to record system
        push_record_func(
            doc_id,
            f"img{img_idx}",
            enhanced_description,
            img_path,
            raw_text=json.dumps({
                'context': context_info,
                'page_context': page_context,
                'ocr_results': img_ocr_results
            }, indent=2)
        )
        
        return {
            'path': str(img_path),
            'caption': enhanced_description,
            'context': context_info,
            'page_context': page_context,
            'index': img_idx,
            'original_ref': img_ref
        }
    
    def _get_image_file(self, src: str, file_path: Path, out_dir: Path, img_idx: int) -> Optional[Path]:
        """Download or copy image file"""
        
        try:
            img_extension = Path(src).suffix or '.png'
            img_path = out_dir / f"img{img_idx}{img_extension}"
            
            if src.startswith(('http://', 'https://')):
                # Download from URL
                response = requests.get(src, timeout=10)
                response.raise_for_status()
                
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                    
            elif src.startswith('data:'):
                # Handle data URLs
                if 'base64,' in src:
                    header, data = src.split('base64,', 1)
                    img_data = base64.b64decode(data)
                    
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                else:
                    return None
                    
            else:
                # Local file - resolve relative to markdown file
                source_path = file_path.parent / src
                if source_path.exists():
                    # Copy to output directory
                    import shutil
                    shutil.copy2(source_path, img_path)
                else:
                    self.logger.warning(f"Image file not found: {source_path}")
                    return None
            
            return img_path
            
        except Exception as e:
            self.logger.warning(f"Failed to get image {src}: {e}")
            return None
    
    def _get_comprehensive_image_context(self, img_ref: Dict, content: str, 
                                       doc_structure: Dict, img_idx: int) -> Dict[str, Any]:
        """Get comprehensive context for an image"""
        
        context = {
            'alt_text': img_ref['alt'],
            'title': img_ref['title'],
            'src': img_ref['src'],
            'line_number': img_ref['line'],
            'document_title': doc_structure.get('title', ''),
            'section_context': {},
            'surrounding_text': {},
            'document_metadata': doc_structure.get('metadata', {}),
            'related_headers': [],
            'nearby_content': []
        }
        
        # Find the section containing this image
        section_info = self._find_containing_section(img_ref['line'], doc_structure)
        if section_info:
            context['section_context'] = section_info
        
        # Get surrounding text
        lines = content.split('\n')
        img_line = img_ref['line'] - 1  # Convert to 0-based index
        
        # Get surrounding lines
        start_line = max(0, img_line - 5)
        end_line = min(len(lines), img_line + 6)
        
        context['surrounding_text'] = {
            'before': '\n'.join(lines[start_line:img_line]),
            'after': '\n'.join(lines[img_line + 1:end_line]),
            'paragraph': self._get_containing_paragraph(lines, img_line)
        }
        
        # Find related headers (nearest headers before and after)
        context['related_headers'] = self._find_related_headers(img_ref['line'], doc_structure['headers'])
        
        # Get nearby content (lists, code blocks, tables near the image)
        context['nearby_content'] = self._get_nearby_content(img_ref['line'], doc_structure)
        
        return context
    
    def _get_comprehensive_caption(self, img_ref: Dict, img_path: Path, 
                                 context_info: Dict, img_idx: int) -> str:
        """Generate comprehensive caption using multiple methods"""
        
        captions = []
        
        # Method 1: Use alt text if meaningful
        if img_ref['alt'] and len(img_ref['alt'].strip()) > 3:
            captions.append(('alt_text', img_ref['alt'], 0.9))
        
        # Method 2: Use title if available
        if img_ref['title'] and len(img_ref['title'].strip()) > 3:
            captions.append(('title', img_ref['title'], 0.85))
        
        # Method 3: AI-generated caption
        try:
            ai_caption = auto_caption(str(img_path))
            if ai_caption and len(ai_caption.strip()) > 5:
                captions.append(('ai_caption', ai_caption, 0.8))
        except Exception as e:
            self.logger.warning(f"AI captioning failed: {e}")
        
        # Method 4: OCR on image
        try:
            img_array = np.array(Image.open(img_path))
            ocr_result = self.ocr.extract_text_multilingual(img_array)
            if ocr_result.confidence > 0.7 and len(ocr_result.text.strip()) > 5:
                captions.append(('ocr_text', f"Image containing text: {ocr_result.text[:100]}", 0.7))
        except Exception as e:
            self.logger.warning(f"Image OCR failed: {e}")
        
        # Method 5: Context-based caption
        section_title = context_info.get('section_context', {}).get('title', '')
        doc_title = context_info.get('document_title', '')
        
        context_parts = []
        if section_title:
            context_parts.append(f"from section '{section_title}'")
        if doc_title:
            context_parts.append(f"in document '{doc_title}'")
        
        if context_parts:
            context_caption = f"Image {' '.join(context_parts)}"
            captions.append(('context', context_caption, 0.6))
        
        # Method 6: Filename-based caption
        filename = Path(img_ref['src']).stem
        if filename and filename.lower() not in ['image', 'img', 'picture', 'photo']:
            filename_caption = f"Image: {filename.replace('_', ' ').replace('-', ' ')}"
            captions.append(('filename', filename_caption, 0.5))
        
        # Select best caption or combine
        if captions:
            captions.sort(key=lambda x: x[2], reverse=True)
            return captions[0][1]
        else:
            return f"Image {img_idx + 1} from markdown document"
    
    def _get_full_page_context(self, img_ref: Dict, content: str, doc_structure: Dict) -> Dict[str, Any]:
        """Get the full context of the page/section where the image appears"""
        
        # Find the section containing this image
        section_info = self._find_containing_section(img_ref['line'], doc_structure)
        
        if section_info:
            section_content = section_info.get('content', '')
            
            return {
                'section_title': section_info.get('title', ''),
                'section_level': section_info.get('level', 0),
                'section_content': section_content[:2000],  # First 2000 chars
                'section_word_count': len(section_content.split()),
                'section_line_range': section_info.get('line_range', []),
                'document_title': doc_structure.get('title', ''),
                'document_summary': self._create_document_summary(doc_structure)
            }
        else:
            # Return full document context if no specific section found
            return {
                'section_title': 'Document Root',
                'section_content': content[:2000],
                'document_title': doc_structure.get('title', ''),
                'document_summary': self._create_document_summary(doc_structure)
            }
    
    def _process_code_blocks(self, content: str, doc_structure: Dict, 
                           doc_id: str, push_record_func) -> List[Dict[str, Any]]:
        """Process code blocks with context"""
        
        code_results = []
        
        for code_idx, code_block in enumerate(doc_structure['code_blocks']):
            try:
                # Find containing section
                section_info = self._find_containing_section(code_block['line'], doc_structure)
                
                # Create description
                description = f"Code block ({code_block['language']}) from "
                if section_info:
                    description += f"section '{section_info['title']}'"
                else:
                    description += "document"
                
                # Add code preview
                code_preview = code_block['code'][:100]
                description += f" | Preview: {code_preview}"
                
                context = {
                    'language': code_block['language'],
                    'line_number': code_block['line'],
                    'section_context': section_info,
                    'code_length': len(code_block['code']),
                    'line_count': code_block['code'].count('\n') + 1
                }
                
                push_record_func(
                    doc_id,
                    f"code{code_idx}",
                    description,
                    None,
                    raw_text=code_block['code']
                )
                
                code_results.append({
                    'description': description,
                    'context': context,
                    'code': code_block['code'],
                    'index': code_idx
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process code block {code_idx}: {e}")
        
        return code_results
    
    def _process_markdown_tables(self, content: str, doc_structure: Dict, 
                               doc_id: str, push_record_func) -> List[Dict[str, Any]]:
        """Process markdown tables with context"""
        
        table_results = []
        
        for table_idx, table_info in enumerate(doc_structure['tables']):
            try:
                parsed_table = table_info['parsed']
                
                # Find containing section
                section_info = self._find_containing_section(table_info['line'], doc_structure)
                
                # Create description
                description = f"Table ({len(parsed_table)} rows × {len(parsed_table[0]) if parsed_table else 0} columns)"
                if section_info:
                    description += f" from section '{section_info['title']}'"
                
                # Add header info
                if parsed_table and len(parsed_table) > 0:
                    headers = parsed_table[0][:3]  # First 3 headers
                    description += f" | Headers: {', '.join(headers)}"
                    if len(parsed_table[0]) > 3:
                        description += "..."
                
                context = {
                    'line_number': table_info['line'],
                    'section_context': section_info,
                    'row_count': len(parsed_table),
                    'column_count': len(parsed_table[0]) if parsed_table else 0,
                    'headers': parsed_table[0] if parsed_table else []
                }
                
                push_record_func(
                    doc_id,
                    f"table{table_idx}",
                    description,
                    None,
                    raw_text=json.dumps(parsed_table, indent=2)
                )
                
                table_results.append({
                    'description': description,
                    'context': context,
                    'data': parsed_table,
                    'index': table_idx
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process table {table_idx}: {e}")
        
        return table_results
    
    def _process_mathematical_content(self, content: str, doc_structure: Dict, 
                                    doc_id: str, push_record_func) -> List[Dict[str, Any]]:
        """Process mathematical formulas (LaTeX/MathJax)"""
        
        formula_results = []
        formula_idx = 0
        
        # Patterns for mathematical content
        math_patterns = [
            (r'\$\$([^$]+)\$\$', 'display_math'),  # Display math
            (r'\$([^$]+)\$', 'inline_math'),       # Inline math
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'equation'),  # LaTeX equations
            (r'\\begin\{align\}(.*?)\\end\{align\}', 'align'),          # LaTeX align
            (r'\\begin\{matrix\}(.*?)\\end\{matrix\}', 'matrix'),       # LaTeX matrix
        ]
        
        for pattern, math_type in math_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                try:
                    formula_content = match.group(1).strip()
                    line_number = content[:match.start()].count('\n') + 1
                    
                    # Find containing section
                    section_info = self._find_containing_section(line_number, doc_structure)
                    
                    # Create description
                    description = f"Mathematical formula ({math_type})"
                    if section_info:
                        description += f" from section '{section_info['title']}'"
                    
                    # Add formula preview
                    formula_preview = formula_content[:100]
                    description += f" | Formula: {formula_preview}"
                    
                    context = {
                        'type': math_type,
                        'line_number': line_number,
                        'section_context': section_info,
                        'formula_length': len(formula_content)
                    }
                    
                    push_record_func(
                        doc_id,
                        f"formula{formula_idx}",
                        description,
                        None,
                        raw_text=formula_content
                    )
                    
                    formula_results.append({
                        'description': description,
                        'context': context,
                        'formula': formula_content,
                        'type': math_type,
                        'index': formula_idx
                    })
                    
                    formula_idx += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process formula {formula_idx}: {e}")
        
        return formula_results
    
    def _create_document_overview_record(self, doc_structure: Dict, processing_results: Dict, 
                                       doc_id: str, file_path: Path, push_record_func):
        """Create a comprehensive document overview record"""
        
        overview = {
            'title': doc_structure.get('title', file_path.stem),
            'structure': {
                'headers': len(doc_structure['headers']),
                'sections': len(doc_structure['sections']),
                'images': processing_results.get('processed_images', 0),
                'code_blocks': len(processing_results.get('code_blocks', [])),
                'tables': len(processing_results.get('tables', [])),
                'formulas': len(processing_results.get('formulas', [])),
                'links': len(doc_structure.get('links', [])),
                'lists': len(doc_structure.get('lists', []))
            },
            'statistics': {
                'word_count': doc_structure.get('word_count', 0),
                'line_count': doc_structure.get('line_count', 0),
                'character_count': doc_structure.get('character_count', 0)
            },
            'table_of_contents': doc_structure.get('toc', []),
            'metadata': doc_structure.get('metadata', {}),
            'summary': self._create_document_summary(doc_structure)
        }
        
        description = f"Markdown document: {overview['title']} | "
        description += f"{overview['statistics']['word_count']} words, "
        description += f"{overview['structure']['headers']} headers, "
        description += f"{overview['structure']['images']} images"
        
        push_record_func(
            doc_id,
            "document_overview",
            description,
            None,
            raw_text=json.dumps(overview, indent=2)
        )
    
    # Helper methods
    def _create_anchor(self, text: str) -> str:
        """Create URL anchor from header text"""
        return re.sub(r'[^\w\s-]', '', text).strip().lower().replace(' ', '-')
    
    def _build_toc(self, headers: List[Dict]) -> List[Dict]:
        """Build table of contents from headers"""
        toc = []
        for header in headers:
            toc.append({
                'level': header['level'],
                'text': header['text'],
                'anchor': header['anchor'],
                'line': header['line']
            })
        return toc
    
    def _extract_sections(self, content: str, headers: List[Dict]) -> Dict[str, Any]:
        """Extract sections based on headers"""
        sections = {}
        lines = content.split('\n')
        
        for i, header in enumerate(headers):
            section_start = header['line'] - 1  # Convert to 0-based
            
            # Find section end (next header of same or higher level)
            section_end = len(lines)
            for j in range(i + 1, len(headers)):
                if headers[j]['level'] <= header['level']:
                    section_end = headers[j]['line'] - 1
                    break
            
            section_content = '\n'.join(lines[section_start:section_end])
            
            sections[header['anchor']] = {
                'title': header['text'],
                'level': header['level'],
                'content': section_content,
                'line_range': [section_start + 1, section_end],
                'word_count': len(section_content.split())
            }
        
        return sections
    
    def _parse_markdown_table(self, table_content: str) -> List[List[str]]:
        """Parse markdown table into 2D array"""
        lines = table_content.strip().split('\n')
        table_data = []
        
        for line in lines:
            if '|' in line and not re.match(r'^\s*\|[\s:|-]+\|\s*$', line):
                # Split by | and clean up
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty cells at start/end
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]
                
                if cells:
                    table_data.append(cells)
        
        return table_data
    
    def _find_containing_section(self, line_number: int, doc_structure: Dict) -> Optional[Dict]:
        """Find the section containing the given line number"""
        for section_info in doc_structure['sections'].values():
            line_range = section_info.get('line_range', [])
            if len(line_range) >= 2 and line_range[0] <= line_number <= line_range[1]:
                return section_info
        return None
    
    def _find_related_headers(self, line_number: int, headers: List[Dict]) -> List[Dict]:
        """Find headers related to the given line number"""
        related = []
        
        # Find the header immediately before this line
        prev_header = None
        for header in headers:
            if header['line'] < line_number:
                prev_header = header
            else:
                break
        
        if prev_header:
            related.append(prev_header)
        
        # Find the next header
        for header in headers:
            if header['line'] > line_number:
                related.append(header)
                break
        
        return related
    
    def _get_nearby_content(self, line_number: int, doc_structure: Dict) -> List[Dict]:
        """Get content elements near the given line number"""
        nearby = []
        search_range = 10  # Lines before and after
        
        # Check code blocks
        for code_block in doc_structure.get('code_blocks', []):
            if abs(code_block['line'] - line_number) <= search_range:
                nearby.append({
                    'type': 'code_block',
                    'language': code_block['language'],
                    'line': code_block['line']
                })
        
        # Check tables
        for table in doc_structure.get('tables', []):
            if abs(table['line'] - line_number) <= search_range:
                nearby.append({
                    'type': 'table',
                    'line': table['line']
                })
        
        # Check lists
        for list_item in doc_structure.get('lists', []):
            if abs(list_item['line'] - line_number) <= search_range:
                nearby.append({
                    'type': 'list',
                    'marker': list_item['marker'],
                    'line': list_item['line']
                })
        
        return nearby
    
    def _get_containing_paragraph(self, lines: List[str], img_line: int) -> str:
        """Get the paragraph containing the image"""
        
        # Find paragraph boundaries (empty lines)
        start = img_line
        while start > 0 and lines[start - 1].strip():
            start -= 1
        
        end = img_line
        while end < len(lines) - 1 and lines[end + 1].strip():
            end += 1
        
        return '\n'.join(lines[start:end + 1])
    
    def _create_document_summary(self, doc_structure: Dict) -> str:
        """Create a brief document summary"""
        summary_parts = []
        
        if doc_structure.get('title'):
            summary_parts.append(f"Document: {doc_structure['title']}")
        
        summary_parts.append(f"{doc_structure.get('word_count', 0)} words")
        
        if doc_structure.get('headers'):
            summary_parts.append(f"{len(doc_structure['headers'])} sections")
        
        if doc_structure.get('image_references'):
            summary_parts.append(f"{len(doc_structure['image_references'])} images")
        
        return ' | '.join(summary_parts)
    
    def _ocr_analyze_image(self, img_path: Path) -> Optional[Dict[str, Any]]:
        """Perform OCR analysis on a single image"""
        try:
            img_array = np.array(Image.open(img_path))
            results = self.ocr.process_document_page(img_array)
            
            all_text = []
            for region in results.get('text_regions', []):
                if region.confidence > 0.5:
                    all_text.append(region.text)
            
            return {
                'text': ' '.join(all_text),
                'has_tables': len(results.get('table_regions', [])) > 0,
                'has_formulas': len(results.get('formula_regions', [])) > 0,
                'confidence': max([r.confidence for r in results.get('text_regions', [])], default=0.0)
            }
        except Exception as e:
            self.logger.warning(f"OCR analysis failed for {img_path}: {e}")
            return None
    
    def _create_enhanced_description(self, base_caption: str, context_info: Dict) -> str:
        """Create enhanced description combining caption and context"""
        
        description_parts = [base_caption]
        
        # Add document context
        if context_info.get('document_title'):
            description_parts.append(f"From document: '{context_info['document_title']}'")
        
        # Add section context
        section_context = context_info.get('section_context', {})
        if section_context.get('title'):
            description_parts.append(f"Section: '{section_context['title']}'")
        
        # Add OCR text if present
        if context_info.get('image_contains_text') and context_info.get('image_ocr_text'):
            text_preview = context_info['image_ocr_text'][:100]
            description_parts.append(f"Contains text: {text_preview}")
        
        # Add special content flags
        special_content = []
        if context_info.get('image_has_tables'):
            special_content.append('tables')
        if context_info.get('image_has_formulas'):
            special_content.append('formulas')
        
        if special_content:
            description_parts.append(f"Contains: {', '.join(special_content)}")
        
        return ' | '.join(description_parts)


# Backward compatibility function
def process_md(path: Path, push_record):
    """
    Enhanced Markdown processing function with Revolutionary OCR
    """
    try:
        # Initialize enhanced processor
        config = get_revolutionary_ocr_config()
        processor = EnhancedMarkdownProcessor(config['ocr'])
        
        # Process with advanced capabilities
        results = processor.process_markdown_advanced(path, push_record)
        
        # Log processing summary
        logger.info(f"Processed Markdown {path.name}: "
                   f"{results.get('processed_images', 0)}/{results.get('total_images', 0)} images, "
                   f"{len(results.get('code_blocks', []))} code blocks, "
                   f"{len(results.get('tables', []))} tables, "
                   f"{len(results.get('formulas', []))} formulas")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process Markdown {path}: {e}")
        # Fallback to basic processing
        return _process_md_basic(path, push_record)


def _process_md_basic(path: Path, push_record):
    """Fallback to basic Markdown processing if advanced processing fails"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc_id = path.stem
        out_dir = OUT_DIR / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic image extraction
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)(?:\s+"([^"]*)")?\)'
        img_idx = 0
        
        for match in re.finditer(img_pattern, content):
            alt_text = match.group(1)
            src = match.group(2)
            title = match.group(3) or ''
            
            try:
                # Simple image copying for local files
                if not src.startswith(('http://', 'https://', 'data:')):
                    source_path = path.parent / src
                    if source_path.exists():
                        img_path = out_dir / f"img{img_idx}.png"
                        import shutil
                        shutil.copy2(source_path, img_path)
                        
                        caption = alt_text or title or auto_caption(str(img_path))
                        push_record(doc_id, f"img{img_idx}", caption, img_path, raw_text=content[:1000])
                        img_idx += 1
            except Exception as e:
                logger.warning(f"Failed to process image {img_idx}: {e}")
                
    except Exception as e:
        logger.error(f"Basic Markdown processing also failed: {e}")
        raise
