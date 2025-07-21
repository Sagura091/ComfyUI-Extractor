"""
OCR Configuration Settings
Configure the Revolutionary OCR system with various parameters
"""

import os
from typing import List, Dict, Any

# OCR Engine Configuration
OCR_CONFIG = {
    # Supported languages (expand as needed)
    'languages': [
        'en',  # English
        'zh',  # Chinese
        'ja',  # Japanese
        'ko',  # Korean
        'fr',  # French
        'de',  # German
        'es',  # Spanish
        'it',  # Italian
        'pt',  # Portuguese
        'ru',  # Russian
        'ar',  # Arabic
        'hi',  # Hindi
    ],
    
    # Feature toggles
    'enable_table_recognition': True,
    'enable_formula_recognition': True,
    'enable_handwriting': True,
    'enable_multilingual': True,
    
    # Quality thresholds
    'confidence_threshold': float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.7')),
    'min_text_length': int(os.getenv('OCR_MIN_TEXT_LENGTH', '3')),
    
    # Performance settings
    'use_gpu': os.getenv('OCR_USE_GPU', 'true').lower() == 'true',
    'batch_size': int(os.getenv('OCR_BATCH_SIZE', '1')),
    'max_image_size': int(os.getenv('OCR_MAX_IMAGE_SIZE', '2048')),
    
    # Advanced settings
    'enable_preprocessing': True,  # Image enhancement before OCR
    'enable_postprocessing': True,  # Text cleanup after OCR
    'enable_ensemble': True,  # Use multiple OCR engines
    
    # Model-specific settings
    'paddle_ocr': {
        'det_model_dir': None,  # Use default models
        'rec_model_dir': None,
        'cls_model_dir': None,
        'use_angle_cls': True,
        'lang': 'en',
        'det_db_thresh': 0.3,
        'det_db_box_thresh': 0.6,
        'det_db_unclip_ratio': 1.5,
        'use_dilation': False,
        'det_db_score_mode': 'fast',
    },
    
    'easy_ocr': {
        'decoder': 'beamsearch',  # or 'greedy'
        'beamWidth': 5,
        'batch_size': 1,
        'workers': 0,
        'allowlist': None,
        'blocklist': None,
        'detail': 1,
        'paragraph': False,
    },
    
    'tesseract': {
        'psm': 6,  # Page segmentation mode
        'oem': 3,  # OCR Engine Mode
        'config': '--tessdata-dir /usr/share/tesseract-ocr/4.00/tessdata',
    }
}

# Language detection configuration
LANGUAGE_DETECTION_CONFIG = {
    'enabled': True,
    'auto_switch_ocr_language': True,
    'confidence_threshold': 0.8,
    'fallback_language': 'en',
}

# Table recognition configuration
TABLE_CONFIG = {
    'min_rows': 2,
    'min_cols': 2,
    'line_thickness_threshold': 2,
    'cell_min_width': 20,
    'cell_min_height': 15,
    'merge_nearby_lines': True,
    'line_merge_threshold': 5,
}

# Formula recognition configuration
FORMULA_CONFIG = {
    'confidence_threshold': 0.6,
    'enable_latex_conversion': True,
    'enable_mathml_output': False,
    'symbol_dictionary': {
        # Mathematical symbols mapping
        'alpha': r'\alpha',
        'beta': r'\beta',
        'gamma': r'\gamma',
        'delta': r'\delta',
        'epsilon': r'\varepsilon',
        'zeta': r'\zeta',
        'eta': r'\eta',
        'theta': r'\theta',
        'iota': r'\iota',
        'kappa': r'\kappa',
        'lambda': r'\lambda',
        'mu': r'\mu',
        'nu': r'\nu',
        'xi': r'\xi',
        'pi': r'\pi',
        'rho': r'\rho',
        'sigma': r'\sigma',
        'tau': r'\tau',
        'upsilon': r'\upsilon',
        'phi': r'\phi',
        'chi': r'\chi',
        'psi': r'\psi',
        'omega': r'\omega',
        'integral': r'\int',
        'sum': r'\sum',
        'product': r'\prod',
        'infinity': r'\infty',
        'partial': r'\partial',
        'nabla': r'\nabla',
        'sqrt': r'\sqrt',
        'pm': r'\pm',
        'mp': r'\mp',
        'leq': r'\leq',
        'geq': r'\geq',
        'neq': r'\neq',
        'approx': r'\approx',
        'equiv': r'\equiv',
        'subset': r'\subset',
        'superset': r'\supset',
        'in': r'\in',
        'notin': r'\notin',
        'union': r'\cup',
        'intersection': r'\cap',
        'forall': r'\forall',
        'exists': r'\exists',
        'therefore': r'\therefore',
        'because': r'\because',
    }
}

# Handwriting recognition configuration
HANDWRITING_CONFIG = {
    'confidence_threshold': 0.5,  # Lower threshold for handwriting
    'enable_word_segmentation': True,
    'enable_line_segmentation': True,
    'preprocessing': {
        'enable_skew_correction': True,
        'enable_noise_removal': True,
        'enable_contrast_enhancement': True,
    }
}

# Image preprocessing configuration
PREPROCESSING_CONFIG = {
    'enable_auto_rotation': True,
    'enable_deskewing': True,
    'enable_denoising': True,
    'enable_contrast_enhancement': True,
    'enable_resolution_enhancement': True,
    'target_dpi': 300,
    'max_resolution': (4096, 4096),
}

# Performance optimization
PERFORMANCE_CONFIG = {
    'enable_caching': True,
    'cache_ttl_seconds': 3600,  # 1 hour
    'enable_parallel_processing': True,
    'max_workers': os.cpu_count(),
    'memory_limit_mb': 4096,
    'enable_model_quantization': False,  # Reduce model size for speed
    'enable_tensorrt_optimization': False,  # NVIDIA TensorRT
}

# Output configuration
OUTPUT_CONFIG = {
    'include_confidence_scores': True,
    'include_bounding_boxes': True,
    'include_word_level_details': True,
    'include_language_detection': True,
    'include_processing_metadata': True,
    'output_formats': ['json', 'text'],  # Available: json, text, xml, csv
}

# Error handling and logging
ERROR_CONFIG = {
    'log_level': os.getenv('OCR_LOG_LEVEL', 'INFO'),
    'enable_error_recovery': True,
    'max_retries': 3,
    'retry_delay_seconds': 1,
    'fallback_to_basic_ocr': True,
}

# Environment-specific overrides
def get_ocr_config() -> Dict[str, Any]:
    """
    Get OCR configuration with environment-specific overrides
    """
    config = OCR_CONFIG.copy()
    
    # Override with environment variables
    if os.getenv('OCR_LANGUAGES'):
        config['languages'] = os.getenv('OCR_LANGUAGES').split(',')
    
    if os.getenv('OCR_ENABLE_TABLES'):
        config['enable_table_recognition'] = os.getenv('OCR_ENABLE_TABLES').lower() == 'true'
    
    if os.getenv('OCR_ENABLE_FORMULAS'):
        config['enable_formula_recognition'] = os.getenv('OCR_ENABLE_FORMULAS').lower() == 'true'
    
    if os.getenv('OCR_ENABLE_HANDWRITING'):
        config['enable_handwriting'] = os.getenv('OCR_ENABLE_HANDWRITING').lower() == 'true'
    
    return config

def get_language_config() -> Dict[str, Any]:
    """Get language detection configuration"""
    return LANGUAGE_DETECTION_CONFIG

def get_table_config() -> Dict[str, Any]:
    """Get table recognition configuration"""
    return TABLE_CONFIG

def get_formula_config() -> Dict[str, Any]:
    """Get formula recognition configuration"""
    return FORMULA_CONFIG

def get_handwriting_config() -> Dict[str, Any]:
    """Get handwriting recognition configuration"""
    return HANDWRITING_CONFIG

def get_preprocessing_config() -> Dict[str, Any]:
    """Get image preprocessing configuration"""
    return PREPROCESSING_CONFIG

def get_performance_config() -> Dict[str, Any]:
    """Get performance optimization configuration"""
    return PERFORMANCE_CONFIG

def get_output_config() -> Dict[str, Any]:
    """Get output configuration"""
    return OUTPUT_CONFIG

def get_error_config() -> Dict[str, Any]:
    """Get error handling configuration"""
    return ERROR_CONFIG

# Validation functions
def validate_config() -> bool:
    """
    Validate OCR configuration
    Returns True if configuration is valid, False otherwise
    """
    config = get_ocr_config()
    
    # Check required fields
    required_fields = ['languages', 'confidence_threshold']
    for field in required_fields:
        if field not in config:
            print(f"Missing required field: {field}")
            return False
    
    # Validate confidence threshold
    if not 0.0 <= config['confidence_threshold'] <= 1.0:
        print("Confidence threshold must be between 0.0 and 1.0")
        return False
    
    # Validate languages
    if not isinstance(config['languages'], list) or not config['languages']:
        print("Languages must be a non-empty list")
        return False
    
    return True

# Export main configuration function
def get_revolutionary_ocr_config() -> Dict[str, Any]:
    """
    Get complete configuration for Revolutionary OCR system
    """
    return {
        'ocr': get_ocr_config(),
        'language_detection': get_language_config(),
        'table_recognition': get_table_config(),
        'formula_recognition': get_formula_config(),
        'handwriting_recognition': get_handwriting_config(),
        'preprocessing': get_preprocessing_config(),
        'performance': get_performance_config(),
        'output': get_output_config(),
        'error_handling': get_error_config(),
    }

# Initialize and validate configuration on import
if __name__ == "__main__":
    if validate_config():
        print("OCR configuration is valid")
        config = get_revolutionary_ocr_config()
        print(f"Configuration loaded with {len(config['ocr']['languages'])} languages")
    else:
        print("OCR configuration validation failed")
