#!/usr/bin/env python3
"""
Gemini Text to Knowledge Processor
Processes text files using Google's Gemini AI and saves structured markdown output.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

def load_environment() -> str:
    """Load environment variables and return API key."""
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file")
        sys.exit(1)
    return api_key

def get_available_models() -> List[str]:
    """Get list of known Gemini models."""
    return [
        'gemini-pro',
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gemini-2.0-flash-exp',
        'gemini-2.0-pro-exp',
        'gemini-2.0-pro-exp-02-05',
        'gemini-2.0-flash-exp-02-05'
    ]

def find_best_model_match(target_model: str, available_models: List[str]) -> str:
    """Find the best matching model from available models, handling version suffixes."""
    if not available_models:
        return target_model
    
    # Exact match first
    if target_model in available_models:
        return target_model
    
    # Try to find partial matches (handles version suffixes)
    target_base = target_model.split('-')[0]  # Get base name like 'gemini'
    
    for model in available_models:
        if model.startswith(target_base):
            # Check if it's a good match (same major version)
            target_parts = target_model.split('-')
            model_parts = model.split('-')
            
            # Match major version (e.g., gemini-2.0-pro-exp matches gemini-2.0-pro-exp-02-05)
            if len(target_parts) >= 3 and len(model_parts) >= 3:
                if target_parts[1] == model_parts[1] and target_parts[2] == model_parts[2]:
                    return model
    
    # Fallback: find any model that starts with the same base
    for model in available_models:
        if model.startswith(target_base):
            return model
    
    return target_model

def get_safety_config():
    """Get safety configuration for Gemini AI."""
    return [
        {
            "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
            "threshold": HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "threshold": HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "threshold": HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "threshold": HarmBlockThreshold.BLOCK_NONE,
        },
    ]

def preprocess_content_for_safety(text_content: str) -> str:
    """Preprocess content to reduce likelihood of content policy violations."""
    # Remove or replace potentially problematic content
    processed = text_content
    
    # Remove censored profanity completely
    processed = processed.replace('[ __ ]', '')
    processed = processed.replace('[__]', '')
    
    # Remove common profanity patterns (case insensitive)
    import re
    
    # Common profanity words to remove (add more as needed)
    profanity_patterns = [
        r'\b\w*f\w*k\w*\b',  # f-word variations
        r'\b\w*s\w*t\w*\b',  # s-word variations
        r'\b\w*a\w*s\w*s\w*\b',  # a-word variations
        r'\b\w*b\w*t\w*c\w*h\w*\b',  # b-word variations
        r'\b\w*d\w*m\w*n\w*\b',  # d-word variations
        r'\b\w*h\w*e\w*l\w*l\w*\b',  # h-word variations
    ]
    
    for pattern in profanity_patterns:
        processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
    
    # Clean up extra whitespace and empty lines
    processed = re.sub(r'\n\s*\n', '\n', processed)  # Remove empty lines
    processed = re.sub(r' +', ' ', processed)  # Remove extra spaces
    
    # Add a note about content safety
    safety_note = "\n\nNote: This transcript has been preprocessed for content safety. Profanity and potentially problematic content has been removed."
    processed += safety_note
    
    return processed

def query_api_models(api_key: str) -> List[str]:
    """Query the Gemini API to get actually available models."""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        
        # Filter for generative models
        generative_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                generative_models.append(model.name)
        
        return generative_models
    except Exception as e:
        print(f"Warning: Could not query API for models: {e}")
        print("Falling back to known model list")
        return get_available_models()

def setup_gemini(api_key: str) -> genai.GenerativeModel:
    """Initialize Gemini AI model."""
    genai.configure(api_key=api_key)
    
    # Query API for actually available models
    print("Querying Gemini API for available models...")
    api_models = query_api_models(api_key)
    
    # Check if user wants to see detailed model list
    show_model_list = os.getenv('SHOW_MODEL_LIST', 'false').lower() == 'true'
    
    # Command line override for --show-models
    if '--show-models' in sys.argv:
        show_model_list = True
    
    if api_models:
        if show_model_list:
            print("API accessible models:")
            for model in api_models:
                print(f"  - {model}")
        else:
            print(f"API accessible models: {len(api_models)} models found")
        print()
    else:
        print("No models accessible via API")
    
    # Get model from environment variable or use default
    model_name = os.getenv('GEMINI_MODEL', 'gemini-pro')
    
    # Find the best matching model from API models
    if api_models:
        best_api_match = find_best_model_match(model_name, api_models)
        if best_api_match != model_name:
            print(f"Model '{model_name}' not found, using best match: '{best_api_match}'")
            print(f"   (This handles version suffixes like -02-05)")
            model_name = best_api_match
            print(f"Selected model '{model_name}' is accessible via API")
        else:
            print(f"Selected model '{model_name}' is accessible via API")
    else:
        # Fallback to known models if API query failed
        best_known_match = find_best_model_match(model_name, get_available_models())
        if best_known_match != model_name:
            print(f"Model '{model_name}' not found, using best match: '{best_known_match}'")
            print(f"   (This handles version suffixes like -02-05)")
            model_name = best_known_match
        print(f"Warning: '{model_name}' not accessible via API. Trying anyway...")
    
    print(f"Using Gemini model: {model_name}")
    
    # Create model with safety configuration
    model = genai.GenerativeModel(model_name)
    
    # Apply safety configuration to allow more content types
    safety_config = get_safety_config()
    print("Safety configuration applied: All harm categories set to BLOCK_NONE")
    
    return model

def read_prompt() -> str:
    """Read the Gemini prompt from file."""
    try:
        with open('gemini_prompt.txt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("Error: gemini_prompt.txt not found")
        sys.exit(1)

def get_processed_files(filename: str) -> Tuple[List[str], List[str]]:
    """Read already processed and failed files."""
    complete_files = []
    failed_files = []
    
    if os.path.exists('gemini_complete.txt'):
        with open('gemini_complete.txt', 'r', encoding='utf-8') as f:
            complete_files = [line.strip() for line in f.readlines()]
    
    if os.path.exists('gemini_failed.txt'):
        with open('gemini_failed.txt', 'r', encoding='utf-8') as f:
            failed_files = [line.strip() for line in f.readlines()]
    
    return complete_files, failed_files

def process_text_with_gemini(model: genai.GenerativeModel, prompt: str, text_content: str) -> str:
    """Process text content with Gemini AI."""
    try:
        # Create a more focused prompt to avoid policy violations
        safe_prompt = f"{prompt}\n\nIMPORTANT: Focus on factual analysis and avoid any content that could violate policies. Be professional and objective.\n\nTranscript:\n{text_content}"
        
        # Apply safety configuration to the generation
        safety_config = get_safety_config()
        response = model.generate_content(
            safe_prompt,
            safety_settings=safety_config
        )
        
        if response.text:
            return response.text
        else:
            return "Error: No response generated"
            
    except Exception as e:
        error_msg = str(e)
        if "violation" in error_msg.lower():
            return "Error: Content policy violation - The content may contain language or topics that violate Gemini's safety policies. Consider:\n- Removing profanity or offensive language\n- Focusing on factual analysis rather than opinions\n- Avoiding specific financial advice\n- Using more professional language"
        elif "501" in error_msg.lower():
            return "Error: Service unavailable (501)"
        elif "quota" in error_msg.lower():
            return "Error: API quota exceeded - You may have reached your daily limit"
        elif "rate" in error_msg.lower():
            return "Error: Rate limit exceeded - Please wait before trying again"
        else:
            return f"Error: {error_msg}"

def save_output(content: str, filename: str) -> bool:
    """Save processed content to output directory."""
    try:
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{Path(filename).stem}.md"
        
        # Ensure content is in markdown format with bold sections
        formatted_content = format_markdown_sections(content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def format_markdown_sections(content: str) -> str:
    """Format content with bold section headers."""
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('-') and ':' in line:
            # Bold the section headers
            section_name = line.split(':')[0].strip('- ')
            formatted_lines.append(f"**{section_name}:**")
            if ':' in line:
                formatted_lines.append(line.split(':', 1)[1].strip())
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def update_tracking_files(filename: str, success: bool):
    """Update tracking files based on processing result."""
    if success:
        with open('gemini_complete.txt', 'a', encoding='utf-8') as f:
            f.write(f"{filename}\n")
    else:
        with open('gemini_failed.txt', 'a', encoding='utf-8') as f:
            f.write(f"{filename}\n")

def main():
    """Main processing function."""
    # Check for help argument
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Gemini Text to Knowledge Processor")
        print("=" * 40)
        print("Usage: python gemini_text_to_knowledge.py [options]")
        print()
        print("Options:")
        print("  -h, --help, help    Show this help message")
        print("  --models            Show known Gemini models")
        print("  --api-models        Query API for accessible models (requires .env)")
        print("  --all-models        Show both known and API accessible models")
        print("  --test-api          Test API connection and show model info")
        print("  --test-matching    Test model name matching logic")
        print("  --test-safety      Test content preprocessing for safety")
        print("  --show-models      Force show detailed model list")
        print("  --test-safety-config Test safety configuration settings")
        print()
        print("Known Gemini models (API will be queried for actual availability):")
        for model in get_available_models():
            print(f"  - {model}")
        print()
        print("Note: The script will query your API key to show which models are actually accessible.")
        print()
        print("Environment variables:")
        print("  GOOGLE_API_KEY      Your Google AI API key (required)")
        print("  GEMINI_MODEL        Gemini model to use (default: gemini-pro)")
        print("  OUTPUT_DIR          Output directory (default: output)")
        print("  INPUT_DIR           Input directory (default: input)")
        print("  SHOW_MODEL_LIST     Show detailed model list (default: false)")
        return
    
    # Check for models argument
    if len(sys.argv) > 1 and sys.argv[1] == '--models':
        print("Known Gemini models (API will be queried for actual availability):")
        for model in get_available_models():
            print(f"  - {model}")
        print()
        print("Note: Run the script with your API key to see which models are actually accessible.")
        return
    
    # Check for api-models argument
    if len(sys.argv) > 1 and sys.argv[1] == '--api-models':
        try:
            api_key = load_environment()
            print("Querying Gemini API for accessible models...")
            api_models = query_api_models(api_key)
            if api_models:
                print(f"Models accessible with your API key:")
                for model in api_models:
                    print(f"  - {model}")
            else:
                print("No models accessible with your API key")
        except Exception as e:
            print(f"Error querying API: {e}")
        return
    
    # Check for all-models argument
    if len(sys.argv) > 1 and sys.argv[1] == '--all-models':
        print("Known Gemini models:")
        for model in get_available_models():
            print(f"  - {model}")
        print()
        try:
            api_key = load_environment()
            print("Querying Gemini API for accessible models...")
            api_models = query_api_models(api_key)
            if api_models:
                print(f"Models accessible with your API key:")
                for model in api_models:
                    print(f"  - {model}")
            else:
                print("No models accessible with your API key")
        except Exception as e:
            print(f"Error querying API: {e}")
        return
    
    # Check for test-api argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-api':
        try:
            api_key = load_environment()
            print("Testing Gemini API connection...")
            print(f"API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if len(api_key) > 8 else '***'}")
            print()
            
            # Test API connection
            api_models = query_api_models(api_key)
            if api_models:
                print(f"API connection successful!")
                print(f"Models accessible: {len(api_models)}")
                print("Available models:")
                for model in api_models:
                    print(f"    - {model}")
                
                # Test model creation
                try:
                    model = genai.GenerativeModel('gemini-pro')
                    print("Model creation test successful")
                except Exception as e:
                    print(f"Model creation test failed: {e}")
            else:
                print("API connection failed - no models accessible")
        except Exception as e:
            print(f"API test failed: {e}")
        return
    
    # Check for test-matching argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-matching':
        print("Testing model name matching logic...")
        print()
        
        # Test cases
        test_cases = [
            'gemini-2.0-pro-exp',
            'gemini-2.0-pro-exp-02-05',
            'gemini-1.5-pro',
            'gemini-pro',
            'unknown-model'
        ]
        
        known_models = get_available_models()
        print("Known models:")
        for model in known_models:
            print(f"  - {model}")
        print()
        
        print("Matching results:")
        for test_model in test_cases:
            best_match = find_best_model_match(test_model, known_models)
            if best_match == test_model:
                print(f"'{test_model}' -> exact match")
            else:
                print(f"'{test_model}' -> best match: '{best_match}'")
        return
    
    # Check for test-safety argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-safety':
        print("Testing content safety preprocessing...")
        print()
        
        # Test content with potential violations
        test_content = """This is a test transcript with some [ __ ] content.
It contains trading advice and casual language.
The market is going to [__] crash tomorrow!
Buy this stock now for massive profits!"""
        
        print("Original content:")
        print("-" * 40)
        print(test_content)
        print("-" * 40)
        print()
        
        print("Preprocessed content:")
        print("-" * 40)
        safe_content = preprocess_content_for_safety(test_content)
        print(safe_content)
        print("-" * 40)
        print()
        
        print("Safety improvements made:")
        print("Removed censored profanity completely")
        print("Removed common profanity patterns")
        print("Cleaned up extra whitespace")
        print("Added content safety note")
        print("Maintained readability while improving safety")
        return
    
    # Check for test-safety-config argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-safety-config':
        print("Testing Gemini AI safety configuration...")
        print()
        
        try:
            safety_config = get_safety_config()
            print("Safety configuration:")
            print("-" * 40)
            for setting in safety_config:
                category = setting["category"].name
                threshold = setting["threshold"].name
                print(f"  {category}: {threshold}")
            print("-" * 40)
            print()
            print("All harm categories set to BLOCK_NONE")
            print("This allows processing of content that might otherwise be blocked")
            print("Note: Content is still preprocessed for safety before sending to API")
        except Exception as e:
            print(f"Error testing safety config: {e}")
        return
    
    # Check for show-models argument
    if len(sys.argv) > 1 and sys.argv[1] == '--show-models':
        try:
            api_key = load_environment()
            print("Querying Gemini API for accessible models...")
            api_models = query_api_models(api_key)
            if api_models:
                print(f"Models accessible with your API key ({len(api_models)} total):")
                for model in api_models:
                    print(f"  - {model}")
            else:
                print("No models accessible with your API key")
        except Exception as e:
            print(f"Error querying API: {e}")
        return
    
    print("Gemini Text to Knowledge Processor")
    print("=" * 40)
    
    # Check if input directory exists and has files
    input_dir = Path('input')
    if not input_dir.exists():
        print("Error: input directory not found")
        sys.exit(1)
    
    txt_files = list(input_dir.glob('*.txt'))
    if not txt_files:
        print("No .txt files found in input directory")
        sys.exit(0)
    
    # Load environment and setup
    api_key = load_environment()
    model = setup_gemini(api_key)
    prompt = read_prompt()
    
    # Display model information
    model_name = os.getenv('GEMINI_MODEL', 'gemini-pro')
    print(f"Model Configuration: {model_name}")
    print()
    
    # Get already processed files
    complete_files, failed_files = get_processed_files('gemini_complete.txt')
    
    # Initialize counters
    total_files = len(txt_files)
    processed_success = 0
    processed_failed = 0
    skipped = 0
    
    print(f"Found {total_files} text files to process")
    print(f"Already completed: {len(complete_files)}")
    print(f"Previously failed: {len(failed_files)}")
    print()
    
    # Process each text file
    for txt_file in txt_files:
        filename = txt_file.name
        
        # Skip if already processed or failed
        if filename in complete_files:
            print(f"Skipping {filename} (already completed)")
            skipped += 1
            continue
            
        if filename in failed_files:
            print(f"Skipping {filename} (previously failed)")
            skipped += 1
            continue
        
        print(f"Processing {filename}...")
        
        try:
            # Read text content
            with open(txt_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Preprocess content for safety
            print(f"  Preprocessing content for safety...")
            safe_content = preprocess_content_for_safety(text_content)
            
            # Process with Gemini
            response = process_text_with_gemini(model, prompt, safe_content)
            
            # Check if processing was successful
            if response.startswith('Error:'):
                print(f"  Failed: {response}")
                update_tracking_files(filename, False)
                processed_failed += 1
            else:
                # Save output
                if save_output(response, filename):
                    print(f"  Success: Saved to output/{txt_file.stem}.md")
                    update_tracking_files(filename, True)
                    processed_success += 1
                else:
                    print(f"  Failed: Could not save output")
                    update_tracking_files(filename, False)
                    processed_failed += 1
                    
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            update_tracking_files(filename, False)
            processed_failed += 1
    
    # Print summary
    print("\n" + "=" * 40)
    print("PROCESSING SUMMARY")
    print("=" * 40)
    print(f"Model used: {os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')}")
    print(f"Total text files found: {total_files}")
    print(f"Successfully processed: {processed_success}")
    print(f"Processed but failed: {processed_failed}")
    print(f"Skipped (already processed): {skipped}")
    print(f"Not processed: {total_files - processed_success - processed_failed - skipped}")

if __name__ == "__main__":
    main()
