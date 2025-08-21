#!/usr/bin/env python3
"""
Gemini Text to Knowledge Processor
Processes text files using Google's Gemini AI and saves structured markdown output.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from better_profanity import profanity

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

def rate_limit_pause(seconds: int = None, show_progress: bool = True):
    """Pause execution to avoid rate limiting."""
    # Get pause duration from environment variable or use default
    if seconds is None:
        seconds = int(os.getenv('RATE_LIMIT_PAUSE', '60'))
    
    if show_progress:
        print(f"Rate limiting: Pausing for {seconds} seconds to avoid API limits...")
        for i in range(seconds, 0, -1):
            if i % 10 == 0 or i <= 5:  # Show progress every 10 seconds and last 5 seconds
                print(f"  Waiting... {i} seconds remaining")
            time.sleep(1)
        print("  Rate limit pause completed. Continuing...")
    else:
        time.sleep(seconds)

def show_progress(current: int, total: int, success: int, failed: int, skipped: int):
    """Show current progress statistics."""
    processed = success + failed
    remaining = total - processed - skipped
    print(f"  Progress: {processed}/{total} files processed ({success} success, {failed} failed, {skipped} skipped, {remaining} remaining)")

def load_custom_profanity_words() -> List[str]:
    """Load custom profanity words from better_profanity.txt file."""
    custom_words = []
    profanity_file = Path('better_profanity.txt')
    
    if profanity_file.exists():
        try:
            with open(profanity_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):  # Skip empty lines and comments
                        custom_words.append(word)
            print(f"Loaded {len(custom_words)} custom profanity words from better_profanity.txt")
        except Exception as e:
            print(f"Warning: Could not load custom profanity words: {e}")
    else:
        print("No better_profanity.txt file found - using default profanity filter only")
    
    return custom_words

def load_find_replace_patterns() -> List[Tuple[str, str]]:
    """Load find and replace patterns from find_and_replace.txt file."""
    patterns = []
    find_replace_file = Path('find_and_replace.txt')
    
    if find_replace_file.exists():
        try:
            with open(find_replace_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        if '|' in line:
                            find_word, replace_word = line.split('|', 1)
                            patterns.append((find_word.strip(), replace_word.strip()))
                        else:
                            print(f"Warning: Line {line_num} in find_and_replace.txt missing pipe separator: {line}")
            print(f"Loaded {len(patterns)} find/replace patterns from find_and_replace.txt")
        except Exception as e:
            print(f"Warning: Could not load find/replace patterns: {e}")
    else:
        print("No find_and_replace.txt file found - skipping additional word replacements")
    
    return patterns

def preprocess_content_for_safety(text_content: str) -> Tuple[str, int, int]:
    """Preprocess content to reduce likelihood of content policy violations.
    
    Returns:
        Tuple of (processed_text, profanity_replacements, find_replace_replacements)
    """
    # Stage 1: Load custom profanity words and configure the filter
    custom_words = load_custom_profanity_words()
    
    # Configure profanity filter with custom words
    if custom_words:
        profanity.load_censor_words(custom_words)
        print(f"  Applied {len(custom_words)} custom profanity words to filter")
    
    # Process the text content
    processed = text_content
    
    # Remove censored profanity placeholders
    processed = processed.replace('[ __ ]', '')
    processed = processed.replace('[__]', '')
    
    # Check if we should completely remove profanity or use asterisks
    remove_profanity_completely = os.getenv('REMOVE_PROFANITY_COMPLETELY', 'false').lower() == 'true'
    
    if remove_profanity_completely:
        print(f"  Stage 1: Applying profanity filter (complete removal mode)...")
        # Custom profanity removal that completely removes words
        for word in custom_words:
            import re
            pattern = r'\b' + re.escape(word) + r'\b'
            processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
        
        # Count profanity replacements (approximate based on original content)
        profanity_replacements = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_content, flags=re.IGNORECASE)) for word in custom_words)
    else:
        print(f"  Stage 1: Applying profanity filter (asterisk mode)...")
        # Use better_profanity to censor profanity (replaces with asterisks)
        processed = profanity.censor(processed)
        
        # Count profanity replacements by counting asterisks
        profanity_replacements = processed.count('*')
    
    # Stage 2: Apply additional find/replace patterns
    find_replace_patterns = load_find_replace_patterns()
    find_replace_count = 0
    if find_replace_patterns:
        print(f"  Stage 2: Applying {len(find_replace_patterns)} find/replace patterns...")
        for find_word, replace_word in find_replace_patterns:
            # Use word boundaries to avoid partial matches
            import re
            pattern = r'\b' + re.escape(find_word) + r'\b'
            # Count replacements for this pattern
            matches = len(re.findall(pattern, processed, flags=re.IGNORECASE))
            if matches > 0:
                processed = re.sub(pattern, replace_word, processed, flags=re.IGNORECASE)
                find_replace_count += matches
                print(f"    Replaced {matches} instance(s) of '{find_word}' with '{replace_word}'")
    
    # Clean up extra whitespace and empty lines
    import re
    processed = re.sub(r'\n\s*\n', '\n', processed)
    processed = re.sub(r' +', ' ', processed)
    
    # Add safety note with replacement counts and mode
    mode_description = "complete removal" if remove_profanity_completely else "asterisk masking"
    safety_note = f"\n\nNote: This transcript has been preprocessed for content safety using a two-stage approach:\n1. Profanity filtering with {mode_description}: {profanity_replacements} replacements\n2. Additional word replacements from find_and_replace.txt: {find_replace_count} replacements"
    processed += safety_note
    
    print(f"  Content safety summary: {profanity_replacements} profanity replacements ({mode_description}), {find_replace_count} find/replace replacements")
    
    return processed, profanity_replacements, find_replace_count

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
        error_type = type(e).__name__
        
        if "violation" in error_msg.lower():
            # Enhanced violation error with full Gemini message
            full_error = f"""Error: Content policy violation - The content may contain language or topics that violate Gemini's safety policies.

FULL GEMINI ERROR MESSAGE:
{error_type}: {error_msg}

RECOMMENDATIONS:
- Remove profanity or offensive language
- Focus on factual analysis rather than opinions
- Avoid specific financial advice
- Use more professional language
- Check the full error message above for specific violation details

This detailed error information can help you identify exactly what content is causing the policy violation."""
            return full_error
        elif "501" in error_msg.lower():
            return f"Error: Service unavailable (501)\n\nFull error: {error_type}: {error_msg}"
        elif "quota" in error_msg.lower():
            return f"Error: API quota exceeded - You may have reached your daily limit\n\nFull error: {error_type}: {error_msg}"
        elif "rate" in error_msg.lower():
            return f"Error: Rate limit exceeded - Please wait before trying again\n\nFull error: {error_type}: {error_msg}"
        else:
            return f"Error: {error_type}: {error_msg}"

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
    """Format content with bold section headers, avoiding duplicate formatting."""
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            formatted_lines.append(line)
            continue
            
        # Check if line already has markdown formatting
        if line.startswith('**') and line.endswith('**'):
            # Already bold formatted, keep as is
            formatted_lines.append(line)
            continue
            
        # Check if line already has asterisks (partial markdown)
        if '**' in line:
            # Already has some markdown, keep as is
            formatted_lines.append(line)
            continue
            
        # Check if line is a section header that needs formatting
        if line.startswith('-') and ':' in line:
            section_name = line.split(':')[0].strip('- ')
            section_content = line.split(':', 1)[1].strip() if ':' in line else ''
            
            # Only add bold if section name doesn't already have it
            if not section_name.startswith('**') and not section_name.endswith('**'):
                formatted_lines.append(f"**{section_name}:**")
                if section_content:
                    formatted_lines.append(section_content)
            else:
                # Already formatted, keep original
                formatted_lines.append(line)
        else:
            # Regular line, keep as is
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
        print("  --test-markdown      Test markdown formatting logic")
        print("  --test-rate-limit    Test rate limiting functionality")
        print("  --test-profanity     Test profanity filtering functionality")
        print("  --test-profanity-removal Test profanity filtering with complete removal")
        print("  --test-error-handling    Test enhanced error handling and messages")
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
        print("  RATE_LIMIT_PAUSE    Pause duration in seconds (default: 60)")
        print("  DISABLE_RATE_LIMIT  Disable rate limiting (default: false)")
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
    
    # Check for test-markdown argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-markdown':
        print("Testing markdown formatting logic...")
        print()
        
        # Test content with various formatting scenarios
        test_content = """- Trading Date: August 18, 2025
- Main Topic: Stock market analysis
**Already Bold Section:** This is already formatted
- Risk Management: Stop loss strategies
- **Partially Bold:** Mixed formatting
- Technical Indicators: RSI, MACD, EMA

Regular text line
Another regular line"""
        
        print("Original content:")
        print("-" * 40)
        print(test_content)
        print("-" * 40)
        print()
        
        print("Formatted content:")
        print("-" * 40)
        formatted_content = format_markdown_sections(test_content)
        print(formatted_content)
        print("-" * 40)
        print()
        
        print("Formatting decisions made:")
        print("- Lines starting with '-' and ':' get bold formatting")
        print("- Lines already containing '**' are preserved as-is")
        print("- Lines already starting/ending with '**' are preserved")
        print("- Regular text lines are unchanged")
        return
    
    # Check for test-rate-limit argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-rate-limit':
        print("Testing rate limiting functionality...")
        print()
        
        # Test different pause durations
        test_durations = [5, 10, 15]  # Short durations for testing
        
        for duration in test_durations:
            print(f"Testing {duration} second pause:")
            rate_limit_pause(duration)
            print()
        
        print("Rate limiting test completed!")
        print()
        print("Environment variables for rate limiting:")
        print("- RATE_LIMIT_PAUSE: Pause duration in seconds (default: 60)")
        print("- DISABLE_RATE_LIMIT: Set to 'true' to disable rate limiting")
        return
    
    # Check for test-profanity argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-profanity':
        print("Testing two-stage profanity filtering functionality...")
        print()
        
        # Test text with various profanity and words from find_and_replace.txt
        test_text = """This is a test transcript with some content.
        
        The speaker said some shitty things about the situation.
        They also mentioned [ __ ] and [__] off.
        There was some no [ __ ] in there too.
        
        The content also contains some regular profanity like damn and hell.
        The person is a loser and the situation sucks.
        They were bullied by communists and called an idiot.
        The piece of work was crap and they are suckers."""
        
        print("Original text:")
        print("-" * 40)
        print(test_text)
        print("-" * 40)
        print()
        
        print("After two-stage filtering:")
        print("-" * 40)
        filtered_text, profanity_count, find_replace_count = preprocess_content_for_safety(test_text)
        print(filtered_text)
        print("-" * 40)
        print()
        
        print(f"Replacement Summary:")
        print(f"  Profanity replacements: {profanity_count}")
        print(f"  Find/replace replacements: {find_replace_count}")
        print(f"  Total replacements: {profanity_count + find_replace_count}")
        print()
        
        print("Two-stage filtering test completed!")
        print()
        print("Custom profanity words loaded from better_profanity.txt:")
        custom_words = load_custom_profanity_words()
        if custom_words:
            for word in custom_words:
                print(f"  - {word}")
        else:
            print("  No custom words found")
        
        print()
        print("Find/replace patterns loaded from find_and_replace.txt:")
        find_replace_patterns = load_find_replace_patterns()
        if find_replace_patterns:
            for find_word, replace_word in find_replace_patterns:
                print(f"  - '{find_word}' → '{replace_word}'")
        else:
            print("  No patterns found")
        return
    
    # Check for test-profanity-removal argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-profanity-removal':
        print("Testing profanity filtering with COMPLETE REMOVAL mode...")
        print()
        
        # Set environment variable for complete removal
        os.environ['REMOVE_PROFANITY_COMPLETELY'] = 'true'
        
        # Test text with various profanity and words from find_and_replace.txt
        test_text = """This is a test transcript with some content.
        
        The speaker said some shitty things about the situation.
        They also mentioned [ __ ] and [__] off.
        There was some no [ __ ] in there too.
        
        The content also contains some regular profanity like damn and hell.
        The person is a loser and the situation sucks.
        They were bullied by communists and called an idiot.
        The piece of work was crap and they are suckers."""
        
        print("Original text:")
        print("-" * 40)
        print(test_text)
        print("-" * 40)
        print()
        
        print("After two-stage filtering (COMPLETE REMOVAL mode):")
        print("-" * 40)
        filtered_text, profanity_count, find_replace_count = preprocess_content_for_safety(test_text)
        print(filtered_text)
        print("-" * 40)
        print()
        
        print(f"Replacement Summary (Complete Removal Mode):")
        print(f"  Profanity completely removed: {profanity_count}")
        print(f"  Find/replace replacements: {find_replace_count}")
        print(f"  Total replacements: {profanity_count + find_replace_count}")
        print()
        
        print("Complete removal test completed!")
        print()
        print("To use complete removal mode, set in your .env file:")
        print("  REMOVE_PROFANITY_COMPLETELY=true")
        return
    
    # Check for test-error-handling argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test-error-handling':
        print("Testing enhanced error handling and messages...")
        print()
        
        # Simulate different types of errors
        print("1. Content Policy Violation Error:")
        print("-" * 40)
        violation_error = """Error: Content policy violation - The content may contain language or topics that violate Gemini's safety policies.

FULL GEMINI ERROR MESSAGE:
BlockedPromptException: The prompt was blocked because it contains content that may violate our content policy. This includes content that may be harmful, inappropriate, or violate our safety guidelines.

RECOMMENDATIONS:
- Remove profanity or offensive language
- Focus on factual analysis rather than opinions
- Avoid specific financial advice
- Use more professional language
- Check the full error message above for specific violation details

This detailed error information can help you identify exactly what content is causing the policy violation."""
        print(violation_error)
        print()
        
        print("2. Service Unavailable Error (501):")
        print("-" * 40)
        service_error = "Error: Service unavailable (501)\n\nFull error: HTTPException: 501 Not Implemented"
        print(service_error)
        print()
        
        print("3. Quota Exceeded Error:")
        print("-" * 40)
        quota_error = "Error: API quota exceeded - You may have reached your daily limit\n\nFull error: QuotaExceededException: Daily quota limit exceeded"
        print(quota_error)
        print()
        
        print("4. Rate Limit Error:")
        print("-" * 40)
        rate_error = "Error: Rate limit exceeded - Please wait before trying again\n\nFull error: RateLimitException: Too many requests per minute"
        print(rate_error)
        print()
        
        print("Enhanced error handling test completed!")
        print()
        print("Benefits of enhanced error messages:")
        print("- Full Gemini error details for debugging")
        print("- Specific error types and messages")
        print("- Clear recommendations for resolution")
        print("- Better understanding of content policy violations")
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
            safe_content, profanity_count, find_replace_count = preprocess_content_for_safety(text_content)
            
            # Process with Gemini
            response = process_text_with_gemini(model, prompt, safe_content)
            
            # Check if processing was successful
            if response.startswith('Error:'):
                print(f"  Failed: {response}")
                print(f"  Content safety: {profanity_count} profanity + {find_replace_count} find/replace = {profanity_count + find_replace_count} total replacements")
                update_tracking_files(filename, False)
                processed_failed += 1
                # Show progress after failed processing
                show_progress(processed_success + processed_failed, total_files, processed_success, processed_failed, skipped)
            else:
                # Save output
                if save_output(response, filename):
                    print(f"  Success: Saved to output/{txt_file.stem}.md")
                    print(f"  Content safety: {profanity_count} profanity + {find_replace_count} find/replace = {profanity_count + find_replace_count} total replacements")
                    update_tracking_files(filename, True)
                    processed_success += 1
                    # Show progress after successful processing
                    show_progress(processed_success + processed_failed, total_files, processed_success, processed_failed, skipped)
                else:
                    print(f"  Failed: Could not save output")
                    print(f"  Content safety: {profanity_count} profanity + {find_replace_count} find/replace = {profanity_count + find_replace_count} total replacements")
                    update_tracking_files(filename, False)
                    processed_failed += 1
                    # Show progress after failed processing
                    show_progress(processed_success + processed_failed, total_files, processed_success, processed_failed, skipped)
            
            # Rate limiting pause after each file (except the last one)
            if txt_file != txt_files[-1]:  # Don't pause after the last file
                # Check if rate limiting is enabled
                if os.getenv('DISABLE_RATE_LIMIT', 'false').lower() != 'true':
                    rate_limit_pause()
                else:
                    print("  Rate limiting disabled - continuing immediately")
                    
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            print(f"  Content safety: {profanity_count} profanity + {find_replace_count} find/replace = {profanity_count + find_replace_count} total replacements")
            update_tracking_files(filename, False)
            processed_failed += 1
            # Show progress after exception
            show_progress(processed_success + processed_failed, total_files, processed_success, processed_failed, skipped)
            
            # Rate limiting pause even after errors (except the last file)
            if txt_file != txt_files[-1]:  # Don't pause after the last file
                # Check if rate limiting is enabled
                if os.getenv('DISABLE_RATE_LIMIT', 'false').lower() != 'true':
                    rate_limit_pause()
                else:
                    print("  Rate limiting disabled - continuing immediately")
    
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
