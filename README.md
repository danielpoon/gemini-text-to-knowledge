# Gemini Text to Knowledge Processor

Version 1.0

by Daniel Poon

## The Why?

You can have alot of textual information lying around (like books) such as maybe a full commencement speech from Taylor Swift. It all sounded good to have it, but that's not knowledge. Knowledge is what we could ultimately get benefits from it. We may choose to summarize it or we may want to extract key insights or topics from these contents. This maybe the essence of research back in the days when you had to sift through books and journals, in order to find the pieces of knowledge (from a variety of sources), and then form your conclusion, and out came the knowledge gained.

But surely we have to make these things simpler and faster. This is why I wrote this script to do the above. For example, if you possess all the great speeches from all the past leaders such as Martin Luther King, Ronald Reagan, George Washington, Elon Musk etc, you can use this script with AI to generate the required knowledge from each of these, and feed it into another AI model like an LLM. You can then create an AI agent that will remind people to do great things and be good people, right? You get the idea. This is just an example. And I have my own use for this.

## Overview

A Python script that processes one or more text files using Google's Gemini AI to extract structured knowledge and insights. The script reads text files from a data directory, processes them with a custom prompt, and saves the AI-generated responses as markdown files.

## Features

- Processes multiple text files in batch
- Uses Google's Gemini AI for intelligent text analysis
- Converts AI responses to structured markdown format
- Tracks successful and failed processing attempts
- Skips already processed files to avoid duplication
- Comprehensive error handling and logging
- Configurable prompt system
- **Two-stage content filtering system for safety compliance**
- **Customizable profanity filtering with better_profanity library**
- **Configurable find/replace patterns for content modification**

## Prerequisites

- Python 3.7 or higher
- Google AI API key
- Internet connection for Gemini AI access

## Installation

1. Clone or download this repository

2. Install required dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google AI API key:
   
   - Create a `.env` file in the project root
   - Add your API key: `GOOGLE_API_KEY=your_actual_api_key_here`

4. **Set up content filtering (optional but recommended):**
   
   - Customize profanity words in `better_profanity.txt`
   - Configure find/replace patterns in `find_and_replace.txt`
   - Set environment variables for filtering behavior

## Directory Structure

```
gemini-text-to-knowledge/
├── input/                   # Input text files directory
├── output/                  # Generated markdown files
├── gemini_prompt.txt        # Custom prompt for Gemini AI
├── better_profanity.txt     # Custom profanity words for content filtering
├── find_and_replace.txt     # Custom find/replace patterns for content safety
├── gemini_text_to_knowledge.py  # Main processing script
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Usage

1. Place your text files in the `input/` directory

2. Customize the prompt in `gemini_prompt.txt` if needed

3. Run the script:
   
   ```bash
   python gemini_text_to_knowledge.py
   ```

## How It Works

1. **File Discovery**: Scans the `input/` directory for `.txt` files
2. **Content Preprocessing**: Applies two-stage content filtering for safety compliance
3. **Prompt Loading**: Reads the custom prompt from `gemini_prompt.txt`
4. **Processing**: Sends each filtered text file to Gemini AI with the prompt
5. **Output Generation**: Converts AI responses to markdown format
6. **File Saving**: Saves processed content to `output/` directory
7. **Tracking**: Maintains logs of successful and failed processing

## Output Files

- **Markdown Files**: Each processed text file generates a corresponding `.md` file in the `output/` directory
- **gemini_complete.txt**: List of successfully processed files
- **gemini_failed.txt**: List of files that failed processing

## Error Handling

The script handles various error scenarios:

- Content policy violations
- Service unavailability (501 errors)
- File read/write errors
- API connection issues

## Customization

### Modifying the Prompt

Edit `gemini_prompt.txt` to change how Gemini AI processes your text files. The current prompt is designed for stock trading transcript analysis but can be adapted for any domain.

### Content Filtering Configuration

#### Profanity Filtering
- **File**: `better_profanity.txt` - Add one profanity word per line
- **Behavior**: Set `REMOVE_PROFANITY_COMPLETELY=true` in `.env` to completely remove words, or `false` (default) to mask with asterisks
- **Example**: Add words like "damn", "hell", "crap" to the file

#### Find/Replace Patterns
- **File**: `find_and_replace.txt` - Use format: `find_word|replace_word`
- **Behavior**: Case-insensitive replacement of specific terms
- **Example**: `loser|losing position`, `bullied|forced`

#### Environment Variables
```bash
REMOVE_PROFANITY_COMPLETELY=false  # true = remove completely, false = mask with ****
RATE_LIMIT_PAUSE=60                # seconds to pause between API calls
DISABLE_RATE_LIMIT=false           # true = no pause, false = use pause
SHOW_MODEL_LIST=false              # true = show detailed model list, false = summary only
```

### Output Format

The script automatically formats AI responses with bold section headers. Modify the `format_markdown_sections()` function to customize the output formatting.

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `.env` file contains a valid `GOOGLE_API_KEY`
2. **No Files Found**: Check that your `input/` directory contains `.txt` files
3. **Permission Errors**: Ensure the script has read/write permissions for all directories
4. **Module Import Error**: Activate the virtual environment before running: `source .venv/bin/activate`
5. **Content Policy Violations**: Check that content filtering is properly configured in `better_profanity.txt` and `find_and_replace.txt`

### API Limits

- Be aware of Google AI API rate limits and quotas
- Large text files may take longer to process
- Consider processing files in smaller batches if needed
- **Rate Limiting**: Script includes configurable pauses between API calls (default: 60 seconds)
- **Content Safety**: Two-stage filtering helps prevent content policy violations

## Dependencies

- `google-generativeai`: Google's official Gemini AI Python library
- `python-dotenv`: Environment variable management
- `better-profanity`: Content filtering and profanity detection
- `urllib3<2.0.0`: HTTP client library (compatible with LibreSSL)
- Standard Python libraries: `os`, `sys`, `pathlib`, `typing`, `re`

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:

1. Check the error messages in the console output
2. Verify your API key and internet connection
3. Ensure all required files and directories exist
