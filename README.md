# Essay Evaluation CLI

This project provides a command-line application for evaluating batches of student essays with the xAI Grok API. It accepts up to 30 essay text files, combines them with course materials and a test question, and generates scored feedback along with individual PDF reports per student.

## Quick Start

1. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure the API key**
   - Copy `.env.example` to `.env`.
   - Set `XAI_API_KEY` to your xAI key inside `.env`.

3. **Prepare inputs**
   - Place up to 30 cleaned essay `.txt` or `.pdf` files in a directory. Each file should list the student on the first page using `Student: Full Name` or `Name: Full Name`. If no label is present, the filename will be used. PDF files are converted to text automatically.
   - Provide `.txt` or `.pdf` files containing the reading material and the test question.
   - Supply a context string via the CLI or a text file.

4. **Run the evaluator**
   ```bash
   python app.py \
     --essays_dir path/to/essays \
     --material_file path/to/material.txt \
     --question_file path/to/question.txt \
     --context "10th-grade English class focusing on modern American literature."
   ```

   Optional flags:
   - `--context_file path/to/context.txt` (instead of `--context`)
   - `--output_dir evaluations_output` to control PDF destination
   - `--csv_path evaluations.csv` to control CSV location
   - `--summarise-material` to request an AI-generated summary of the reading material
   - `--verbose` for detailed logging

5. **Outputs**
   - `evaluations.csv` summarising numeric scores.
   - Individual PDFs per student inside the chosen output directory.

## Notes

- The app batches essays to stay within the model context window and issues parallel API requests when helpful.
- Token usage is logged when provided by the API.
- Errors including malformed API responses are surfaced with meaningful CLI messages.
