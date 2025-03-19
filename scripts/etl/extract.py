# Extract datasets from sources

import PyPDF2

# Path to the uploaded PDF file
pdf_path = "../../therapists_guide_to_brief_cbtmanual.pdf"
txt_output_path = "../../therapists_guide_to_brief_cbtmanual.txt"

# Extract text from the PDF
with open(pdf_path, "rb") as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Save the extracted text to a .txt file
with open(txt_output_path, "w", encoding="utf-8") as txt_file:
    txt_file.write(extracted_text)

# Provide the file path for user download
txt_output_path