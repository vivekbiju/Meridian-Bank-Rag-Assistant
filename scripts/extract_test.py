import pdfplumber

def extract_pdf():
    text = ""
    with pdfplumber.open("corpus/meridian-handbook.pdf") as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    with open("extracted_output.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    extract_pdf()
    print("Extraction complete. Check extracted_output.txt")