import pdfplumber

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def is_duplicate(new_text, existing_texts):
    for text in existing_texts:
        if new_text[:300] == text[:300]:
            return True
    return False


def generate_answer(question, documents, age_mode):
    combined_text = " ".join(documents)[:3000]

    if not combined_text:
        return "No documents available. Please upload PDFs first."

    if age_mode == "child":
        return "😊 Simple Explanation:\n" + combined_text[:400]
    elif age_mode == "adult":
        return "📘 Detailed Explanation:\n" + combined_text[:900]
    else:
        return "📖 Student-Friendly Answer:\n" + combined_text[:600]


def generate_summary(documents):
    combined_text = " ".join(documents)
    if not combined_text:
        return "No documents available to summarize."
    return combined_text[:800]