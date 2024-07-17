import fitz  # PyMuPDF
import os
import re


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# 라인 지우기
def remove_specific_lines(text, filename):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if "법제처" in line and "국가법령정보센터" in line:
            continue
        if line.strip() == filename:
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def save_text_to_file(text, output_path):
    with open(output_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)

def remove_patterns(text):
    patterns = [
        r"\[전문개정.*?\]",
        r"\[제목개정.*?\]",
        r"\[본조신설.*?\]",
        r"\<개정.*\>",
        r"\<신설.*?\>",

    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return text


# 문서 분할 함수 (장 -> 조)
def split_into_chapters_and_articles(text):
    chapters = []
    current_chapter = []
    for line in text.split('\n'):
        if line.strip().startswith("제") and "장" in line:
            if current_chapter:
                chapters.append('\n'.join(current_chapter))
                current_chapter = []
        current_chapter.append(line)
    if current_chapter:
        chapters.append('\n'.join(current_chapter))

    articles = []
    for chapter in chapters:
        current_article = []
        for line in chapter.split('\n'):
            if line.strip().startswith("제") and "조" in line:
                if current_article:
                    articles.append('\n'.join(current_article))
                    current_article = []
            current_article.append(line)
        if current_article:
            articles.append('\n'.join(current_article))

    return articles


def convert_pdfs_to_txt(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)

                filename = os.path.splitext(file)[0]

                # Remove the specific lines
                text = remove_specific_lines(text, filename)

                # Remove specific patterns
                text = remove_patterns(text)

                # Create corresponding output path for the text file
                relative_path = os.path.relpath(pdf_path, input_dir)
                txt_filename = os.path.splitext(relative_path)[0] + ".txt"
                output_path = os.path.join(output_dir, txt_filename)

                # Create directories if they don't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                save_text_to_file(text, output_path)
                print(f"Converted {pdf_path} to {output_path}")


input_directory = "datas/construct_day_labor"
output_directory = "datas/change_text"
convert_pdfs_to_txt(input_directory, output_directory)
