import pdfplumber


def extract_text_from_pdf(pdf_path, txt_path):
    text = ""
    # PDF 파일 열기
    with pdfplumber.open(pdf_path) as pdf:
        # 각 페이지의 텍스트 추출
        for page in pdf.pages:
            text += page.extract_text()

    # 텍스트 파일로 저장
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)


# PDF 파일 경로와 저장할 텍스트 파일 경로
pdf_path = 'datas/summary/건설일용근로자_생활법령.pdf'
txt_path = 'datas/summary/건설일용근로자_생활법령.json'

# 함수 호출
extract_text_from_pdf(pdf_path, txt_path)

print(f"텍스트가 {txt_path}에 저장되었습니다.")
