import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_section(section, chunk_size=1200, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "."],
        length_function=len,
    )
    return text_splitter.split_text(section)


if __name__ == "__main__":
    # 텍스트 파일 경로 설정
    file_path = "datas/change_text/임금채권보장법.txt"
    all_documents = []
    file_name = os.path.basename(file_path).replace('.txt', '')

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

        # Step 1: '제0장' 단위로 텍스트 나누기
    chapters = re.split(r'(\n*제\d+장의?\d* [^\n]+)', text)
    print(chapters)
    if len(chapters) > 1:
        chapters = [chapters[i] + chapters[i + 1] for i in range(1, len(chapters), 2)]
    else:
        chapters = [text]
    chunks = []

    # Step 2: 각 '제0장' 단위로 나눠진 텍스트를 '제0조' 단위로 나누기
    for chapter in chapters:
        chapter_title_match = re.match(r'(\n*제\d+장의?\d* [^\n]+)', chapter)
        chapter_title = chapter_title_match.group(1) if chapter_title_match else ""
        sections = re.split(r'(\n*제\d+조의?\d*\([^)]+\))', chapter)
        # 결과 리스트를 생성
        results = []
        for i in range(1, len(sections), 2):
            title = sections[i].strip()
            content = sections[i + 1]
            results.append(title)
            results.append(content.strip())

        # 각 섹션을 청크로 분할하고 파일 이름과 장 제목을 앞에 추가
        previous_section_title = ""
        for i in range(0, len(results), 2):
            title = results[i]
            content = results[i + 1]
            combined_text = f"{title} {content}"

            section_chunks = split_section(combined_text)
            print(f"section_chunks...: {section_chunks}")

            # 이전 섹션 제목을 추가하여 연속성 보장
            for j, chunk in enumerate(section_chunks):
                print(f"연속성{j}: {chunk}")
                if j == 0:
                    chunk = f"{file_name} {chapter_title} {chunk}"
                else:
                    chunk = f"{file_name} {chapter_title} {title} {chunk}"
                chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")

    for doc in chunks:
        all_documents.append({
            "text": doc,
            "source": file_path
        })

    print(all_documents)