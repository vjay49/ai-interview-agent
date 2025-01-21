import os
import requests
import pdfplumber
import tempfile
from diskcache import Cache

# disk-based cache to avoid repeated parsing
cache = Cache("./cache_dir")

def download_file(url: str) -> str:
    '''
    Downloads file from a remote URL to a temporary location.
    Returns local file path of the downloaded file.
    '''
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        return tmp.name


def load_text_from_pdf(pdf_path: str) -> str:
    """
    Load and extract text from local PDF file using pdfplumber.
    Pages with no selectable text are skipped.
    """
    cache_key = f"pdf:{pdf_path}"
    cached_text = cache.get(cache_key)
    if cached_text:
        return cached_text
    
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                all_text.append(extracted_text)

    joined_text = "\n".join(all_text)
    cache[cache_key] = joined_text
    return joined_text


def chunk_text(text: str, max_chunk_size: int = 1000) -> list:
    """
    Splits text into chunks of up to 'max_chunk_size' characters each.
    Helps with large docs and is good for embeddings.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def load_text_from_file(filepath: str) -> str:
    """
    Loads text from a local .txt file with caching.
    """
    cache_key = f"textfile:{filepath}"
    cached_text = cache.get(cache_key)
    if cached_text:
        return cached_text
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    cache[cache_key] = text
    return text


def load_text_from_url(url: str) -> str:
    """
    Loads text from given URL, caching results.
    """
    cache_key = f"url:{url}"
    cached_text = cache.get(cache_key)
    if cached_text:
        return cached_text

    response = requests.get(url)
    response.raise_for_status()
    text = response.text

    cache[cache_key] = text
    return text


def ingest_data(job_path: str, company_path: str, resume_path: str, do_chunking: bool = True) -> tuple:
    """
    Ingests job post, company profile, and candidate resume from local .txt, .pdf or URL-based .txt. PDF is handled via pdfplumber.
    - do_chunking: if True, chunk large texts to improve embedding performance.
    Returns (job_post, company_profile, candidate_resume) as strings.
    """
    # 1. Job Post
    if job_path.startswith("http"):
        job_post = load_text_from_url(job_path)
    else:
        job_post = load_text_from_file(job_path)

    # 2. Company Profile
    if company_path.startswith("http"):
        company_profile = load_text_from_url(company_path)
    else:
        company_profile = load_text_from_file(company_path)

    # 3. Candidate Resume
    resume_text = None
    if resume_path.lower().startswith("http"):
        # Download the PDF, then parse it
        local_pdf = download_file(resume_path)
        resume_text = load_text_from_pdf(local_pdf)
    else:
        _, ext = os.path.splitext(resume_path.lower())
        if ext == '.pdf':
            resume_text = load_text_from_pdf(resume_path)
        else:
            resume_text = load_text_from_file(resume_path)

    if do_chunking:
        # Convert to chunks and rejoin as a single string or keep as list
        chunks = chunk_text(resume_text, max_chunk_size=1000)
        # Store it as a single string for downstream consistency
        resume_text = "\n".join(chunks)

    return job_post, company_profile, resume_text