import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text_chunk(chunk: str) -> str:
    """
    Cleans/lemmatizes a chunk of text using SpaCy.
    """
    doc = nlp(chunk.strip())
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(lemmatized_tokens)

def preprocess_document(text: str) -> str:
    """
    Processes text which may already be chunked (newline separated).
    Split on newlines and process chunk-by-chunk.
    """
    chunks = text.split("\n")
    processed_chunks = []
    for chunk in chunks:
        if chunk.strip():
            processed_chunks.append(preprocess_text_chunk(chunk))
    return "\n".join(processed_chunks)