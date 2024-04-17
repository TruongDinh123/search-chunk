import uuid


def generate_unique_id():
    unique_id = uuid.uuid4().hex
    return str(unique_id)

def get_passage_emb_template(
    title: str,
    types: str,
    authors: str,
    chunk_text: str,
):
    """Dùng khi embedding"""
    return f"passage: Tiêu đề: {title} ; Tác giả: {authors} ; Thể loại: {types} ; Nội dung: {chunk_text}"

def get_query_emb_template(
    query: str,
):
    """Dùng khi embedding"""
    return f"passage: {query}"