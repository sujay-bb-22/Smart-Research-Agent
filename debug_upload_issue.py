import os, tempfile
from types import SimpleNamespace
from fastapi.testclient import TestClient
import main

td = tempfile.mkdtemp()
os.chdir(td)
os.makedirs('data', exist_ok=True)
os.makedirs('db', exist_ok=True)
main.db = None
main.retriever = None
main.embeddings = None

def fake_get_pdf_chunks(path, document_id=None, filename=None):
    content = 'Document A content' if 'A' in path else 'Document B content'
    return [SimpleNamespace(page_content=content, metadata={'source': path,'page':1,'document_id':document_id,'filename':filename,'chunk_id':f'{document_id}-chunk-001'})]

main.get_pdf_chunks = fake_get_pdf_chunks
main.load_db = lambda: None
client = TestClient(main.app)
for idx in [1,2]:
    r = client.post('/upload', files={'file': ('report.pdf', b'A' if idx == 1 else b'B', 'application/pdf')})
    print('STATUS', idx, r.status_code)
    print('BODY', r.text)
    print('DB type', type(main.db))
    if main.db is not None:
        print('HAS EMBED', hasattr(main.db, '_embedding_function'), main.db._embedding_function)
        print('COLLECTION', getattr(main.db, '_collection', None))
    print('EMB', main.embeddings)
    print('DATA', os.listdir('data'))
    print('---')
