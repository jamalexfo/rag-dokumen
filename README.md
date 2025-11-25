# Edubia RAG System

Sistem RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan berdasarkan dokumen lokal menggunakan Google Gemini 2.5 Flash dan HuggingFace embeddings.

## Fitur

- 📄 Support dokumen PDF dan TXT
- 🤖 Menggunakan Google Gemini 2.5 Flash sebagai LLM
- 🔍 Vector search dengan ChromaDB dan HuggingFace embeddings
- 💬 Fallback ke base model jika jawaban tidak ditemukan di dokumen
- 🎯 3 cara penggunaan: Terminal, REST API, dan Streamlit UI

## Instalasi

### 1. Clone atau Download Project

```bash
cd path/to/project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Buat file `.env` di root folder:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Tambahkan Dokumen

Letakkan dokumen PDF atau TXT Anda di folder `documents/`:

```
documents/
├── file1.pdf
├── file2.txt
└── ...
```

## Cara Penggunaan

### 1. Terminal Interaktif

Cara paling sederhana untuk testing:

```bash
python main.py
```

Kemudian ketik pertanyaan Anda:

```
Ask a question: apa itu edubia
```

Ketik `exit`, `quit`, atau `q` untuk keluar.

**Kelebihan:**
- Mudah dan cepat untuk testing
- Tidak perlu setup server
- Langsung lihat hasil

### 2. REST API dengan FastAPI

Untuk integrasi dengan aplikasi lain:

```bash
uvicorn api:app --reload --port 8001
```

API akan berjalan di `http://localhost:8001`

#### Endpoints

**GET /** - Health check
```bash
curl http://localhost:8001/
```

**POST /query** - Tanyakan pertanyaan

Request:
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "apa itu Retrieval-Augmented Generation"}'
```

Response:
```json
{
  "answer": "Retrieval-Augmented Generation adalah...",
  "sources": ["documents/rag_info.txt"]
}
```

**Kelebihan:**
- Bisa diintegrasikan dengan aplikasi lain
- Support multiple concurrent requests
- RESTful API standard

### 3. Streamlit UI

Untuk interface web yang user-friendly:

```bash
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`

**Cara Pakai:**
1. Klik tombol "Initialize RAG System" di sidebar
2. Tunggu hingga sistem selesai loading dokumen
3. Ketik pertanyaan di chat input
4. Lihat jawaban dan sources-nya

**Kelebihan:**
- UI yang menarik dan mudah digunakan
- Chat history tersimpan
- Bisa lihat sources dengan mudah
- Cocok untuk demo atau presentasi

## Struktur Project

```
.
├── documents/              # Folder untuk dokumen PDF/TXT
├── .env                    # Environment variables (API key)
├── .env.example           # Template untuk .env
├── requirements.txt       # Python dependencies
├── rag_core.py           # Core RAG logic
├── main.py               # Terminal interface
├── api.py                # FastAPI REST API
├── app.py                # Streamlit UI
└── README.md             # Dokumentasi ini
```

## Cara Kerja

1. **Load Documents**: Sistem membaca semua file PDF dan TXT dari folder `documents/`
2. **Split Text**: Dokumen dipecah menjadi chunks (1000 karakter dengan overlap 200)
3. **Create Embeddings**: Setiap chunk diubah menjadi vector menggunakan HuggingFace `all-MiniLM-L6-v2`
4. **Store in VectorDB**: Vector disimpan di ChromaDB untuk pencarian cepat
5. **Query Processing**:
   - User mengirim pertanyaan
   - Sistem mencari chunks yang relevan dari VectorDB
   - LLM (Gemini) menjawab berdasarkan context yang ditemukan
   - Jika tidak ditemukan (NOT_FOUND), fallback ke base model

## Troubleshooting

### Port 8000 sudah digunakan

Gunakan port lain:
```bash
uvicorn api:app --reload --port 8001
```

### Error: GOOGLE_API_KEY not found

Pastikan file `.env` sudah dibuat dan berisi API key yang valid.

### Error: No documents found

Pastikan ada file PDF atau TXT di folder `documents/`.

### Model not found error

Pastikan menggunakan model yang benar: `models/gemini-2.5-flash`

Cek model yang tersedia:
```bash
python check_model.py
```

## Tips

- Untuk hasil terbaik, gunakan dokumen yang terstruktur dengan baik
- Chunk size bisa disesuaikan di `rag_core.py` (default: 1000)
- Temperature bisa diatur untuk kontrol kreativitas jawaban
- Gunakan Streamlit untuk demo, API untuk production

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **API**: FastAPI
- **UI**: Streamlit
- **Document Loaders**: PyPDF, TextLoader

## License

MIT License
