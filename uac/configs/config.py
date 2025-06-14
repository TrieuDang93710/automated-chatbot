class Config:
    def __init__(self):
        self.generate_model_name = "llama3.2:latest"
        self.function_calling_model_name = "llama3.2:latest"
        self.extraction_model_name = "llama3-70b-8192"
        self.preprocess_model_name = "llama3.2:latest"
        self.embedding_model_name = "nomic-embed-text:v1.5"
        self.storage_directory_path = "storage"
        self.collection_name = "knowledge_base"
        self.user_management_collection_name = "user_database"
        self.db_name = "admission_db"
        self.vector_field_name = "embedding"
        self.index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 150},
        }
        self.vector_dim = 768
        self.cosine_retrieval_threshold = 0.42
        self.topk = 3
        self.availabel_data_dir = "data/raw/"
        self.embedding_model_cache_dir = "/HDD/models/ai-tutor/"
        self.bot_name = "ADUchat, Chatbot hỗ trợ tư vấn tìm kiếm việc làm"
        self.bot_type = "uac"
