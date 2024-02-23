# FILE UPLOAD
FILE_UPLOAD_PATH = "./uploaded_files"

# DATABASE ENDPOINT
QDRANTDB_URL = "http://103.176.178.107:6333"
DB_BATCH_SIZE = 256
DB_TIMEOUT = 60

# BM25
ENABLE_BM25 = False
BM25_TOP_K = 100

# EMBEDDING MODEL
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
EMBEDDING_DIM = 768
EMBEDDING_MAX_LENGTH = 256
EMBEDDING_TOP_K = 3

# LLM API ENDPOINT
LLM_MODEL = "google/gemma-7b-it"
TGI_URL = "http://localhost:10025"
API_KEY = "hf_sample_api_key"
MAX_ANSWER_LENGTH = 4096
MAX_MODEL_LENGTH = 8192
STOP_WORDS = ["<end_of_turn>"]
REPETITION_PENALTY = 1.0

# FAQ HYPERPARAMETERS
FAQ_FILE = "data/hcmut_data_faq.csv"
FAQ_THRESHOLD = 80
FAQ_TEMPERATURE = 0.5
FAQ_TOP_P = 0.99
FAQ_TOP_K = 50
# FAQ_PROMPT = """<s> [INST] <<SYS>> Hãy viết lại câu trả lời dùng thông tin bên dưới. <</SYS>>
# Câu hỏi: {query}
# Trả lời: {answers[0]["answer"]}

# Câu trả lời được viết lại: [/INST]"""
# FAQ_PROMPT = """<s> <|im_start|>system
# You are a helpful assistant.</s><|im_start|>user
# Hãy viết lại câu trả lời dùng thông tin bên dưới.
# Câu hỏi: {query}
# Trả lời: {answers[0]["answer"]}</s><|im_start|>assistant
# """
FAQ_PROMPT = """<start_of_turn>user
Hãy viết lại câu trả lời theo một cách khác dùng thông tin bên dưới.
{query}
<end_of_turn>
<start_of_turn>model
Câu trả lời mới: """
FAQ_QUERY_TEMPLATE = """Câu hỏi: {query}
Trả lời: {answer}"""

# WEB HYPERPARAMETERS
WEB_FILE = "data/hcmut_data_web.csv"
WEB_THRESHOLD = 50
WEB_TEMPERATURE = 0.5
WEB_TOP_P = 0.99
WEB_TOP_K = 50
# WEB_PROMPT = """<s> [INST] <<SYS>> Trả lời câu hỏi bằng ngữ cảnh cho sẵn. <</SYS>>
# Ngữ cảnh: '''
# {join(documents, "\n")}
# '''
# Câu hỏi: {query}
# Trả lời: [/INST]"""
# WEB_PROMPT = """<s> <|im_start|>system
# You are a helpful assistant.</s><|im_start|>user
# Trả lời câu hỏi bằng ngữ cảnh cho sẵn.
# Ngữ cảnh: '''
# {join(documents, "\n")}
# '''
# Câu hỏi: {query}</s><|im_start|>assistant
# """
WEB_PROMPT = """<start_of_turn>user
Trả lời câu hỏi bằng ngữ cảnh cho sẵn.
Ngữ cảnh: '''
{join(documents, "\n")}
'''
Câu hỏi: {query}
<end_of_turn>
<start_of_turn>model
Trả lời: """

FREE_PROMPT = """<start_of_turn>user
{query}
<end_of_turn>
<start_of_turn>model
"""

# OTHERS
WARNING_NOTES = [
    "----------\nXin lưu ý rằng câu trả lời phía trên được sinh ra hoàn toàn từ mô hình ngôn ngữ lớn và không có trong dữ liệu của tôi. Do đó, câu trả lời có thể chứa thông tin sai sự thật hoặc lỗi thời.",
    "----------\nLưu ý: Câu trả lời có thể sai sự thật hoặc lỗi thời do không có trong dữ liệu của chúng tôi."
]

DEFAULT_ANSWERS = [
    "Xin lỗi! Tôi chưa có dữ liệu về câu hỏi bạn yêu cầu. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
    "Rất tiếc, tôi chưa có câu trả lời cho yêu cầu của bạn. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
    "Tôi chưa hiểu câu hỏi của bạn. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
]
