import unicodedata
import re
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = SnowballStemmer("russian")

from logger_config import setup_logger
logger = setup_logger(__name__)


def clean_content_noise(text: str) -> str:
    """Очистка специфического мусора (точки, лишние пробелы)"""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\uf0b7\uf02d\u25cf\u2713]', '-', text)
    text = re.sub(r'[\uE000-\uF8FF]', ' ', text)    
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\|', '\n|', text)
    
    return text.strip()



def is_junk_text(text: str) -> bool:
    """Проверяет, является ли текст мусором (например, просто список цифр страниц)"""
    clean = text.replace('Раздел:', '').strip()
    if re.fullmatch(r'[\d\s\.]+', clean):
        return True
    return False



def normalize_for_fts(text: str) -> str:

    """Нормализация текста для полнотекстового поиска (FTS)"""
    if not text:
        return ""
    
    # К нижнему регистру
    text = text.lower() 
    # Оставляем только буквы и цифры
    words = re.findall(r'[а-яёa-z0-9]+', text)    
    # Применяем стемминг (обрезаем окончания)
    stemmed_words = [stemmer.stem(word) for word in words]    
    return " ".join(stemmed_words)




def prepare_query_from_natural_language(question):
    """
    Превращает человеческий вопрос в запрос для FTS5.
    """
    custom_stops = {'какие', 'какой', 'какая', 'как', 'почему', 'сколько', 'когда', 'где'}
    stop_words = set(stopwords.words('russian'))
    stop_words.update(custom_stops) 

    tokens = re.findall(r'[а-яёa-z0-9]+', question.lower())
    
    valid_stems = []
    for word in tokens:
        if word not in stop_words:
            stem = stemmer.stem(word)
            valid_stems.append(stem)
    
    return valid_stems


rules = {
    "e5": {"PASSAGE": "passage: ", "QUERY": "query: "},
    "giga": {"PASSAGE": "", "QUERY": "Ищу: "},
    "user2": {"PASSAGE": "search_document: ", "QUERY": "search_query: "}
}

def format_text(text, model_type, query_type):
    """
    text: сам текст чанка или вопроса
    model_type: 'e5', 'giga' или 'user2'
    query_type: 'PASSAGE' (для бд) или 'QUERY' (для поиска)
    """
    try:
        prefix = rules[model_type][query_type]
        return f"{prefix}{text}"
    except KeyError:
        logger.error("ВНИМАНИЕ! Не поддерживаемая модель или формат!")
        return text
        
    