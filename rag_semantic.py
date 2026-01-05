"""
rag_semantic.py - НАСТОЯЩИЙ СЕМАНТИЧЕСКИЙ ПОИСК
Использует одну модель для векторизации вопросов и документов
"""

print("=" * 70)
print("🤖 RAG С СЕМАНТИЧЕСКИМ ПОИСКОМ")
print("=" * 70)

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings('ignore')

# Загружаем данные
try:
    with open('rag_data_step2.pkl', 'rb') as f:
        rag_data = pickle.load(f)

    print("✅ База знаний загружена")

    chunks = rag_data["chunks"]
    chunk_info = rag_data["chunk_info"]
    chunk_embeddings = rag_data["embeddings"]

    print(f"📚 В базе: {len(chunks)} фрагментов")
    print(f"📏 Размерность эмбеддингов: {chunk_embeddings.shape[1]}")

except FileNotFoundError:
    print("❌ Файл данных не найден!")
    print("Сначала запустите:")
    print("  python 1_data_preparation.py")
    print("  python 2_embedding_visual.py")
    exit()

# Загружаем модель для векторизации ВОПРОСОВ
# ТА ЖЕ модель, что использовалась для документов!
print("\n🧠 Загружаю модель для семантического поиска...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Модель готова!")


def semantic_search(question, top_k=3):
    """
    НАСТОЯЩИЙ семантический поиск:
    1. Векторизуем вопрос
    2. Ищем ближайшие векторы документов
    3. Возвращаем результаты
    """

    print(f"\n🔍 Вопрос: '{question}'")
    print("   ↓")
    print("   📊 Преобразую в вектор...")

    # 1. ВЕКТОРИЗАЦИЯ ВОПРОСА
    question_embedding = model.encode(question, convert_to_tensor=True)

    # 2. ПОИСК БЛИЖАЙШИХ ВЕКТОРОВ
    print("   📊 Ищу похожие фрагменты...")

    # Используем косинусное сходство
    cosine_scores = util.cos_sim(question_embedding, chunk_embeddings)[0]

    # Получаем топ-K результатов
    top_results = np.argsort(cosine_scores.numpy())[::-1][:top_k]

    print(f"   ✅ Найдено {len(top_results)} релевантных фрагментов")

    # 3. ФОРМИРУЕМ РЕЗУЛЬТАТЫ
    results = []
    for rank, idx in enumerate(top_results, 1):
        similarity = cosine_scores[idx].item()

        print(f"\n   #{rank} (сходство: {similarity:.3f}):")
        print(f"   📄 {chunks[idx][:100]}...")

        results.append({
            "rank": rank,
            "chunk_index": idx,
            "similarity": similarity,
            "text": chunks[idx],
            "doc_id": chunk_info[idx]["doc_id"]
        })

    return results, question_embedding


def generate_answer_from_context(question, search_results):
    """
    Генерирует ответ на основе найденных фрагментов
    (В реальности здесь был бы вызов ChatGPT)
    """

    print("\n🧠 Формирую ответ на основе найденного...")

    # Собираем контекст из найденных фрагментов
    context_parts = []
    for result in search_results:
        context_parts.append(f"[Фрагмент #{result['rank']} (сходство: {result['similarity']:.3f})]:")
        context_parts.append(result['text'])
        context_parts.append("")

    context = "\n".join(context_parts[:500])  # Ограничиваем длину

    # Анализируем, что нашли
    print("\n📋 АНАЛИЗ НАЙДЕННОГО:")

    # Определяем тему вопроса
    question_lower = question.lower()

    # Простая логика для демонстрации
    if any(word in question_lower for word in ["машинн", "ml", "machine learning", "искусственный интеллект"]):
        topic = "машинное обучение"
    elif any(word in question_lower for word in ["глубок", "deep learning", "нейросеть", "нейронная"]):
        topic = "глубокое обучение"
    elif any(word in question_lower for word in ["трансформер", "gpt", "bert", "llm", "языковая модель"]):
        topic = "трансформеры"
    elif any(word in question_lower for word in ["rag", "retrieval", "поиск и генерация"]):
        topic = "RAG"
    else:
        topic = "технологии ИИ"

    print(f"   Тема вопроса: {topic}")

    # Формируем ответ на основе контекста
    print("\n🤖 ОТВЕТ:")
    print("-" * 50)

    if search_results:
        # Берем самый релевантный фрагмент и дополняем
        best_result = search_results[0]
        best_text = best_result["text"]

        # Упрощаем и структурируем
        sentences = best_text.replace('\n', ' ').split('. ')
        if sentences:
            # Берем ключевые предложения
            key_sentences = []
            for sentence in sentences:
                if len(sentence) > 20:  # Не слишком короткие
                    if topic.lower() in sentence.lower():
                        key_sentences.append(sentence)

            if not key_sentences:
                key_sentences = sentences[:3]  # Первые 3 предложения

            answer = f"На основе найденной информации о {topic}:\n\n"
            answer += ". ".join(key_sentences[:3]) + ".\n\n"

            # Добавляем дополнительные детали из других фрагментов
            if len(search_results) > 1:
                second_result = search_results[1]
                if second_result["similarity"] > 0.6:
                    second_sentences = second_result["text"].replace('\n', ' ').split('. ')
                    if second_sentences:
                        answer += f"Также:\n{second_sentences[0]}.\n"

            answer += f"\n📚 Источники: фрагменты #{', #'.join(str(r['rank']) for r in search_results[:2])}"

            print(answer)
        else:
            print("Нашел информацию, но не смог сформировать связный ответ.")
    else:
        print("К сожалению, не нашел релевантной информации в базе знаний.")

    print("-" * 50)

    return context


# ТЕСТИРОВАНИЕ СЕМАНТИЧЕСКОГО ПОИСКА
print("\n" + "=" * 70)
print("🧪 ТЕСТИРУЕМ СЕМАНТИЧЕСКИЙ ПОИСК")
print("=" * 70)

test_questions = [
    "Что такое машинное обучение?",
    "Объясни глубокое обучение",
    "Что такое трансформеры?",
    "Как работает RAG система?",
    # Семантические тесты:
    "Нейросети для обработки текста",
    "Модели с механизмом внимания",
    "Обучение на размеченных данных",
    "Автоматическое извлечение признаков из данных"
]

print("\n📋 ТЕСТОВЫЕ ВОПРОСЫ:")
for i, q in enumerate(test_questions, 1):
    print(f"{i}. {q}")

print("\n🚀 Запускаю тесты семантического поиска...")

for i, question in enumerate(test_questions[:4], 1):  # Первые 4 вопроса
    print(f"\n{'=' * 70}")
    print(f"ТЕСТ {i}: '{question}'")
    print('=' * 70)

    # Семантический поиск
    results, question_embedding = semantic_search(question)

    # Генерация ответа
    context = generate_answer_from_context(question, results)

print("\n" + "=" * 70)
print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ С СЕМАНТИЧЕСКИМ ПОИСКОМ")
print("=" * 70)

print("""
Теперь вы можете задавать вопросы ЛЮБЫМИ словами!
Система ищет по СМЫСЛУ, а не по точным словам.

Примеры вопросов, которые теперь должны работать:
• "Нейросети для текста" → найдет про трансформеры
• "Модели с вниманием" → найдет про трансформеры  
• "Обучение на примерах" → найдет про supervised learning
• "Извлечение признаков" → найдет про глубокое обучение

Команды:
• 'выход' - завершить
• 'тест' - запустить тесты
• 'стат' - статистика базы
""")

# ИНТЕРАКТИВНЫЙ РЕЖИМ
while True:
    print("\n" + "=" * 50)
    question = input("\n❓ Ваш вопрос: ").strip()

    if question.lower() in ['выход', 'exit', 'quit', 'q']:
        print("\n👋 До свидания!")
        break

    if question.lower() in ['тест', 'test']:
        print("\n🧪 Запускаю полное тестирование...")
        for q in test_questions:
            print(f"\n--- {q} ---")
            results, _ = semantic_search(q, top_k=2)
        continue

    if question.lower() in ['стат', 'stats', 'статистика']:
        print(f"\n📊 СТАТИСТИКА БАЗЫ ЗНАНИЙ:")
        print(f"• Фрагментов: {len(chunks)}")
        print(f"• Размерность векторов: {chunk_embeddings.shape[1]}")
        print(f"• Средняя длина фрагмента: {np.mean([len(c) for c in chunks]):.0f} символов")

        # Распределение по темам
        topics = {"ML": 0, "DL": 0, "Трансформеры": 0, "RAG": 0, "Другое": 0}
        for chunk in chunks:
            chunk_lower = chunk.lower()
            if any(word in chunk_lower for word in ["машинн", "ml", "machine"]):
                topics["ML"] += 1
            elif any(word in chunk_lower for word in ["глубок", "deep", "нейрон"]):
                topics["DL"] += 1
            elif any(word in chunk_lower for word in ["трансформер", "gpt", "bert"]):
                topics["Трансформеры"] += 1
            elif any(word in chunk_lower for word in ["rag", "retrieval"]):
                topics["RAG"] += 1
            else:
                topics["Другое"] += 1

        print("\n📚 Распределение по темам:")
        for topic, count in topics.items():
            if count > 0:
                print(f"  {topic}: {count} фрагментов")
        continue

    if not question:
        continue

    # Обработка вопроса
    try:
        results, _ = semantic_search(question)
        context = generate_answer_from_context(question, results)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("Попробуйте другой вопрос")

print("\n" + "=" * 70)
print("✅ СЕМАНТИЧЕСКИЙ ПОИСК РЕАЛИЗОВАН!")
print("=" * 70)