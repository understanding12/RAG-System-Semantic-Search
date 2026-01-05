"""
rag_interactive_real.py - НАСТОЯЩАЯ RAG СИСТЕМА
Генерирует ответы на основе найденных фрагментов
"""

print("=" * 70)
print("🤖 НАСТОЯЩАЯ RAG СИСТЕМА")
print("=" * 70)

# Загружаем данные
import pickle
import numpy as np

try:
    with open('rag_data_step3.pkl', 'rb') as f:
        rag_data = pickle.load(f)

    print("✅ База знаний загружена")

    chunks = rag_data["chunks"]
    chunk_info = rag_data["chunk_info"]
    embeddings = rag_data["embeddings"]

    print(f"📚 В базе: {len(chunks)} фрагментов")

except FileNotFoundError:
    print("❌ Файл данных не найден!")
    exit()


def calculate_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)


def ask_rag_real(question):
    """
    НАСТОЯЩАЯ RAG: генерирует ответ на основе найденных фрагментов
    """
    print(f"\n🔍 Ищу ответ на: '{question}'")

    # 1. ПОИСК (как раньше, но улучшенный)
    query_words = question.lower().split()
    query_embedding = np.zeros(embeddings.shape[1])
    matching_chunks = 0

    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in query_words):
            query_embedding += embeddings[i]
            matching_chunks += 1

    if matching_chunks > 0:
        query_embedding /= matching_chunks
    else:
        # Если нет совпадений слов, ищем по смыслу
        for i, chunk_embedding in enumerate(embeddings):
            query_embedding += chunk_embedding
        query_embedding /= len(embeddings)

    # Ищем похожие чанки
    similarities = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity = calculate_similarity(query_embedding, chunk_embedding)
        similarities.append((i, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:3]  # Берем 3 лучших

    print(f"📊 Найдено {len(top_results)} релевантных фрагментов:")
    for rank, (idx, similarity) in enumerate(top_results, 1):
        print(f"  #{rank} (сходство: {similarity:.3f})")
        print(f"    {chunks[idx][:80]}...")

    # 2. ФОРМИРУЕМ ПРОМПТ С НАЙДЕННЫМИ ФРАГМЕНТАМИ
    print("\n📝 Формирую промпт для генерации ответа...")

    # Собираем найденные фрагменты
    context_parts = []
    for rank, (idx, similarity) in enumerate(top_results, 1):
        context_parts.append(f"[Источник #{rank}, сходство: {similarity:.3f}]:")
        context_parts.append(chunks[idx])
        context_parts.append("")

    context = "\n".join(context_parts)

    # Промпт для "виртуальной LLM"
    prompt = f"""На основе предоставленных фрагментов текста ответь на вопрос.

ФРАГМЕНТЫ ТЕКСТА:
{context}

ВОПРОС: {question}

ИНСТРУКЦИИ:
1. Используй ТОЛЬКО информацию из фрагментов выше
2. Если информации недостаточно, скажи "На основе предоставленных фрагментов не могу дать полный ответ"
3. Будь точным и конкретным
4. Укажи, из каких источников (#номера) ты берешь информацию

ОТВЕТ:"""

    # 3. "ГЕНЕРАЦИЯ" ОТВЕТА (имитация LLM)
    # В реальности здесь был бы вызов ChatGPT:
    # import openai
    # response = openai.ChatCompletion.create(...)

    print("\n🤖 ГЕНЕРИРУЮ ОТВЕТ НА ОСНОВЕ НАЙДЕННЫХ ФРАГМЕНТОВ...")
    print("-" * 50)

    # Простая логика генерации на основе найденного
    answer_parts = []

    # Анализируем найденные фрагменты
    relevant_info = []
    sources_used = []

    for rank, (idx, similarity) in enumerate(top_results, 1):
        if similarity > 0.5:  # Порог релевантности
            chunk_text = chunks[idx]

            # Простой анализ: что в этом фрагменте полезного для ответа?
            if any(word in question.lower() for word in ["машинн", "ml", "machine"]):
                if "машинн" in chunk_text.lower():
                    relevant_info.append(chunk_text)
                    sources_used.append(rank)

            elif any(word in question.lower() for word in ["глубок", "deep"]):
                if "глубок" in chunk_text.lower():
                    relevant_info.append(chunk_text)
                    sources_used.append(rank)

            elif any(word in question.lower() for word in ["трансформер", "gpt", "bert"]):
                if any(word in chunk_text.lower() for word in ["трансформер", "gpt", "bert", "t5"]):
                    relevant_info.append(chunk_text)
                    sources_used.append(rank)

            elif any(word in question.lower() for word in ["rag", "retrieval"]):
                if "rag" in chunk_text.lower():
                    relevant_info.append(chunk_text)
                    sources_used.append(rank)
            else:
                # Для общего вопроса берем фрагмент с наибольшим сходством
                if rank == 1:
                    relevant_info.append(chunk_text)
                    sources_used.append(rank)

    # Формируем ответ
    if relevant_info:
        answer_parts.append("На основе найденных фрагментов информации:")
        answer_parts.append("")

        for i, info in enumerate(relevant_info[:2]):  # Берем не более 2 фрагментов
            # Упрощаем и структурируем информацию
            sentences = info.replace('\n', ' ').split('. ')
            key_sentences = [s for s in sentences if len(s) > 20][:3]  # Первые 3 осмысленных предложения

            answer_parts.append(f"{'. '.join(key_sentences)}.")
            answer_parts.append("")

        answer_parts.append(f"Источники: #{', #'.join(map(str, sources_used))}")
    else:
        answer_parts.append("На основе предоставленных фрагментов не нашел прямой информации для ответа на ваш вопрос.")
        answer_parts.append("Попробуйте переформулировать вопрос или задать о машинном обучении, глубоком обучении, трансформерах или RAG системах.")

    answer = "\n".join(answer_parts)
    print(answer)
    print("-" * 50)

    # 4. ПОКАЗЫВАЕМ, ЧТО ИСПОЛЬЗОВАЛОСЬ
    print("\n📚 ИСПОЛЬЗОВАННЫЕ ФРАГМЕНТЫ:")
    for rank, (idx, similarity) in enumerate(top_results, 1):
        if rank in sources_used:
            doc_title = chunk_info[idx]["doc_title"]
            print(f"  • {doc_title} (источник #{rank}, сходство: {similarity:.3f})")


# Интерактивный режим
print("""
🤖 НАСТОЯЩАЯ RAG СИСТЕМА

Отличие от предыдущей версии:
✅ Находит релевантные фрагменты
✅ Анализирует их содержание
✅ Формирует ответ НА ОСНОВЕ найденного
✅ Не использует готовые шаблоны

Попробуйте задать вопросы:
""")

while True:
    print("\n" + "=" * 50)
    question = input("\n❓ Ваш вопрос: ").strip()

    if question.lower() in ['выход', 'exit', 'quit', 'q']:
        print("\n👋 До свидания!")
        break

    if not question:
        continue

    try:
        ask_rag_real(question)
    except Exception as e:
        print(f"❌ Ошибка: {e}")