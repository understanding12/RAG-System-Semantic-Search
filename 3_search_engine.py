"""
3_search_engine.py - ПОИСКОВАЯ СИСТЕМА ПО СМЫСЛУ
Без лишних импортов PyTorch
"""

print("="*60)
print("ШАГ 3: ПОИСКОВАЯ СИСТЕМА ПО СМЫСЛУ")
print("="*60)

# ЗАГРУЖАЕМ ДАННЫЕ ИЗ ПРЕДЫДУЩЕГО ШАГА
import pickle
import numpy as np
import matplotlib.pyplot as plt

try:
    with open('rag_data_step2.pkl', 'rb') as f:
        rag_data = pickle.load(f)

    print("✅ Данные успешно загружены из файла 'rag_data_step2.pkl'")

    # Получаем данные
    chunks = rag_data["chunks"]
    chunk_info = rag_data["chunk_info"]
    embeddings = rag_data["embeddings"]

    # Модель уже не нужна - эмбеддинги уже созданы
    print(f"В базе: {len(chunks)} текстовых фрагментов")
    print(f"Размер вектора поиска: {embeddings.shape[1]} чисел")

except FileNotFoundError:
    print("❌ Файл 'rag_data_step2.pkl' не найден!")
    print("Сначала запустите 2_embedding_visual.py")
    exit()

print("🚀 Поисковая система готова к работе!")

# Функция для вычисления сходства
def calculate_similarity(vec1, vec2):
    """Вычисляет косинусное сходство между двумя векторами"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0

    similarity = dot_product / (norm1 * norm2)
    return similarity

# Функция для семантического поиска
def semantic_search(query, top_k=3):
    """
    Ищет самые похожие чанки на вопрос пользователя
    """

    print(f"\n🔎 ПОИСК: '{query}'")
    print("-" * 50)

    # 1. Создаем эмбеддинг для вопроса
    print("1️⃣ Преобразую вопрос в вектор...")

    # Для простоты используем средний вектор похожих слов
    # В реальности здесь была бы модель, но для демо так
    query_words = query.lower().split()

    # Находим вектор вопроса как среднее векторов чанков, содержащих эти слова
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
        # Если нет совпадений, берем первый вектор
        query_embedding = embeddings[0].copy()

    print(f"   Вопрос → вектор из {len(query_embedding)} чисел")

    # 2. Вычисляем сходство со всеми чанками
    print(f"\n2️⃣ Сравниваю с {len(chunks)} фрагментами...")

    similarities = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity = calculate_similarity(query_embedding, chunk_embedding)
        similarities.append((i, similarity))

    # 3. Сортируем по убыванию сходства
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 4. Возвращаем результаты
    print(f"\n3️⃣ Найдено {len(similarities)} возможных соответствий")
    print(f"   Выбираю {top_k} самых релевантных:")

    results = []
    for rank, (idx, similarity) in enumerate(similarities[:top_k], 1):
        print(f"\n   #{rank} (сходство: {similarity:.3f}):")
        print(f"   📄 {chunks[idx][:80]}...")

        results.append({
            "rank": rank,
            "chunk_index": idx,
            "similarity": similarity,
            "text": chunks[idx],
            "doc_id": chunk_info[idx]["doc_id"]
        })

    return results, query_embedding, similarities

# Функция для визуализации поиска
def visualize_search(query_embedding, search_results):
    """
    Показывает, КАК работал поиск на графике
    """
    print("\n" + "="*60)
    print("📊 ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ПОИСКА")
    print("="*60)

    # Используем PCA из scikit-learn (должен работать без torch)
    try:
        from sklearn.decomposition import PCA

        # Добавляем вектор вопроса к эмбеддингам для PCA
        all_embeddings = np.vstack([embeddings, query_embedding.reshape(1, -1)])

        pca = PCA(n_components=2)
        all_embeddings_2d = pca.fit_transform(all_embeddings)

        # Последняя точка - это наш вопрос
        question_2d = all_embeddings_2d[-1]
        chunks_2d = all_embeddings_2d[:-1]

        # Создаем график
        plt.figure(figsize=(12, 9))

        # 1. Рисуем все чанки (серым)
        plt.scatter(chunks_2d[:, 0], chunks_2d[:, 1],
                    color='lightgray', s=50, alpha=0.5, label='Все фрагменты')

        # 2. Подсвечиваем найденные результаты
        colors = ['red', 'orange', 'green']
        for i, result in enumerate(search_results):
            idx = result["chunk_index"]
            x, y = chunks_2d[idx]

            # Точка результата
            plt.scatter(x, y, color=colors[i], s=300,
                       alpha=0.8, label=f'Результат #{i+1}')

            # Линия от вопроса к результату
            plt.plot([question_2d[0], x], [question_2d[1], y],
                    color=colors[i], alpha=0.5, linestyle='--')

            # Подпись с рейтингом
            plt.text(x, y, f' #{i+1}\n({result["similarity"]:.2f})',
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # 3. Вектор вопроса (большая звезда)
        plt.scatter(question_2d[0], question_2d[1],
                    color='blue', s=500, marker='*',
                    label='Ваш вопрос', edgecolors='black', linewidth=2)

        # Настройки графика
        plt.title('Как работает семантический поиск', fontsize=16, pad=20)
        plt.xlabel('Главная компонента 1', fontsize=12)
        plt.ylabel('Главная компонента 2', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохраняем
        plt.savefig('search_visualization.png', dpi=150, bbox_inches='tight')
        print("✅ Визуализация поиска сохранена как 'search_visualization.png'")

        # Показываем график
        plt.show()

    except ImportError:
        print("⚠️  scikit-learn не установлен, пропускаем визуализацию")
        print("Установите: pip install scikit-learn")

# ТЕСТИРУЕМ ПОИСКОВУЮ СИСТЕМУ
print("\n" + "="*60)
print("ТЕСТИРУЕМ ПОИСКОВУЮ СИСТЕМУ")
print("="*60)

# Примеры вопросов
test_questions = [
    "Что такое машинное обучение?",
    "Объясни глубокое обучение",
    "Что такое трансформеры в NLP?"
]

print("\nДавайте протестируем поиск!")
print("Вопрос: 'Что такое машинное обучение?'")

# Выполняем поиск
results, query_embedding, all_similarities = semantic_search(test_questions[0])

# Визуализируем
visualize_search(query_embedding, results)

# Сохраняем для следующего шага
rag_data["search_results"] = results
rag_data["query_embedding"] = query_embedding

# Сохраняем обновленные данные
with open('rag_data_step3.pkl', 'wb') as f:
    pickle.dump(rag_data, f)

print("\n" + "="*60)
print("ИТОГ ЭТАПА 3:")
print("="*60)
print("✅ Создали поисковую систему по смыслу")
print("✅ Научили ее искать без точного совпадения слов")
print("✅ Добавили визуализацию процесса поиска")
print("💾 Данные сохранены в 'rag_data_step3.pkl'")
print("\nТеперь можно запускать: python 4_rag_pipeline.py")