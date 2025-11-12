import json
import os

from datasets import Dataset
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from api.agent import WarAndPeaceAgent

load_dotenv()
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
OLLAMA_BASE_URL = f'http://{OLLAMA_HOST}:{OLLAMA_PORT}'

llm = ChatOllama(
    model=os.getenv('EVALUATE_LLM', 'llama3.1:8b'),
    base_url=OLLAMA_BASE_URL,
)
embedder = OllamaEmbeddings(
    model=os.getenv('EMBEDDING_MODEL', 'bge-m3:567m'),
    base_url=OLLAMA_BASE_URL
)


def run_ragas_evaluation():
    agent = WarAndPeaceAgent()

    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'dataset.json')

    with open(dataset_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    questions = []
    answers = []
    contexts = []

    print('*** Запуск оценки "RAGAS" ***')
    for item in questions_data:
        query = item['question']
        answer = agent.invoke(query=query)
        tool = agent.tool_instance
        context = tool.get_last_context()
        questions.append(query)
        answers.append(answer)
        contexts.append([context] if context.strip() else [''])

    dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts
    })

    result = evaluate(
        dataset=dataset,
        embeddings=embedder,
        metrics=[
            faithfulness,
            answer_relevancy,
            # context_recall
        ],
        llm=llm,
    )
    print('Результаты оценки:')
    print(result)

    result_dict = result.to_pandas().to_dict()
    result_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f'Результаты сохранены в: {result_path}')
    return result


if __name__ == '__main__':
    run_ragas_evaluation()
