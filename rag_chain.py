import os, json, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

from dotenv import load_dotenv
load_dotenv()
class MeetingRAG:
    def __init__(self, index_dir='data/index', api_key=None, use_openrouter=True):
        self.index_dir = index_dir
        emb_path = os.path.join(index_dir, 'embeddings.npy')
        chunks_path = os.path.join(index_dir, 'chunks.json')
        if not os.path.exists(emb_path) or not os.path.exists(chunks_path):
            raise RuntimeError('Index not found. Run chunk_and_embed.py first')

        self.emb = np.load(emb_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self.embed_model = None

        # Key + mode
        self.use_openrouter = use_openrouter
        if self.use_openrouter:
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def retrieve(self, query, topk=5):
        if self.embed_model is None:
            raise RuntimeError('Embedding model not available for query embedding')
        q_emb = self.embed_model.encode([query])[0]
        sims = cosine_similarity([q_emb], self.emb)[0]
        idx = sims.argsort()[::-1][:topk]
        return [(i, float(sims[i]), self.chunks[i]) for i in idx]

    def _call_model(self, prompt):
        if not self.api_key:
            raise RuntimeError('API key not set.')

        if self.use_openrouter:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            resp = client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",   # change to any OpenRouter model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            return resp.choices[0].message.content

        else:
            import openai
            openai.api_key = self.api_key
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            return resp['choices'][0]['message']['content']

    def summarize(self, topk=6):
        combined = ' '.join(self.chunks[:min(len(self.chunks), topk)])
        prompt = f"Summarize the following meeting transcript passages into a concise 6-bullet summary: \n\n{combined}"
        return self._call_model(prompt)

    def extract_actions(self, topk=8):
        combined = ' '.join(self.chunks[:min(len(self.chunks), topk)])
        prompt = f"Extract action items from the passages below. Return JSON array with fields: task, owner (if present), due (if present), evidence. Passages:\n\n{combined}"
        return self._call_model(prompt)

    def answer(self, question, topk=6):
        retrieved = self.retrieve(question, topk=topk)
        passages = '\n\n'.join([f'[{i}] {p}' for i, _, p in retrieved])
        prompt = f"Use only the passages below to answer the question. Provide a short answer (1-2 sentences) and list the passage IDs you used.\n\nQuestion: {question}\n\nPassages:\n{passages}"
        return self._call_model(prompt)


# --- CLI wrapper ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='data/index', help='Path to embeddings index')
    parser.add_argument('--question', help='Question to ask the meeting')
    parser.add_argument('--summary', action='store_true', help='Generate summary')
    parser.add_argument('--actions', action='store_true', help='Extract action items')
    parser.add_argument('--use-openai', action='store_true', help='Force using OpenAI instead of OpenRouter')
    args = parser.parse_args()

    rag = MeetingRAG(index_dir=args.index, use_openrouter=not args.use_openai)

    if args.summary:
        print("=== Summary ===")
        print(rag.summarize())
    if args.actions:
        print("=== Action Items ===")
        print(rag.extract_actions())
    if args.question:
        print("=== Answer ===")
        print(rag.answer(args.question))
