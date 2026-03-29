import { config } from './config';
import { Suggestion } from './types';

export interface EmbeddingResult {
  embedding: number[];
  model: string;
}

export async function getEmbedding(text: string): Promise<EmbeddingResult> {
  const response = await fetch(`${config.ollamaBaseUrl}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: config.embeddingModel,
      prompt: text,
    }),
  });

  if (!response.ok) {
    throw new Error(`Embedding API error: ${response.statusText}`);
  }

  return response.json();
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same dimension');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export interface IndexedRecord {
  suggestion: Suggestion;
  embedding: number[];
  text: string;
}

export class SuggestionIndex {
  private records: IndexedRecord[] = [];

  async buildIndex(suggestions: Suggestion[]): Promise<void> {
    console.log(`Building index for ${suggestions.length} records...`);

    for (const suggestion of suggestions) {
      const text = `${suggestion.Description} ${suggestion.Suggestion} ${suggestion.Gid}`;
      const { embedding } = await getEmbedding(text);

      this.records.push({
        suggestion,
        embedding,
        text,
      });

      console.log(`Indexed: ${suggestion.Gid} - ${suggestion.Name}`);
    }

    console.log('Index built successfully!');
  }

  search(query: string, topK: number = config.topK): Array<{
    suggestion: Suggestion;
    similarity: number;
  }> {
    return this.records
      .map((record) => ({
        suggestion: record.suggestion,
        similarity: cosineSimilarity(this.lastQueryEmbedding, record.embedding),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  async searchAsync(query: string, topK: number = config.topK): Promise<Array<{
    suggestion: Suggestion;
    similarity: number;
  }>> {
    const { embedding: queryEmbedding } = await getEmbedding(query);
    this.lastQueryEmbedding = queryEmbedding;

    return this.records
      .map((record) => ({
        suggestion: record.suggestion,
        similarity: cosineSimilarity(queryEmbedding, record.embedding),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  private lastQueryEmbedding: number[] = [];
}
