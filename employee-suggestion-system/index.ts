import { loadSuggestionData, getOutputFields } from './data-loader';
import { SuggestionIndex } from './embeddings';
import { OutputFields, PredictionResult } from './types';

let index: SuggestionIndex | null = null;

export async function buildIndex(): Promise<void> {
  const suggestions = await loadSuggestionData();
  index = new SuggestionIndex();
  await index.buildIndex(suggestions);
  console.log('Index ready!');
}

export async function predict(
  description: string,
  suggestion: string,
  gid: string
): Promise<PredictionResult> {
  if (!index) {
    await buildIndex();
  }

  const query = `${description} ${suggestion} ${gid}`;
  const results = await index!.searchAsync(query, config.topK);

  if (results.length === 0) {
    throw new Error('No similar records found');
  }

  console.log('\n=== Top matches ===');
  for (const r of results) {
    console.log(`Similarity: ${r.similarity.toFixed(3)} - ${r.suggestion.Name}`);
  }

  const topResult = results[0];
  
  const output = getOutputFields(topResult.suggestion);

  return output as PredictionResult;
}

import { config } from './config';

if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args[0] === 'index') {
    buildIndex().then(() => process.exit(0));
  } else if (args[0] === 'predict') {
    const description = args[1] || '';
    const suggestion = args[2] || '';
    const gid = args[3] || '';
    
    predict(description, suggestion, gid).then(result => {
      console.log('\n=== Prediction Result ===');
      console.log(JSON.stringify(result, null, 2));
    }).catch(err => {
      console.error(err);
      process.exit(1);
    });
  } else {
    console.log('Usage:');
    console.log('  npm run index    - Build the search index');
    console.log('  npm run predict  - Predict output fields');
  }
}
