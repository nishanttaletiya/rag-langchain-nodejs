const axios = require("axios");
const { Pinecone } = require("@pinecone-database/pinecone");
require("dotenv").config();

const DB_INDEX = "rag-langchain-nodejs-new";
const NAMESPACE = process.env.NAMESPACE_NAME;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY_ENV;

const pc = new Pinecone({ apiKey: process.env.PINECONE_KEY });

async function createEmbeddings(text) {
  try {
    const response = await axios.post(
      "https://api.openai.com/v1/embeddings",
      {
        model: "text-embedding-ada-002",
        input: text,
      },
      {
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );
    return response.data.data[0].embedding;
  } catch (error) {
    console.error("Error creating embeddings:", error.message);
    throw error;
  }
}

async function storeEmbeddings(embeddings, namespace = NAMESPACE) {
  const index = pc.index(DB_INDEX);

  try {
    const upsertPromises = embeddings.map((embeddingData, i) => {
      return index.namespace(namespace).upsert([
        {
          id: `chunk-${Date.now()}-${i}`, // Unique ID to avoid conflicts
          values: embeddingData.embedding,
          metadata: { chunk: embeddingData.chunk },
        },
      ]);
    });

    await Promise.all(upsertPromises);
    console.log("New embeddings stored successfully.");
  } catch (error) {
    console.error("Error storing embeddings:", error.message);
  }
}

const createIndex = async () => {
  try {
    await pc.createIndex({
      name: DB_INDEX,
      dimension: 1536, // Correct dimension for the OpenAI model
      metric: "cosine",
      spec: {
        serverless: {
          cloud: "aws",
          region: "us-east-1",
        },
      },
    });
    console.log("Index created:", DB_INDEX);
  } catch (error) {
    console.error("Error creating index:", error.message);
  }
};

const deleteIndex = async () => {
  try {
    await pc.deleteIndex({ name: DB_INDEX });
    console.log("Index deleted:", DB_INDEX);
  } catch (error) {
    console.error("Error deleting index:", error.message);
  }
};

async function checkIndexExists() {
  try {
    const response = await pc.listIndexes();
    const indexes = response.indexes;

    return indexes.find((item) => item.name === DB_INDEX);
  } catch (error) {
    console.error("Error checking index existence:", error.message);
    return null;
  }
}

async function retrieveRelevantChunks(query, namespace = NAMESPACE) {
  try {
    const queryEmbedding = await createEmbeddings(query); // Generate embedding for the query
    const index = pc.index(DB_INDEX);

    // Limit to top 5 results, but you can change the topK to a lower value if needed
    const results = await index.namespace(namespace).query({
      vector: queryEmbedding,
      topK: 5, // Fetch top 5 relevant chunks
      includeValues: true,
      includeMetadata: true,
    });

    // Extract only relevant chunks and limit length of content
    const relevantChunks = Array.from(
      new Set(results.matches.map((match) => match.metadata.chunk))
    );

    // Limit the size of the total content sent (let's assume max 1000 tokens total)
    let selectedChunks = [];
    let tokenCount = 0;

    // Add chunks until we reach the token limit
    for (let chunk of relevantChunks) {
      const chunkTokenCount = chunk.split(" ").length; // Simple word count approximation for tokens

      // If adding this chunk exceeds token limit, break
      if (tokenCount + chunkTokenCount > 1000) {
        break;
      }

      selectedChunks.push(chunk);
      tokenCount += chunkTokenCount;
    }

    console.log("Relevant Chunks Fetched:", selectedChunks); // Debug: Log the relevant chunks

    // Limit to top 3 relevant chunks
    return selectedChunks.slice(0, 3); // Limit to top 3 chunks
  } catch (error) {
    console.error("Error retrieving relevant chunks:", error.message);
    return []; // Return an empty array if there's an error
  }
}

module.exports = {
  storeEmbeddings,
  createIndex,
  retrieveRelevantChunks,
  checkIndexExists,
  deleteIndex,
};
