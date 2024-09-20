const fs = require("fs");
const pdf = require("pdf-parse");
const axios = require("axios");
require("dotenv").config();
const {
  storeEmbeddings,
  createIndex,
  checkIndexExists,
} = require("./pinecone-helpers");

const OPENAI_API_KEY = process.env.OPENAI_API_KEY_ENV;

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

async function extractTextFromPDF(filePath) {
  const dataBuffer = fs.readFileSync(filePath);
  const data = await pdf(dataBuffer);
  return data.text;
}

async function runApp() {
  try {
    const indexExists = await checkIndexExists();
    if (!indexExists) {
      await createIndex(); // Create index if it doesn't exist
    }

    const pdfText = await extractTextFromPDF("./file/mscict.pdf");
    const embeddings = await createEmbeddings(pdfText);

    const embeddingData = [
      {
        embedding: embeddings,
        chunk: pdfText,
      },
    ];
    await storeEmbeddings(embeddingData);

    console.log("Embeddings stored successfully.");
  } catch (error) {
    console.error("Error running the application:", error.message);
  }
}

runApp();
