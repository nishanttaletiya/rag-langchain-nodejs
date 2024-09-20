const axios = require("axios");
const { retrieveRelevantChunks } = require("./pinecone-helpers");
require("dotenv").config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY_ENV;

async function generateResponse(query) {
  try {
    // Retrieve relevant chunks from Pinecone
    const relevantChunks = await retrieveRelevantChunks(query);

    // Ensure we have relevant chunks
    if (!Array.isArray(relevantChunks) || relevantChunks.length === 0) {
      throw new Error("No relevant chunks found");
    }

    // Combine relevant chunks into a single context
    const context = relevantChunks.join(" ").substring(0, 3000); // Limit context length

    console.log("Context to be Sent to OpenAI:", context); // Debug log

    // Prepare the data to be sent to OpenAI
    const dataToSend = {
      model: "gpt-3.5-turbo", // Use gpt-3.5-turbo or gpt-4
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant.",
        },
        {
          role: "user",
          content: `Based on the following information, answer the query: "${query}"\n\nContext: ${context}\n\nAnswer:`,
        },
      ],
      max_tokens: 150,
      temperature: 0.7,
    };

    console.log(
      "Data sent to OpenAI API:",
      JSON.stringify(dataToSend, null, 2)
    ); // Log the data sent

    // Generate a response using OpenAI
    const response = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      dataToSend,
      {
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );

    return response.data.choices[0].message.content.trim();
  } catch (error) {
    console.error(
      "Error generating response:",
      error.response?.data || error.message
    );
    throw error;
  }
}

// Example usage:
async function runApp() {
  try {
    const query = "Who is nishant?"; // Example query
    const response = await generateResponse(query);
    console.log("Response:", response);
  } catch (error) {
    console.error("Error running the application:", error.message);
  }
}

runApp();
