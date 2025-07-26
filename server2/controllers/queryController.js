const { LLMAgent } = require("../services/llmService");

exports.processQuery = async (req, res) => {
  const { query, userId } = req.body;

  if (!query || !userId) {
    return res.status(400).json({ error: "Query and userId are required." });
  }

  try {
    const llmAgent = new LLMAgent(userId);
    const response = await llmAgent.runQuery(query);
    res.json({ answer: response.output });
  } catch (error) {
    console.error("Error processing query:", error);
    res.status(500).json({ error: "Internal server error." });
  }
};
