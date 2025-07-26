const { ChatOpenAI } = require("@langchain/openai");
const { AgentExecutor, createReactAgent } = require("langchain/agents");
const { PromptTemplate } = require("@langchain/core/prompts");
const { Tool } = require("@langchain/core/tools");
const { z } = require("zod");

// Arcade Tool as a LangChain Tool
class ArcadeToolWrapper extends Tool {
  name = "ArcadeTool";
  description = "Tool for interacting with Arcade.dev services. Input should be a string in the format \"toolName::toolInputJsonString\".";
  schema = z.object({
    toolName: z.string().describe("The name of the Arcade tool to execute."),
    toolInput: z.record(z.any()).describe("A JSON string representing the input for the Arcade tool."),
  });

  constructor(userId) {
    super();
    this.arcadeClient = new ArcadeTool(userId); // Assuming ArcadeTool is your original class
  }

  async _call(input) {
    // The input to _call is already parsed by the schema
    return await this.arcadeClient.executeTool(input.toolName, input.toolInput);
  }
}

// Placeholder for original Arcade Tool logic (if it's not a LangChain Tool itself)
class ArcadeTool {
  constructor(userId) {
    this.userId = userId;
  }

  async executeTool(toolName, toolInput) {
    // In a real scenario, you would integrate with the Arcade API here.
    // For now, let's simulate a response.
    console.log(`Executing Arcade tool: ${toolName} with input:`, toolInput);
    if (toolName === "Math.Sqrt") {
      const num = parseInt(toolInput.a);
      return { value: Math.sqrt(num) };
    }
    return { value: `Simulated response for ${toolName}` };
  }
}

class LLMAgent {
  constructor(userId) {
    this.llm = new ChatOpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
      temperature: 0,
    });
    
    this.tools = [
      new ArcadeToolWrapper(userId),
    ];

    this.prompt = PromptTemplate.fromTemplate(
      `You are a helpful AI assistant. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}`
    );

    // createReactAgent returns a Runnable, which is what AgentExecutor expects
    this.agent = createReactAgent({
      llm: this.llm,
      tools: this.tools,
      prompt: this.prompt,
    });
    
    this.agentExecutor = AgentExecutor.fromAgentAndTools({
      agent: this.agent,
      tools: this.tools,
      verbose: true,
    });
  }

  async runQuery(query) {
    return await this.agentExecutor.invoke({ input: query });
  }
}

module.exports = { LLMAgent };