import inspect
import re
from dotenv import load_dotenv

load_dotenv()

from google import genai  # Google's unified Python SDK
from google.genai import types  # SDK configuration classes
from langsmith import traceable

# Initialize the GenAI client (reads GEMINI_API_KEY from environment)
client = genai.Client()

MAX_ITERATIONS = 10
MODEL = "gemini-3.6-flash" # Standard fast reasoning model for ReAct tasks


# --- Tools ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(
            tool_function, "__wrapped__", tool_function
        )
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# Traceable LLM wrapper targeting the new SDK structure
@traceable(name="Google GenAI Chat", run_type="llm")
def google_genai_chat_traced(
    model: str,
    contents: str,
    config: dict | None = None,  # Parameter name is 'config'
):
    return client.models.generate_content(
        model=model, contents=contents, config=config
    )

# --- Agent Loop ---


@traceable(name="Google GenAI Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        # Configure stop sequences and low temperature using GenAI config object
        config_obj = types.GenerateContentConfig(
            stop_sequences=["\nObservation"],
            temperature=0.0,
        )
        # Convert pydantic config model to dict to avoid LangSmith type check crashes
        config_dict = config_obj.model_dump(exclude_none=True)

        response = google_genai_chat_traced(
            model=MODEL,
            contents=full_prompt,
            config=config_dict,
        )

        # Access model generated text from response.text property
        output = response.text
        print(f"LLM Output:\n{output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        print(
            f"  [Parsing] Looking for Action and Action Input in LLM output..."
        )

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Google GenAI ReAct Agent!")
    print()
    result = run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )