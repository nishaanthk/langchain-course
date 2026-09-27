#Not using langchain makes you write lot of boilerplate code.
#1. Define tools
##2. Describe tools to the LLM
#3. Send user message
#4. Inspect tool calls
#5. Identify the requested tool
#6. Execute the tool
#7. Add the tool result to history
#8. Send history back to the LLM
#9. Repeat until no tool is requested
#10. Return final answer

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from langsmith import traceable


# ============================================================
# CONFIGURATION
# ============================================================

MAX_ITERATIONS = 10
MODEL = "gemini-3.6-flash"


# ============================================================
# TOOLS
# ============================================================

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price in the product catalog."""

    print(f" >> executing get_product_price(product='{product}')")

    prices = {
        "laptop": 1200.10,
        "mobile": 800.25,
        "headset": 500.56,
    }

    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a product price."""

    print(
        f" >> executing apply_discount("
        f"price='{price}', discount='{discount_tier}')"
    )

    discount_percentage = {
        "bronze": 5,
        "silver": 10,
        "gold": 15,
    }

    discount = discount_percentage.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


# ============================================================
# FUNCTION DECLARATIONS
# ============================================================

tools_for_llm = [

    {
        "name": "get_product_price",
        "description": "Look up the price of a product in the catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": (
                        "The product name, e.g. "
                        "'laptop', 'mobile', or 'headset'"
                    ),
                }
            },
            "required": ["product"],
        },
    },

    {
        "name": "apply_discount",
        "description": (
            "Apply a discount tier to a price and return "
            "the final price."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "description": "The original price.",
                },
                "discount_tier": {
                    "type": "string",
                    "description": (
                        "The discount tier: "
                        "'bronze', 'silver', or 'gold'."
                    ),
                    "enum": [
                        "bronze",
                        "silver",
                        "gold",
                    ],
                },
            },
            "required": [
                "price",
                "discount_tier",
            ],
        },
    },
]


# ============================================================
# GEMINI TOOL WRAPPER
# ============================================================

gemini_tools = types.Tool(
    function_declarations=tools_for_llm
)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client()


# ============================================================
# GEMINI CALL
# ============================================================

@traceable(name="Gemini Chat", run_type="llm")
def genai_chat_traced(messages, system_instruction):

    response = client.models.generate_content(
        model=MODEL,
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[gemini_tools],
        ),
    )

    return response


# ============================================================
# MANUAL AGENT LOOP
# ============================================================

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    print("=" * 60)
    print(f"question: {question}")
    print("=" * 60)

    # --------------------------------------------------------
    # Map tool names to actual Python functions
    # --------------------------------------------------------

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    # --------------------------------------------------------
    # System instructions
    # --------------------------------------------------------

    system_instruction = """
    You are a helpful shopping assistant.

    Use get_product_price when you need to find
    the price of a product.

    Use apply_discount when you need to calculate
    a discounted price.

    When answering a question that requires multiple
    steps, use the tools in the appropriate order.
    """

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    messages = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=question
                )
            ],
        )
    ]

    # --------------------------------------------------------
    # Agent loop
    # --------------------------------------------------------

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n--- Iteration {iteration} ---")

        # Ask Gemini what to do next
        response = genai_chat_traced(
            messages=messages,
            system_instruction=system_instruction,
        )

        # ----------------------------------------------------
        # Check whether Gemini requested a tool
        # ----------------------------------------------------

        tool_calls = response.function_calls

        # No tool call means Gemini has the final answer
        if not tool_calls:

            print("\nFinal Answer:")
            print(response.text)

            return response.text

        # ----------------------------------------------------
        # Take the first requested tool call
        # ----------------------------------------------------

        tool_call = tool_calls[0]

        tool_name = tool_call.name
        tool_args = tool_call.args

        print(f"Tool requested: {tool_name}")
        print(f"Arguments: {tool_args}")

        # ----------------------------------------------------
        # Find the corresponding Python function
        # ----------------------------------------------------

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(
                f"Unknown tool requested: {tool_name}"
            )

        # ----------------------------------------------------
        # Execute the Python function
        # ----------------------------------------------------

        observation = tool_to_use(**tool_args)

        print(f"Tool result: {observation}")

        # ----------------------------------------------------
        # Add Gemini's tool-call message to history
        # ----------------------------------------------------

        messages.append(
            response.candidates[0].content
        )

        # ----------------------------------------------------
        # Add the tool result to history
        # ----------------------------------------------------

        messages.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=tool_name,
                        response={
                            "result": observation
                        }
                    )
                ]
            )
        )
    # --------------------------------------------------------
    # Safety limit
    # --------------------------------------------------------

    print("\nMaximum iterations reached.")

    return "Maximum iterations reached."


# ============================================================
# RUN THE AGENT
# ============================================================

if __name__ == "__main__":

    result = run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )