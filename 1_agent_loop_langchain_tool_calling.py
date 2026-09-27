from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage,SystemMessage,ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL= "gemini-3.8-flash"

@tool
def get_product_price(product:str)->float:
    """Look up the price in the product catalog"""
    print(f" >>executing get price (product='{product}')")
    prices ={"laptop":1200.10,"mobile":800.25,"heaset":500.56}
    return prices.get(product,0)

@tool
def apply_discount(price:float,discount_tier:str)->float:
    """Apply a discount tier to a product and return the final price. available tiers:bronze,silver,gold"""
    print(f" >>executing apply_discount (price='{price}',discount='{discount_tier})")
    discount_percentage={"bronze":5,"silver":10,"gold":15}
    discount=discount_percentage.get(discount_tier,0)
    return round(price * (1-discount/100),2)

##we are creating agent The tools list tells the LangChain agent what tools are available
@traceable(name="Langchaing Agent Loop")
def run_agent(question:str):
    tools=[get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"google_genai:{MODEL}",temperature=0)
    llm_with_tools=llm.bind_tools(tools)

    print(f"question:{question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ =="__main__":
    print("hello Langchain agent (.bind_tools)!")
    print()
    result=run_agent("what is the price of a laptop after applying a gold discount")