from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
MAX_ITERATIONS = 5
# Region Prompts

system_prompt = """
You are a helpful shopping assistant.
    You have access to a product catalog tool
    and a discount tool.\n\n
    STRICT RULES — you must follow these exactly:\n
    1. NEVER guess or assume any product price.
    You MUST call get_product_price first to get the real price.\n
    2. Only call get_product_discount AFTER you have received
    a price from get_product_price. Pass the exact price
    returned by get_product_price — do NOT pass a made-up number.\n
    3. NEVER calculate discounts yourself using math.
    Always use the get_product_discount tool.\n
    4. If the user does not specify a discount tier,
    ask them which tier to use — do NOT assume one.
"""
user_prompt = """
What is the laptop price after the silver tier discount
"""
# endregion

# Region tools

def get_product_price(product:str)->float:
    print(f"executing the get_product_price")
    item_price_dict={"laptop":100000,"headphone":1000,"keyboard":2000}

    return item_price_dict.get(product)

def get_product_discount(price: float, discount_tier:str)->float:
    print(f"executing the get_product_discount")

    items_discount={"bronze":5, "silver": 7,"gold":10}
    discount = items_discount.get(discount_tier)

    return price - round(price * (discount/100),2)

# region model 
MODEL ="gpt-5.5"
TEMPERATURE=0
tool_dict={"get_product_price":get_product_price,
      "get_product_discount":get_product_discount}
tools=[{
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Lookup price for a product in the catalog. Products are laptop, headphones and keyboard",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Product from the catalog",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_discount",
            "description": "Apply a discount tier to the price and return the final price. Available discount tiers: bronze, silver, gold",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "Price of the product",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "Discount tier to be one of bronze, silver, gold",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    }]

def init_chat_model(message):
    client = OpenAI()
    response = client.chat.completions.create(model=MODEL, temperature=TEMPERATURE, messages=message, tools=tools)

    return response

# ENDREGION MODEL

# REGION AGENT
def main():
    message=[
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":user_prompt}
    ]
    # ai_message 
    for i in range(MAX_ITERATIONS):
        print(f"Iteration number {i+1}")

        response=init_chat_model(message)
        ai_message=response.choices[0].message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("Success")
            return ai_message.content
        else:
            message.append(ai_message)
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"the tool name {tool_name}")
                observation = tool_dict[tool_name](**args)
                message.append({
                    "role": "tool",
                    "content": str(observation),
                    "tool_call_id": tool_call.id
                })


                    # message.append({"role":"tool", "content":str(observation),"tool_call_id":tool_call_id})

    return message
# ENDREGION AGENT

# MAIN 
if __name__ =="__main__":
    print(f"executing main")
    print(main())
# ENDMAIN 