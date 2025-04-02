#! /usr/bin/env python


from operator import itemgetter

import keyring
from langchain_core.runnables import RunnableMap
from langchain_openai import OpenAI
# Deprecated:
# from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

SERVICE = 'OpenAI'
NAME = 'HomeworkAPIKey'


def get_apikey() -> str:
    return keyring.get_password(SERVICE, NAME)


def cleanup(response: dict[str, str]) -> dict[str, str]:
    return {
        'cuisine': response['cuisine'].strip(),
        'restaurant_name': response['restaurant_name'].strip().translate(
            str.maketrans({"'": "_", '"': "_"})
        ),
        'menu_items': [
            menu_item.strip().strip('"').strip("'")
            for menu_item in response['menu_items'].split(',')
        ],
    }


def get_restaurant_name_and_items(cuisine: str) -> dict[str, str]:
    # Temperature = creativity
    # 0 = play it safe
    # 1 = super creative but higher risk of hallucination
    llm = OpenAI(temperature=0.7, api_key=get_apikey())

    prompt_template_name = PromptTemplate(
        input_variables = ['cuisine'],
        template = 'I want to open a restaurant for {cuisine} food. Suggest a fancy name for this.'
    )
    # Deprecated:
    # name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='restaurant_name')

    prompt_template_items = PromptTemplate(
        input_variables = ['restaurant_name'],
        template = '''
            Suggest some menu items for {restaurant_name}.
            Output only a single line, with each item separated by commas.
        '''
        # (no bullet points, no line breaks, no other commentary).
    )
    # Deprecated:
    # food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key='menu_items')

    '''
    # Old style - deprecated:
    chain = SequentialChain(
        chains = [name_chain, food_items_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name', 'menu_items']
    )
    '''

    name_chain = prompt_template_name | llm
    food_items_chain = prompt_template_items | llm

    # Create a RunnableSequence-style chain
    chain = RunnableMap({
        'restaurant_name': itemgetter('cuisine') | name_chain,
        'menu_items': itemgetter('cuisine') | name_chain | food_items_chain,
        'cuisine': itemgetter('cuisine'),
    })

    response = chain.invoke({'cuisine': cuisine})
    return cleanup(response)


if __name__ == '__main__':
    response = get_restaurant_name_and_items('Thai')
    print(
        f'Based on cuisine of: {response["cuisine"]}\n'
        f'Restaurant Name: {response["restaurant_name"]}\n'
        f'Menu Items:\n* {"\n* ".join(response["menu_items"])}\n'
    )
