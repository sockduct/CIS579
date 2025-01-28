#! /usr/bin/env python3.13


# Standard Library:
import argparse

# 3rd Party Library:
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from ollama import ResponseError


# Globals:
MODELS = ('llama3.1', 'llama3.2', 'deepseek-r1:8b')
TEMPLATE = '''
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
'''


def handle_conversation(*, model: OllamaLLM) -> None:
    context = ''
    terminations = ('exit', 'quit', 'q', 'bye', 'goodbye', 'end', '')
    print('Welcome to the AI ChatBot, type "exit" to quit.')

    while True:
        user_question = input('You: ')
        if user_question.lower() in terminations:
            break

        prompt = ChatPromptTemplate.from_template(TEMPLATE)
        chain = prompt | model
        params = dict(context=context, question=user_question)
        response = chain.invoke(params)
        print(f'AI ChatBot:\n{response}')
        context += f'\nUser: {user_question}\nAI: {response}'


def basic_query(*, model: OllamaLLM, test_run: bool=False,
                question: str='Hey, How are you?') -> None:
    # Super basic:
    if test_run:
        result = model.invoke(input='Hello, World!')
    # Building a conversational template:
    else:
        prompt = ChatPromptTemplate.from_template(TEMPLATE)
        chain = prompt | model

        params = dict(context='', question=question)
        result = chain.invoke(params)

    print(result)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='An example program to allow interacting with local LLMs.  Available models: '
        '* llama3.1 - 8b (default) ** llama3.2 - 3b (quick) *** deepseek-r1:8b - (CoT, slower)',
        epilog='Created as part of CIS 579, Homework 2'
    )
    parser.add_argument(
        '-m', '--model', type=str, default='llama3.1',
        help=f'LLM to use ({", ".join(MODELS)})'
    )
    parser.add_argument(
        '-q', '--question', type=str, default='Hey, How are you?', help='Question to ask the LLM'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-c', '--conversation', action='store_true', default=True, help='Enter conversation mode.'
    )
    group.add_argument(
        '-s', '--single', action='store_true', help='Ask single question and return.'
    )
    group.add_argument(
        '-t', '--testrun', action='store_true', help='Hello, World to test model'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = OllamaLLM(model=args.model)

    if args.verbose:
        print(f'Using model {args.model}.')

    try:
        if args.testrun:
            if args.verbose:
                print('Running in test mode - simple Hello, World question for model.')
            basic_query(model=model, test_run=True)
        elif args.single:
            if args.verbose:
                print('Running in single question mode.')
            basic_query(model=model, question=args.question)
        elif args.conversation:
            if args.verbose:
                print('Running in conversation mode.')
            handle_conversation(model=model)
        else:
            print('Error:  Unhandled argument case.')
            raise SystemExit(1)
    except ResponseError as err:
        print(f'{err.error.capitalize()}.\nAvailable models: {", ".join(MODELS)}')
        raise SystemExit(1) from err
