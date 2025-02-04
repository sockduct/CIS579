#! /usr/bin/env python3.13


# Standard Library:
import argparse
from datetime import datetime
from reprlib import aRepr, repr
from pathlib import Path
from time import perf_counter

# 3rd Party Library:
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from ollama import ResponseError


# Globals:
CWD = Path(__file__).parent
MODELS = ('llama3.1', 'llama3.2', 'deepseek-r1:8b')
TEMPLATE = '''
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
'''


def handle_conversation(*, model: OllamaLLM, context: str='', question: str='',
                        verbose: bool=False) -> None:
    terminations = ('exit', 'quit', 'q', 'bye', 'goodbye', 'end', '')
    print('Welcome to the AI ChatBot, type "exit" to quit.')

    while True:
        if not question:
            user_question = input('You: ')
            if user_question.lower() in terminations:
                break
        else:
            user_question = question

        if verbose:
            aRepr.maxstring = 60
            if context:
                print(f'Supplied context passed to model:\n{repr(context)}')
            if question:
                print(f'Question posed:\n{question}')

        prompt = ChatPromptTemplate.from_template(TEMPLATE)
        chain = prompt | model
        params = dict(context=context, question=user_question)
        response = chain.invoke(params)
        print(f'AI ChatBot:\n{response}')
        context += f'\nUser: {user_question}\nAI: {response}'
        if question:
            question = ''


def basic_query(*, model: OllamaLLM, test_run: bool=False, context: str='',
                question: str='Hey, How are you?', verbose: bool=False) -> None:
    # Super basic:
    if test_run:
        result = model.invoke(input='Hello, World!')
    # Building a conversational template:
    else:
        if verbose:
            aRepr.maxstring = 60
            if context:
                print(f'Supplied context passed to model:\n{repr(context)}')
            print(f'Question posed:\n{question}')

        prompt = ChatPromptTemplate.from_template(TEMPLATE)
        chain = prompt | model

        params = dict(context=context, question=question)
        result = chain.invoke(params)

    if verbose:
        print('AI ChatBot answer:')
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
    parser.add_argument('-t', '--time', action='store_true', help='Time LLM response')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument(
        '-i', '--interactive', action='store_true', default=True,
        help='Enter interactive conversation mode.'
    )
    group1.add_argument(
        '-s', '--single', action='store_true', help='Ask single question and return.'
    )
    group1.add_argument(
        '--testrun', action='store_true', help='Hello, World to test model'
    )

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        '-c', '--context', type=str, help='Enter background context data for question(s).'
    )
    group2.add_argument(
        '-f', '--file', type=str, default='', help='File with background context data for question(s).'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.time:
        start = perf_counter()
        if args.verbose:
            print(f'Start time:  {datetime.now():%I:%M:%S %p}')
    model = OllamaLLM(model=args.model)

    if args.verbose:
        print(f'Using model {args.model}.')

    if args.context:
        context = args.context
    elif args.file:
        file = Path(args.file).resolve()
        if not(file.drive or file.root):
            file = CWD/file

        if not file.exists() or not file.is_file():
            raise ValueError(f'File {file} does not exist or is not a file.')
        with open(file, 'r') as infile:
            context = infile.read().strip()
    else:
        context = ''

    try:
        if args.testrun:
            if args.verbose:
                print('Running in test mode - simple Hello, World question for model.')
            basic_query(model=model, test_run=True)
        elif args.single:
            if args.verbose:
                print('Running in single question mode.')
            basic_query(model=model, context=context, question=args.question, verbose=args.verbose)
        elif args.interactive:
            if args.verbose:
                print('Running in interactive conversation mode.')
            handle_conversation(
                model=model, context=context, question=args.question, verbose=args.verbose
            )
        else:
            print('Error:  Unhandled argument case.')
            raise SystemExit(1)
    except ResponseError as err:
        print(f'{err.error.capitalize()}.\nAvailable models: {", ".join(MODELS)}')
        raise SystemExit(1) from err

    if args.time:
        stop = perf_counter()
        print(f'AI ChatBot response time: {stop - start:,.1f} seconds.')
