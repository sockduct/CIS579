#! /usr/bin/env python
'''
Based on https://www.youtube.com/watch?v=MoqgmWV1fm8

This program accepts one or more URLs from the user, retrieves the information
from each URL, and then allows the user to ask questions about the information
including summarizing, providing sources, and more.
'''


# Update runtime behavior:
# Note that this isn't needed in Python v3.11+
from __future__ import annotations

# Standard Library
import argparse
from datetime import datetime
from functools import cache, wraps
from importlib import import_module
from pathlib import Path
import pickle
import sys
import time
# Because TypeVar and ParamSpec are executed at runtime, they must be imported:
from typing import cast, TYPE_CHECKING, TypeVar, ParamSpec


# Third Party Packages
import keyring
import streamlit as st

# Type Checking Only:
if TYPE_CHECKING:
    # Standard Library:
    from collections.abc import Callable

    # Third Party:
    from langchain import OpenAI
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
    from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
    from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
    from langchain.llms import HuggingFacePipeline
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
    from langchain.vectorstores import FAISS
    from streamlit.delta_generator import DeltaGenerator
    from transformers import AutoTokenizer  # type: ignore


# Globals:
__version__ = '1.0.0'
#
SERVICE = 'OpenAI'
NAME = 'HomeworkAPIKey'
SERVICE2 = 'HuggingFace'
NAME2 = 'General'
#
T5_MODEL_NAME = 'google/long-t5-tglobal-base'  # or 'google/long-t5-tglobal-large'


# Type Annotations:
P = ParamSpec('P')  # ParamSpec captures the parameter types of the decorated function
R = TypeVar('R')    # Return type


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f'{int(hours)}h {int(minutes)}m {secs:.3f}s'
    elif seconds >= 60:
        minutes, secs = divmod(seconds, 60)
        return f'{int(minutes)}m {secs:.3f}s'
    else:
        return f'{seconds:.3f}s'


def get_time_formatted() -> str:
    # Get local time with timezone info and format:
    return datetime.now().astimezone().strftime("%m/%d/%Y %I:%M:%S %p %Z")


def track_time(func: Callable[P, R]) -> Callable[P, R]:
    if getattr(func, '_is_tracked', False):
        print(f'Function "{func.__name__}" is already decorated.')
        return func  # Already decorated

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.time()
        print(f'Started - "{func.__name__}" at {get_time_formatted()}...')

        result = func(*args, **kwargs)

        end = time.time()
        duration = end - start
        print(f'Finished - "{func.__name__}" in {format_duration(duration)}.')

        return result

    # Mark function as decorated to prevent multiple decorations:
    setattr(wrapper, '_is_tracked', True)
    return wrapper


@cache
def get_apikey(service: str=SERVICE, name: str=NAME) -> str|None:
    return keyring.get_password(service, name)


@cache
# 'google/long-t5-tglobal-base'
def get_t5_tokenizer(t5_model_name: str=T5_MODEL_NAME) -> AutoTokenizer:
    # Lazy imports to improve load time:
    AutoTokenizer = import_module('transformers').AutoTokenizer
    return AutoTokenizer.from_pretrained(t5_model_name)


@cache
def get_llm(model: str='OpenAI') -> OpenAI|HuggingFacePipeline:
    if model == 'OpenAI':
        # Lazy imports to improve load time:
        OpenAI_cls = import_module('langchain').OpenAI
        from langchain import OpenAI
        OpenAI_type = cast(type[OpenAI], OpenAI_cls)

        llm_params = {
            'max_tokens': 500,
            'model': 'gpt-3.5-turbo-instruct',
            'openai_api_key': get_apikey(),
            'temperature': 0.9,
        }
        # Type annotations don't understand dict unpacking:
        return OpenAI_type(**llm_params)  # type: ignore
    elif model == 'T5':
        # Lazy imports to improve load time:
        AutoTokenizer = import_module('transformers').AutoTokenizer
        pipeline = import_module('transformers').pipeline
        HuggingFacePipeline_cls = import_module('langchain').llms.HuggingFacePipeline
        from langchain.llms import HuggingFacePipeline
        HuggingFacePipeline_type = cast(type[HuggingFacePipeline], HuggingFacePipeline_cls)
        # T5Tokenizer = import_module('transformers').T5Tokenizer
        # T5ForConditionalGeneration = import_module('transformers').T5ForConditionalGeneration
        LongT5ForConditionalGeneration = import_module(
            'transformers').LongT5ForConditionalGeneration


        # model_name = 't5-base'
        # tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        # model_instance = T5ForConditionalGeneration.from_pretrained(model_name)

        # # Create a pipeline for text-to-text generation.
        # # T5 is a text-to-text model, so you can provide a question/prompt and get a text answer.
        # pipe = pipeline(
        #     'text2text-generation',
        #     model=model_instance,
        #     tokenizer=tokenizer,
        #     max_length=512,   # adjust as needed
        #     do_sample=True,   # enables sampling for more varied outputs
        #     temperature=0.9,  # adjust sampling temperature as desired
        #     device=0          # Use GPU device 0 (if CUDA installed and configured)
        # )
        model_name = T5_MODEL_NAME
        tokenizer = get_t5_tokenizer(model_name)
        model_instance = LongT5ForConditionalGeneration.from_pretrained(model_name)

        pipe = pipeline(
            'text2text-generation',
            model=model_instance,
            tokenizer=tokenizer,
            max_length=16384,    # ✅ expand to LongT5's full context
            do_sample=True,
            temperature=0.9,
            device=0,
        )

        return HuggingFacePipeline_type(pipeline=pipe)
    elif model == 'Mistral':
        # Or other 4-bit model from Hugging Face:
        # Lazy imports to improve load time:
        AutoModelForCausalLM = import_module('transformers').AutoModelForCausalLM
        AutoTokenizer = import_module('transformers').AutoTokenizer
        float16 = import_module('torch').float16
        HuggingFacePipeline_cls = import_module('langchain').llms.HuggingFacePipeline
        from langchain.llms import HuggingFacePipeline
        HuggingFacePipeline_type = cast(type[HuggingFacePipeline], HuggingFacePipeline_cls)
        pipeline = import_module('transformers').pipeline

        model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
        hf_token = get_apikey(service=SERVICE2, name=NAME2)

        # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True,)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model_instance = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=float16,
            # low_cpu_mem_usage=True,
            token=hf_token,
        )

        # Note:  The model config doesn't specify a pad_token_id, so transformers
        # defaults to using the eos_token_id.  May need to change if:
        # * Batched generation (like with multiple prompts at once)
        # * Attention masking or decoder position tracking
        # * Text generation with streaming or left-padded input
        pipe = pipeline(
            'text-generation',
            model=model_instance,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            # device=0,  # Don't use - model loaded with 'accelerate' option
            # streaming=True,
            #
            # Suppress "Setting `pad_token_id` to `eos_token_id`:2 for open-end generation."
            # 👇 Prevents pad_token_id warnings and potential decoding issues:
            pad_token_id=model_instance.config.eos_token_id,
        )

        return HuggingFacePipeline_type(pipeline=pipe)
    else:
        raise ValueError(f'Unsupported model: {model}')


def add_url_field() -> None:
    index = st.session_state.url_count
    st.session_state.urls[index] = ''
    st.session_state.url_count += 1


def remove_url_field() -> None:
    if st.session_state.url_count > 1:
        st.session_state.url_count -= 1
        st.session_state.urls.pop(st.session_state.url_count)


def clear_url_fields() -> None:
    for i in range(st.session_state.url_count):
        st.session_state.urls[i] = ''
        st.session_state[f'url_input_{i}'] = ''


def build_sidebar_urls() -> list[str]:
    # Initialize session state:
    if 'url_count' not in st.session_state:
        st.session_state.url_count = 1
    if 'urls' not in st.session_state:
        st.session_state.urls = {0: ''}

    st.sidebar.title('Article URLs')

    # Buttons for adding/removing URL fields:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button('➕ Add URL', on_click=add_url_field)
    with col2:
        st.button('➖ Remove URL', on_click=remove_url_field)

    # Render the dynamic input fields:
    for i in range(st.session_state.url_count):
        st.session_state.urls[i] = st.sidebar.text_input(
            label=f'URL {i + 1}',
            key=f'url_input_{i}',
            value=st.session_state.urls.get(i, '')
        )

    # Process/Clear buttons (bottom row):
    process_clicked = st.sidebar.button('✅ Process URLs')
    st.sidebar.button('🧹 Clear URLs', on_click=clear_url_fields)

    # Return values only if Process clicked:
    return list(st.session_state.urls.values()) if process_clicked else ['']


def split_tokens_custom(data: list[Document], tokenizer: AutoTokenizer, *, chunk_size: int=500,
                        chunk_overlap: int=50, verbose: bool=False) -> list[Document]:
    # Lazy imports to improve load time:
    Document = import_module('langchain').schema.Document

    split_docs = []
    for doc in data:
        tokens = tokenizer.encode(doc.page_content)
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            split_docs.append(Document(page_content=chunk_text, metadata=doc.metadata))
            if verbose:
                print(
                    f'split_tokens_custom: Chunk text length: {len(chunk_text)}, '
                    f'{start=}, {end=}, length chunk_tokens: {len(chunk_tokens)}',
                    file=sys.stderr, flush=True
                )
            start += chunk_size - chunk_overlap
    return split_docs


def get_text_chunks(llm_model: str, data: list[Document], verbose: bool=False) -> list[Document]:
    # Note:  TokenTextSplitter currently unused but available:
    if llm_model == 'OpenAI':
        text_splitting = 'recursive'
    elif llm_model == 'Mistral':
        text_splitting = 'mistraltoken'
    elif llm_model == 'T5':
        text_splitting = 't5token'
    else:
        raise ValueError(f'Unsupported LLM model: {llm_model}')

    # Split data:
    if verbose:
        print(
            f'build_database:  Using {text_splitting} Text Splitter...', file=sys.stderr, flush=True
        )
    # Type annotations:
    text_splitter: RecursiveCharacterTextSplitter|TokenTextSplitter|None
    if text_splitting == 'recursive':
        # Lazy imports to improve load time:
        RecursiveCharacterTextSplitter = import_module(
            'langchain').text_splitter.RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1_000,
            # chunk_overlap=200
        )
    elif text_splitting == 'token':
        # Lazy imports to improve load time:
        TokenTextSplitter = import_module('langchain').text_splitter.TokenTextSplitter
        # Create a token-based splitter. Adjust chunk_size and chunk_overlap as needed.
        # The `encoding_name` parameter should correspond to your model's tokenizer.
        text_splitter = TokenTextSplitter(
            chunk_size=350,
            chunk_overlap=75,
            encoding_name='gpt2'
        )
    elif text_splitting == 't5token':
        text_splitter = None
        tokenizer = get_t5_tokenizer()
    elif text_splitting == 'mistraltoken':
        # Lazy imports to improve load time:
        AutoTokenizer = import_module('transformers').AutoTokenizer
        text_splitter = None
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    else:
        raise ValueError(f'Unsupported text splitting method: {text_splitting}')

    if text_splitter:
        docs = text_splitter.split_documents(data)
    else:
        if verbose:
            print(
                'build_database:  Using split_tokens_custom with tokenizer '
                f'"{tokenizer.__class__.__name__}"...',
                file=sys.stderr, flush=True
            )
        max_tokens = 1024  # For Mistral and T5
        chunk_size = max_tokens//2
        chunk_overlap = chunk_size//8
        docs = split_tokens_custom(
            data, tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap, verbose=verbose
        )

    return docs


def build_database(urls: list[str], main_placeholder: DeltaGenerator, *,
                   llm_model: str='recursive', embeddings_model: str='OpenAIEmbeddings',
                   verbose: bool=False) -> FAISS:
    # Lazy imports to improve load time:
    WebBaseLoader = import_module('langchain').document_loaders.WebBaseLoader
    FAISS_cls = import_module('langchain').vectorstores.FAISS
    from langchain.vectorstores import FAISS  # runtime import required for cast
    FAISS_type = cast(type[FAISS], FAISS_cls)

    # Load data:
    main_placeholder.text('Data Loading...Started...✅✅✅')
    loaders = WebBaseLoader(urls)
    data = loaders.load()

    main_placeholder.text('Text Splitter...Started...✅✅✅')
    docs = get_text_chunks(llm_model, data, verbose)

    main_placeholder.text(f'Started Building Embeddings Vectors with {embeddings_model}...✅✅✅')
    # Type annotations:
    embeddings: HuggingFaceEmbeddings|OpenAIEmbeddings
    # Create embeddings and save to FAISS index:
    if embeddings_model == 'HuggingFaceELECTRA':
        # Lazy imports to improve load time:
        HuggingFaceEmbeddings = import_module('langchain').embeddings.HuggingFaceEmbeddings
        # Hugging Face ELECTRA Model:
        embeddings = HuggingFaceEmbeddings(model_name='google/electra-small-discriminator')
    elif embeddings_model == 'HuggingFaceEmbeddings':
        # Lazy imports to improve load time:
        HuggingFaceEmbeddings = import_module('langchain').embeddings.HuggingFaceEmbeddings
        # Hugging Face MiniLM Model:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    elif embeddings_model == 'HuggingFaceEmbeddingsHigh':
        # Lazy imports to improve load time:
        HuggingFaceEmbeddings = import_module('langchain').embeddings.HuggingFaceEmbeddings
        # Hugging Face MPNet Model:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    elif embeddings_model == 'OpenAIEmbeddings':
        # Lazy imports to improve load time:
        OpenAIEmbeddings = import_module('langchain').embeddings.OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=get_apikey())
    else:
        raise ValueError(f'Unsupported embeddings model: {embeddings_model}')

    if verbose:
        print(
            f'build_database:  Using {embeddings_model} Embeddings Model...', file=sys.stderr,
            flush=True
        )

    main_placeholder.text('Building Vector Database with FAISS...✅✅✅')
    return FAISS_type.from_documents(docs, embeddings)


def save_database(vectorstore: FAISS, file_path: Path, main_placeholder: DeltaGenerator) -> None:
    main_placeholder.text('Saving the Vector Database to disk...✅✅✅')

    # Save the FAISS index to a pickle file:
    with open(file_path, 'wb') as f:
        pickle.dump(vectorstore, f)


def load_database(file_path: Path, main_placeholder: DeltaGenerator) -> FAISS:
    main_placeholder.text('Loading the Vector Database from disk...✅✅✅')

    # Load the FAISS index from a pickle file:
    if not file_path.is_file():
        raise RuntimeError('Invalid path to Vector Database.')

    # Type annotations:
    vectorstore: FAISS
    with open(file_path, 'rb') as f:
        vectorstore = pickle.load(f)

    return vectorstore


def show_answer(response: dict[str, str], main_placeholder: DeltaGenerator) -> None:
    main_placeholder.empty()
    st.header('Answer:')
    st.write(response['answer'])

    # Display sources, if available
    if sources := response['sources']:
        st.subheader('Sources:')

        # Split the sources by newline only:
        if ', and ' in sources:
            sources = sources.replace(', and ', '\n')
        if ' and ' in sources:
            sources = sources.replace(' and ', '\n')
        if ', ' in sources:
            sources = sources.replace(', ', '\n')
        # Tempted to strip trailing period, but could be legitimate so leaving.
        source_list = sources.split('\n')

        for source in source_list:
            if clean_source := source.strip():
                st.markdown(f'- {clean_source}')


def fit_query(query: str, llm_model: str, *, max_tokens: int=1024, verbose: bool=False) -> str:
    # Mistral and T5 have a token limit of 1,024
    truncate = False

    # Limit query token length to half of context window limit:
    query_max_tokens = max_tokens//2

    if llm_model == 'Mistral':
        # Lazy imports to improve load time:
        AutoTokenizer = import_module('transformers').AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    elif llm_model == 'T5':
        tokenizer = get_t5_tokenizer(T5_MODEL_NAME)
    else:
        raise ValueError(f'Unsupported LLM model: {llm_model}')

    prompt_tokens = tokenizer.encode(query)
    if len(prompt_tokens) > query_max_tokens:
        query = tokenizer.decode(prompt_tokens[:query_max_tokens], skip_special_tokens=True)
        truncate = True

    if truncate and verbose:
        print(
            f'query_llm:  Truncated query to {len(query)} tokens to fit in {llm_model}...',
            file=sys.stderr, flush=True
        )

    return query


def query_llm(query: str, main_placeholder: DeltaGenerator, vectorstore: FAISS, *,
              llm_model: str='OpenAI', verbose: bool=False) -> dict[str, str]:
    main_placeholder.text(f'Querying {llm_model} LLM using Article URL(s)...✅✅✅')

    # Lazy imports to improve load time:
    load_qa_chain = import_module('langchain').chains.question_answering.load_qa_chain
    PromptTemplate = import_module('langchain').prompts.PromptTemplate

    # Type annotations:
    chain: BaseCombineDocumentsChain|BaseQAWithSourcesChain|RetrievalQAWithSourcesChain
    if llm_model in {'Mistral', 'T5'}:
        # Select method and number of documents to retrieve:
        # chain_method = 'map_reduce'  # Consistently over the token limit with this method
        chain_method = 'refine'
        chain_docs = 5

        if verbose:
            print(
                f'query_llm:  Using {llm_model} LLM with {chain_method} chain...', file=sys.stderr,
                flush=True
            )

        query = fit_query(query, llm_model, verbose=verbose)

        # Retrieve a manageable set of document chunks:
        retriever = vectorstore.as_retriever(search_kwargs={'k': chain_docs})
        docs = retriever.get_relevant_documents(query)

        # Define custom prompt templates to force concise answers:
        initial_prompt_template = PromptTemplate(
            input_variables=['question', 'context_str'],
            template=(
                # 'Given the following context, answer the question as concisely as possible '
                # '(limit your answer to 150 words).\n\n'
                # 'question: {question}\n\n'
                # 'context: {context_str}\n\n'
                # 'answer:'
                'You are an expert assistant. Given the context below, answer the question '
                'clearly and concisely.\n\n'
                'Context:\n{context_str}\n\n'
                'Question: {question}\nAnswer:'
            ),
        )
        refine_prompt_template = PromptTemplate(
            input_variables=['existing_answer', 'context_str'],
            template=(
                # 'The current answer is:\n{existing_answer}\n\n'
                # 'Given this additional context, refine the answer to be as concise as possible '
                # '(do not exceed 150 words). \n\n'
                'We have a partial answer: {existing_answer}\n'
                'Given this additional context, refine the answer to be more complete, if '
                'needed.\n\n'
                'New Context:\n{context_str}\n\n'
                'Refined Answer:'
            ),
        )

        # Build a refine chain that processes each document one by one:
        chain = load_qa_chain(
            get_llm(llm_model),
            chain_type='refine',
            question_prompt=initial_prompt_template,
            refine_prompt=refine_prompt_template,
            # For debugging but very verbose:
            # verbose=verbose,
        )

        # Create a map-reduce QA chain:
        # This chain processes each document chunk individually (map step) and
        # then combines the answers (reduce step):
        # chain = load_qa_chain(get_llm(llm_model), chain_type='map_reduce')

        # Another option:
        # chain = load_qa_chain(get_llm('T5'), chain_type='stuff', verbose=True)

        if verbose:
            for i, doc in enumerate(docs, 1):
                print(f'Doc {i} length:  {len(doc.page_content)}...', file=sys.stderr, flush=True)
        result = chain({'input_documents': docs, 'question': query})
        answer = 'output_text'
    elif llm_model == 'OpenAI':
        # Lazy imports to improve load time:
        RetrievalQAWithSourcesChain = import_module('langchain').chains.RetrievalQAWithSourcesChain
        if verbose:
            print(f'query_llm:  Using {llm_model} LLM...', file=sys.stderr, flush=True)
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=get_llm(llm_model), retriever=vectorstore.as_retriever()
        )
        result = chain({'question': query}, return_only_outputs=True)
        answer = 'answer'
    else:
        raise ValueError(f'Unsupported LLM model: {llm_model}')

    if verbose:
        print(f'Chain keys:  {result.keys()}', file=sys.stderr)
    return {
        'answer': result.get(answer, 'No answer provided.'),
        'sources': result.get('sources', ''),
    }


def get_args() -> argparse.Namespace:
    # Setup argument parsing
    parser = argparse.ArgumentParser(
                description='Web application for article research leveraging LLMs',
                epilog=(
                    'This is a streamlit application - run with:  '
                    f'streamlit run {Path(__file__).name} -- <optional-arguments>'
                )
    )
    parser.add_argument('--version', action='version', version=__version__)

    # Add (root) optional argument(s)
    parser.add_argument(
        '-e', '--embeddings_model', default='OpenAIEmbeddings',
        help='one of OpenAIEmbeddings (default, cloud), HuggingFaceEmbeddings (MiniLM, local), '
             'HuggingFaceEmbeddingsHigh (MPNet, local), HuggingFaceELECTRA (Experimental, local)'
    )
    parser.add_argument(
        '-l', '--llm_model', default='OpenAI',
        help='one of OpenAI (default), Mistral (local with cloud updates), '
             'T5 (local, work-in-progress)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='enable additional monitoring output'
    )

    return parser.parse_args()


def main(embeddings_model: str, llm_model: str, verbose: bool=False) -> None:
    # File location:
    cwd = Path(__file__).parent
    file_path = cwd/'faiss_store.pkl'
    vectorstore = None

    # Site layout:
    st.title('Athena: Article Research Tool 📈')
    process_urls = build_sidebar_urls()
    main_placeholder = st.empty()

    # URL input:
    if process_urls and any(url.strip() for url in process_urls):
        decorated_build_database = track_time(build_database) if verbose else build_database
        vectorstore = decorated_build_database(
            process_urls, main_placeholder, llm_model=llm_model,
            embeddings_model=embeddings_model, verbose=verbose
        )
        save_database(vectorstore, file_path, main_placeholder)
        main_placeholder.empty()

    # Question input:
    with st.form(key='question_form'):
        query = st.text_area(
            'Question:', key='user_query', height=100, value=st.session_state.get('user_query', '')
        ).strip()
        submitted = st.form_submit_button(label='Submit')
    if st.button('❌ Clear'):
        st.session_state.pop('user_query', None)
        st.rerun()
    if submitted and query:
        if not vectorstore:
            vectorstore = load_database(file_path, main_placeholder)
        decorated_query_llm = track_time(query_llm) if verbose else query_llm
        response = decorated_query_llm(
            query, main_placeholder, vectorstore, llm_model=llm_model, verbose=verbose
        )
        show_answer(response, main_placeholder)
        main_placeholder.empty()


if __name__ == '__main__':
    args = get_args()

    main(embeddings_model=args.embeddings_model, llm_model=args.llm_model, verbose=args.verbose)
