import streamlit as st
import config
import onnxruntime_genai as og
import argparse
import time
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

credential = AzureKeyCredential(config.ai_search_key)



st.title("Contoso Gaming Customer Assistant")

system_prompt = """


You are a Contoso Gaming Customer Assistant, tasked with responding to customer queries. Refer to the context provided below to respond to the questions.
Your responses must be derived entirely from the context provided. **DO NOT MAKE STUFF UP**. If you don't know the answer, you can say so.
Respond in a succinct and professional manner, and keep the response easily readble, with formatting, etc. The customer is always right, even when they are wrong. Good luck!

"""

def init():
    if "model" not in st.session_state:
        st.session_state.model = og.Model(f'{config.model}')
        st.session_state.tokenizer = og.Tokenizer(st.session_state.model)
        print("Model loaded")
        st.session_state.tokenizer_stream = st.session_state.tokenizer.create_stream()
        print("Tokenizer created")
        print()

        st.session_state.search_options = {}
        st.session_state.search_options['do_sample'] = False
        st.session_state.search_options['max_length'] = 2048


def perform_search_based_qna(query):
    print("Calling Azure Search for query: ", query)
    search_response = None

    search_client = SearchClient(endpoint=config.ai_search_url,
                      index_name=config.ai_index_name,
                      credential=credential)

    results = search_client.search(
        search_text=query, 
        query_type="semantic",
        semantic_configuration_name=config.ai_semantic_config
    )
    
    for index, result in enumerate(results):
        # print(f"Index: {index}, Result: {result}")
        if result['content']:
            if search_response is None:
                search_response = result['content']
            else:
                search_response += result['content'] + \
                " \n --- next document ---- \n"

        if index == 2:
            break
    print("Search response: \n", search_response)
    return search_response



if prompt := st.chat_input("Hello!!!"):

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        init()
        chat_template = '<|user|>instructions:\n{system_prompt}\ncontext:\n{context}\nuser query:\n{input} <|end|>\n<|assistant|>'

        message_placeholder = st.empty()
        search_response = perform_search_based_qna(prompt)


        # If there is a chat template, use it
        prompt = f'{chat_template.format(system_prompt=system_prompt, context=search_response,input=prompt)}'

        input_tokens = st.session_state.tokenizer.encode(prompt)

        params = og.GeneratorParams(st.session_state.model)
        params.set_search_options(**st.session_state.search_options)
        params.input_ids = input_tokens

        generator = og.Generator(st.session_state.model, params)
        print("Generator created")
        print("Running generation loop ...")
        print()
        print("Output: ", end='', flush=True)
        full_response = ""
        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                print(st.session_state.tokenizer_stream.decode(new_token), end='', flush=True)
                full_response += st.session_state.tokenizer_stream.decode(new_token)
                message_placeholder.markdown(full_response)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator
