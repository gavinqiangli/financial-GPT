import streamlit as st 
import time 
import requests
from trubrics.integrations.streamlit import FeedbackCollector
from trubrics import Trubrics 
from researcher import generate_response

def chat_stock():
    """Display the Stock Finance Chatbot."""

    # store llm responses 
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can i help you ?"}]

    # display chat messages
    for message in st.session_state.messages:
        avatar=None 
        if message["role"] == "assistant":
            avatar = 'bird.png'
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    # func for generating llm responses - echo
    # def generate_response(prompt_input):
    #     time.sleep(2)
    #     return f"GPT echos: {prompt_input}"



    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # generate response if last message is not from assistant
        with st.chat_message("assistant", avatar='bird.png'):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.write(response)
                # chunks = response.split()
                # num_chunks = len(chunks) 

                # msg_placeholder = st.empty()
                # full_response = ""
                # for i, chunk in enumerate(chunks):
                #     full_response += chunk + " "
                #     time.sleep(0.05)
                #     # Check if it's the last chunk
                #     if i == num_chunks - 1:
                #         #for last chunk, we don't add the blinking cursor
                #         msg_placeholder.markdown(full_response)
                #     else:
                #         # Add blinking cursor to simulate typing
                #         msg_placeholder.markdown(full_response + "▌")
                # #msg_placeholder.markdown(full_response)  

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

        collector = FeedbackCollector(
            email= "alexandros_1010@outlook.com", #st.secrets.TRUBRICS_EMAIL,
            password= "aliexandros1", #st.secrets.TRUBRICS_PASSWORD,
            project="default"
        )

        user_feedback = collector.st_feedback(
            component="Chat Feedback",
            feedback_type="thumbs",
            model="gpt-3.5-turbo",
            prompt_id=None, #checkout collector.log_prompt() to log users prompts
            open_feedback_label="[Optional] Provide additional feedback",
            metadata={"prompt": prompt, "response": response}
        )
        logged_prompt = collector.log_prompt(
            config_model={"model": "gpt-3.5"},
            prompt=prompt,
            generation=response
        )
    #    if user_feedback:
    #        st.write('Feedback sent')
