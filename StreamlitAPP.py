import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenerator.MCQgenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging


# Loading json file
with open(r'C:\Users\ACER\mcqq\response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Create a title for the app
st.title("MCQs Creator Application with langchain")    

# Create a form using st.form
with st.form("user_inputs"):
    # Form inputs
    uploaded_file = st.file_uploader("Upload a PDF or txt file")
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Insert Subject", max_chars=20)
    tone = st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")
    
    # Changed from st.from_submit_button to st.form_submit_button
    submit_button = st.form_submit_button("Create MCQs")

# Process form submission
if submit_button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("Loading..."):
        try:
            # Read the uploaded file
            text = read_file(uploaded_file)
            
            # Count tokens and the cost of API call
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                )
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("Error")
        else:    

            # Display token usage and cost information
            st.sidebar.write("### Token Usage Information")
            st.sidebar.write(f"Total Tokens: {cb.total_tokens}")
            st.sidebar.write(f"Prompt Tokens: {cb.prompt_tokens}")
            st.sidebar.write(f"Completion Tokens: {cb.completion_tokens}")
            st.sidebar.write(f"Total Cost: ${cb.total_cost:.4f}")
            
            # Process and display the response
            if isinstance(response, dict):
                quiz = response['quiz']
                #st.write("all the response from llm:",response)
                #st.write("just the quiz :", quiz)
                #st.write(type(quiz))
              
                if quiz is not None:
                    

                    table_data = get_table_data(quiz)
                    #st.write("Table data length:", len(table_data))
                    #if not table_data:
                        #st.write("Table data is empty. Check the console for error messages.")
                    #st.write(table_data)
                    if table_data is not None:
                        st.write("### Generated MCQs")
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        st.table(df)
                        
                        
                        st.write("### Review")
                        st.text_area(
                            label="Review",
                            value=response["review"],
                            height=100,
                            disabled=True
                        )
                    else:
                        st.error("Error in processing the table data")
                else:
                    st.error("No quiz data found in the response")
            else:
                st.warning("Unexpected response format")
                st.write(response)
                
        