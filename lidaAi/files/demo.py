import csv
import os
import pandas as pd
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
from lida.utils import plot_raster

lida1 = Manager(text_gen = llm("openai",api_key = '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'))


def save_uploaded_csv_file(file):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique file name, e.g., using the original file name
    file_name = os.path.join(temp_dir, 'test.csv')
    
    # Write the uploaded file to the generated file name
    with open(file_name, 'wb') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    return file_name

# def read_csv_file(csv_file):

#     temp_dir = "temp_csv_uploads"
#     os.makedirs(temp_dir,exist_ok =True)

#     return csv_file

def get_csv_summary(filename):
    
    lida1 = Manager(text_gen = llm("openai",api_key = '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'))
    textgen_config = TextGenerationConfig(n=1,temperature=0.2,model='gpt-3.5-turbo',use_cache=True)
    summary = lida1.summarize(filename,textgen_config=textgen_config)
    return summary

def get_goals(filename):
    lida1 = Manager(text_gen = llm("openai",api_key = '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'))
    summary = get_csv_summary(filename)
    goals = lida1.goals(summary, n=5)
    return goals

def get_visualization(filename,query):
    lida1 = Manager(text_gen = llm("openai",api_key = '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'))
    textgen_config = TextGenerationConfig(n=1,temperature=0.7,model='gpt-3.5-turbo',use_cache=True)
    #filename = 'C:\practice\project-lida\iris.csv'
    summary = lida1.summarize(filename,textgen_config=textgen_config)

    charts = lida1.visualize(summary=summary, goal=query,textgen_config=textgen_config)
    #os.remove(filename)
    return charts


# query = 'Histogram of sepal length'
# charts = get_visualization(filename,query)
# print(plot_raster(charts[0].raster))