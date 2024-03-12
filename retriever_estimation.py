from retriever.retriever import (
    RetrieverPipeline,
    setup_retriever_pipelines
)
from arguments import args
import pandas as pd
from functools import reduce
from envs import *
import os
from typing import List

def read_input_file(filename, query_column, groundtruth_column = None):
    df = pd.read_csv(filename)

    queries = df[query_column].to_list() 
    groundtruth = df[groundtruth_column].to_list() if groundtruth_column != None else [None for i in range(len(queries))]

    return groundtruth, queries

retriever_pipeline = setup_retriever_pipelines(args)

# groundtruth, queries = read_input_file(filename=FAQ_FILE, 
#                                        query_column="Paraphased-question", 
#                                        groundtruth_column="Question")

groundtruth, queries = read_input_file(filename='data/Enriched_FAQ.csv', 
                                       query_column="query")

def retrieve_data(queries: List[str], truth: List[str]):
    r'''
        Return type: Tuple(List[str], List[Union(str, None)], List[Union(str, None)])
    '''
    # res['answer'] now is a List of List[Document]
    res = retriever_pipeline["query_pipeline"](query=queries,
                                        debug=True,
                                        params={"EmbeddingRetriever": {'root_node':'Query', 'index':'faq'}})

    truth_res = [ [doc for doc in retrieves if doc.answer == truth] if truth != None else [] for retrieves, truth in zip(res['answers'], truth) ]
    truth_res = list(map(lambda x: x[0] if len(x) > 0 else None, truth_res))

    top1_res = list(map(
        lambda x: x[0] if len(x) > 0 else None, res['answers']))

    return queries, top1_res, truth_res

def nonGroundOutput(acc, ele):
    r'''acc is a dict with format:
        'Query': []
        'Top 1 Retrieve': []
        'Top 1 Score': []
    '''

    query, top1_res, _ = ele
    acc['Query'].append(query)

    if top1_res:
        acc['Top 1 Retrieve'].append(top1_res.answer)
        acc['Top 1 Score'].append(top1_res.score)
    else:
        acc['Top 1 Retrieve'].append("")
        acc['Top 1 Score'].append("")
    
    return acc

def GroundOutput(acc, ele):
    r'''acc is a dict with format:
        'Query': []
        'Top 1 Retrieve': []
        'Top 1 Score': []
        'Groundtruth': []
        'Groundtruth Score': []
    '''

    query, top1_res, groundtruth_res = ele
    acc['Query'].append(query)

    if top1_res != None:
        acc['Top 1 Retrieve'].append(top1_res.answer)
        acc['Top 1 Score'].append(top1_res.score)

        if groundtruth_res != None:
            if groundtruth_res.answer != top1_res.answer:
                acc['Groundtruth'].append(groundtruth_res.answer)
                acc['Groundtruth Score'].append(groundtruth_res.score)
            else: 
                acc['Groundtruth'].append("")
                acc['Groundtruth Score'].append("")
        else:
            acc['Groundtruth'].append("")
            acc['Groundtruth Score'].append("")
    else:
        acc['Top 1 Retrieve'].append("")
        acc['Top 1 Score'].append("")

        if groundtruth_res != None:
            acc['Groundtruth'].append(groundtruth_res.answer)
            acc['Groundtruth Score'].append(groundtruth_res.score)
        else:
            acc['Groundtruth'].append("")
            acc['Groundtruth Score'].append("")

    return acc


BATCH_SIZE = 10

# Return: List with size = num_batch and each batch has a tuple of three "batch_size-size" lists
retrieve_doc = map(lambda x: retrieve_data(x[0], x[1]), 
                    [(queries[i: i + BATCH_SIZE], groundtruth[i: i + BATCH_SIZE]) for i in range(0, len(queries), BATCH_SIZE)]
                )
# flatten the retrieve data
retrieve_doc = reduce(lambda acc, ele: (acc[0] + ele[0], acc[1] + ele[1], acc[2] + ele[2]), retrieve_doc)

# FORMAT INTO SUITABLE FORM FOR DATAFRAME
output_data = reduce(nonGroundOutput, zip(retrieve_doc[0], retrieve_doc[1], retrieve_doc[2]), 
                    {'Query': [], 'Top 1 Retrieve': [], 'Top 1 Score':[]}
                    )

# Output to excel file
df = pd.DataFrame(output_data)
df.to_excel(os.path.join("data", "retrieve_data.xlsx"), index=False)