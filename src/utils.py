#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import numpy as np
import os


# Calculate the contribution of each country
def calculate_contribution(row):
    countries = [country.strip() for country in row['Country'].split(',')]  # Remove whitespace
    num_countries = len(countries)
    
    if num_countries == 1:
        return {country: 1 for country in countries}  # Full contribution for single country
    else:
        return {country: 1 / num_countries for country in countries}  # Equal contribution for multiple countries

    
# Function to preprocess text and extract only nouns from the Abstract
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tagged_tokens = pos_tag(tokens)  # Perform part-of-speech tagging
    lemmatizer = WordNetLemmatizer()
    # Extract nouns, lemmatize, and filter out stopwords and short tokens
    nouns = [lemmatizer.lemmatize(token, pos='n') for token, pos in tagged_tokens 
             if pos.startswith('NN') and len(token) > 2 and token.isalnum() and token not in stop_words]
    return nouns

# Plotting international research count and ratio
def plot_international_research(research_counts):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('International Collaborative Research Count', color=color)
    ax1.bar(research_counts.index, research_counts['International'], color=color, alpha=0.7, align='center')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('International Collaborative Research Ratio', color=color)
    ax2.plot(research_counts.index, research_counts['International_Ratio'], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    plt.title('International Collaborative Research Count and Ratio per Year')
    plt.show()


 # Create a graph to visualize linked countries
def visualize_country_collaboration(df):
    G = nx.Graph()

    for countries in df['Country'].str.split(', '):
        if len(countries) > 1:
            G.add_nodes_from(countries)
            for i in range(len(countries)):
                for j in range(i + 1, len(countries)):
                    edge = (countries[i], countries[j])
                    if G.has_edge(*edge):
                        G.edges[edge]['weight'] += 1
                    else:
                        G.add_edge(*edge, weight=1)

    # Visualize the graph
    fig, ax = plt.subplots(figsize=(25, 20))
    pos = nx.spring_layout(G, seed=11)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_thickness = [weight / max(edge_weights) * 5 for weight in edge_weights]

    nx.draw(G, pos, with_labels=True, font_size=9, node_color='skyblue',
            node_size=1000, edge_color='gray', font_color='black',
            font_weight='bold', width=edge_thickness, ax=ax)

    plt.title('International Collaborative Research Network (Linked Countries Only)')
    plt.show()


# Create a graph to visualize linked countries
def visualize_top_countries(df, top_countries):
    G = nx.Graph()

    for countries in df['Country'].str.split(', '):
        if len(countries) > 1:
            G.add_nodes_from(countries)
            for i in range(len(countries)):
                for j in range(i + 1, len(countries)):
                    edge = (countries[i], countries[j])
                    if G.has_edge(*edge):
                        G.edges[edge]['weight'] += 1
                    else:
                        G.add_edge(*edge, weight=1)

    # Select the top N countries based on their collaborations
    top_countries_list = df['Country'].str.split(', ').apply(lambda x: x[:top_countries] if len(x) > 1 else x).tolist()
    top_countries_set = set(country for countries in top_countries_list for country in countries)

    # Create a subgraph including only the selected countries
    subgraph = G.subgraph(top_countries_set)

    # Select the edges with the highest weights for the top N countries
    edge_weights = [(u, v, subgraph[u][v]['weight']) for u, v in subgraph.edges()]
    edge_weights.sort(key=lambda x: x[2], reverse=True)
    selected_edges = edge_weights[:top_countries]

    # Create a subgraph for the selected nodes and their edges
    selected_nodes = set(u for u, _, _ in selected_edges) | set(v for _, v, _ in selected_edges)
    selected_subgraph = G.subgraph(selected_nodes)

    # Visualize the graph
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(selected_subgraph, seed=11)
    max_weight = max(selected_subgraph[u][v]['weight'] for u, v in selected_subgraph.edges())
    edge_thickness = [selected_subgraph[u][v]['weight'] / max_weight * 5 for u, v in selected_subgraph.edges()]

    nx.draw(selected_subgraph, pos, with_labels=True, font_size=9,
            node_color='skyblue', node_size=1000, edge_color='gray',
            font_color='black', font_weight='bold', width=edge_thickness, ax=ax)

    plt.title(f'Top {top_countries} Countries with Strong Collaborative Research Network (Linked Countries Only)')
    plt.show()


def visualize_selected_country(df, selected_country):
    """
    Visualizes the collaborative research network for a selected country.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing a 'Country' column.
    selected_country (str): The country to visualize collaborations for.
    """
    # Handle NaN values in the 'Country' column and convert to string
    df['Country'] = df['Country'].fillna('').astype(str)

    # Create a graph to visualize linked countries
    G = nx.Graph()

    # Extract country information from each row and add to the graph
    for countries in df['Country'].str.split(', '):
        if selected_country in countries:
            G.add_nodes_from(countries)  # Add countries as nodes
            for i in range(len(countries)):
                for j in range(i + 1, len(countries)):
                    edge = (countries[i], countries[j])
                    if G.has_edge(*edge):
                        # Increment edge weight if the edge already exists
                        G.edges[edge]['weight'] += 1
                    else:
                        # Add a new edge with weight 1
                        G.add_edge(*edge, weight=1)

    # Visualize the graph
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=11)  # Set graph layout

    # Adjust edge thickness based on the number of collaborations (edge weight)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_thickness = [weight / max(edge_weights) * 5 for weight in edge_weights]

    # Draw the graph
    nx.draw(G, pos, with_labels=True, font_size=9, 
            node_color='skyblue', node_size=700, 
            edge_color='gray', font_color='black', 
            font_weight='bold', width=edge_thickness, ax=ax)

    plt.title(f'Collaborative Research Network of {selected_country}')
    plt.show()


def visualize_author_collaborations(df, selected_author):
    """
    Visualizes the collaboration network for a selected author and their co-authors.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing an 'Authors' column.
    selected_author (str): The author to visualize collaborations for.
    """
    # Handle NaN values in the 'Authors' column and convert to string
    df['Authors'] = df['Authors'].fillna('').astype(str)

    # Extract data related to the selected author
    df_selected_author = df[df['Authors'].str.contains(selected_author)]

    # Create a graph for linked authors
    G = nx.Graph()

    # Extract author information from each row and add to the graph
    for authors in df_selected_author['Authors'].str.split(', '):
        if len(authors) > 1:  # Only consider rows with multiple authors
            G.add_nodes_from(authors)
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    edge = (authors[i], authors[j])
                    if G.has_edge(*edge):
                        # Increment edge weight if the edge already exists
                        G.edges[edge]['weight'] += 1
                    else:
                        # Add a new edge with weight 1
                        G.add_edge(*edge, weight=1)

    # Select the top 30 strongest collaborations among authors
    top_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:30]
    G_top = nx.Graph(top_edges)

    # Visualize the graph
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G_top, seed=16)  # Set graph layout

    # Adjust edge thickness based on the number of collaborations (edge weight)
    edge_weights = [G_top[u][v]['weight'] for u, v in G_top.edges()]
    edge_thickness = [weight / max(edge_weights) * 5 for weight in edge_weights]

    # Draw the graph
    nx.draw(G_top, pos, with_labels=True, font_size=7, 
             node_color='skyblue', node_size=1000, 
             edge_color='gray', font_color='black', 
             font_weight='bold', width=edge_thickness, ax=ax)

    plt.title(f'Collaborations among Authors ({selected_author} and Co-authors)')
    plt.show()


# Keyword centralities
def build_graph(keywords_list):
    """ Build a graph from the given list of keywords. """
    G = nx.Graph()
    for keywords in keywords_list:
        if len(keywords) > 1:
            G.add_nodes_from(keywords)
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    edge = (keywords[i], keywords[j])
                    if G.has_edge(*edge):
                        # Increment edge weight if the edge already exists
                        G.edges[edge]['weight'] += 1
                    else:
                        # Add a new edge with weight 1
                        G.add_edge(*edge, weight=1)
    return G

def calculate_centrality(G):
    """ Calculate centrality measures and convert results to a DataFrame. """
    # Calculate Degree Centrality
    degree_centrality = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_centrality, 'degree_centrality')

    # Calculate Closeness Centrality
    closeness_centrality = nx.closeness_centrality(G)
    nx.set_node_attributes(G, closeness_centrality, 'closeness_centrality')

    # Calculate Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')

    # Calculate Eigenvector Centrality
    eigenvector_centrality = nx.eigenvector_centrality(G)
    nx.set_node_attributes(G, eigenvector_centrality, 'eigenvector_centrality')

    # Convert results to a DataFrame
    centrality_df = pd.DataFrame({
        'Keyword': list(G.nodes),
        'Degree Centrality': list(nx.get_node_attributes(G, 'degree_centrality').values()),
        'Closeness Centrality': list(nx.get_node_attributes(G, 'closeness_centrality').values()),
        'Betweenness Centrality': list(nx.get_node_attributes(G, 'betweenness_centrality').values()),
        'Eigenvector Centrality': list(nx.get_node_attributes(G, 'eigenvector_centrality').values())
    })
    return centrality_df


# Function to preprocess text and extract only nouns from the Abstract
def preprocess_text(text, stop_words):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tagged_tokens = pos_tag(tokens)  # Perform part-of-speech tagging
    lemmatizer = WordNetLemmatizer()
    # Extract nouns, lemmatize, and filter out stopwords and short tokens
    nouns = [lemmatizer.lemmatize(token, pos='n') for token, pos in tagged_tokens 
             if pos.startswith('NN') and len(token) > 2 and token.isalnum() and token not in stop_words]
    return nouns

# Function to evaluate LDA model
def evaluate_lda(corpus, dictionary, df, start, end):
    results_df = pd.DataFrame(columns=['Topics', 'Perplexity', 'Coherence'])
    
    for num_topics in range(start, end + 1):
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=1)
        perplexity = lda_model.log_perplexity(corpus)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=df['nouns'], dictionary=dictionary, coherence='c_v')
        coherence = coherence_model_lda.get_coherence()
        
        # Add results to the DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({'Topics': [num_topics], 'Perplexity': [perplexity], 'Coherence': [coherence]})], ignore_index=True)

    return results_df



