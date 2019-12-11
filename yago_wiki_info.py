#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:12:57 2019

@author: ashik
"""

# -*- coding: utf-8 -*-

import csv
import pandas as pd
from pandas import DataFrame
from rdflib import Graph as RDFGraph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
from networkx import Graph as NXGraph
import matplotlib.pyplot as plt
import statistics
import collections
import os

df_wiki_info = pd.read_csv('yagoWikipediaInfo_fa.tsv', sep='\t')
          
G = nx.Graph()

df = DataFrame(columns={'subject', 'predicate', 'object'})
    
for i in range (5000):
#for i in range (len(df_wiki_info)):
    
    row = df_wiki_info.iloc[i, :]
    G.add_edge(row[1], row[3])
    G[row[1]][row[3]]['predicate'] = row[2]
    #df = df.append([{ 'subject': row[1], 'predicate': row[2], 'object': row[3]}])
    #print(i)
    
#df_nodes = list(G.nodes())
#df_edges = list(G.edges())
#df_edge_data = list(G.edges.data())
    
options = {
    'node_color': 'green',
    'edge_color': 'black',
    'node_size': 10,
    'width': 0.1,
    'alpha': 1,
}
plt.figure(figsize=(10,10))
nx.draw(G, **options, with_labels=False)

#nG = nx.Graph(list(G.edges.data())[:10])
#nx.draw(nG, **options, with_labels=False)


#Number of edges
E = G.number_of_edges()
print('Number of edges:', E)

#Number of nodes
V = G.number_of_nodes()
print('Number of nodes:', V)

#Graph density
print('Graph density:', nx.classes.function.density(G))    

#Global clustering co-efficient
clustering_coefficients = []
global_clustering_coefficients = nx.clustering(G)
#df_gcc = DataFrame(columns={'node', 'clustering_coefficient'})
for node, cc in global_clustering_coefficients.items():
#    df_gcc = df_gcc.append({'node': node, 'clustering_coefficient': cc}, ignore_index=True)
    clustering_coefficients.append({'node': node,
                                    'clustering coefficient': cc})

for row in clustering_coefficients:
    #print('node: {} - Clustering coefficient: {}'.format(row['node'], row['clustering coefficient']))
    print('{}: {}'.format(row['node'], row['clustering coefficient']))
    

#Average degree
avg_degree = 2*E/V
print('Average degree:', avg_degree)


#Diameter
subgraph_list = []

for conn_component in nx.connected_components(G):
    subgraph_list.append(conn_component)

print('Number of subgraphs: {}'.format(len(subgraph_list)))

for i in range (len(subgraph_list)):
    subgraph = nx.Graph()
    node_list = list(subgraph_list[i])
    for node in node_list:
        subgraph.add_edges_from(G.edges(node))
    
    print('Subgraph_{} diameter : {}'.format(i+1, nx.diameter(subgraph)))
    
    
#Degree distribution
degrees = [G.degree(n) for n in G.nodes()]
degrees.sort()
degree_distribution = collections.Counter(degrees)
plt.hist(degrees, bins=max(degrees))
plt.title('Degree distribution histogram')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.figure(figsize=(15,15))
plt.show()
print('Max degree: {}'.format(max(degrees)))
print(degree_distribution)
#for degree in degree_distribution:
#    print('{} : {}'.format(degree, degree_distribution[degree]))



#Frequency distribution
predicate_list = [c for a,b,c in G.edges.data('predicate')]
freq_distribution = collections.Counter(predicate_list)
plt.hist(predicate_list, bins=len(set(predicate_list)))
plt.title('Frequency distribution histogram')
plt.xlabel('Predicate')
plt.ylabel('Number of edges')
#for index,data in enumerate(predicate_list):
#    plt.text(x=index, y =data+1, s=f'{data}', fontdict=dict(fontsize=20))
plt.figure(figsize=(15,15))
plt.show()
print('Predicate with maximum frequency: {}'.format(max(predicate_list)))
for predicate in freq_distribution:
    print('{} : {}'.format(predicate, freq_distribution[predicate]))
    

#rank-frequency plot
rf_table = []
degree = []
frequency = []
degree_distribution_list = degree_distribution.most_common(len(degree_distribution))
top_frequency = degree_distribution_list[0][1]
for index, item in enumerate(degree_distribution_list, start=1):
    rf_table.append({'rank': index,
                     'degree': item[0],
                     'frequency': item[1],
                     'relative frequency':'1/{}'.format(index)})
    
freq_list = list(degree_distribution.values())
#Zipf distribution parameter
#a = 3 
#count, bins, ignored = plt.hist(freq_list, len(freq_list), normed=True)
#plt.title("Zipf plot")
#x = np.arange(1, len(freq_list))
#plt.xlabel("Frequency Rank")
#y = x**(-a) / special.zetac(a)
#plt.ylabel("Absolute Frequency")
#plt.plot(x, y/max(y), linewidth=2, color='r')
#plt.show()
#changed
rank = [row['rank'] for row in rf_table]
freq = [row['frequency'] for row in rf_table]

plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.loglog(rank, freq, basex=10)

print("|  Rank    |  degree    |  Frequency |Zipf fraction|")
format_string = "|{:10}|{:12}|{:12.0f}|{:>12}|"
for index, item in enumerate(rf_table, start=1):
    print(format_string.format(item["rank"],
                               item["degree"],
                               item["frequency"],
                               item["relative frequency"]))