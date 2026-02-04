# 统计每个社区中社区的分布情况
from preprocess_dblp import untar_snap_data
from preprocess_email import untar_email_data
from preprocess_football import untar_football_data
from preprocess_facebook import untar_facebook_data
import networkx as nx
import random
import numpy as np
import pickle
import os
import sys

def compute_triangle_count(graph):
    nodes = set(graph.nodes())

    # 统计子图中的三角形数量
    triangle_count = 0
    for node in nodes:
        neighbors = set(graph.neighbors(node))
        for neighbor1 in neighbors:
            for neighbor2 in neighbors:
                if neighbor1 < neighbor2 and graph.has_edge(neighbor1, neighbor2):
                    triangle_count += 1

    return triangle_count // 3 

def compute_k_truss_values(graph):
    # 获取子图的边
    edges = list(graph.edges())

    # 统计三角形数量
    triangle_count = {}
    for edge in edges:
        neighbors = set(graph.neighbors(edge[0])) & set(graph.neighbors(edge[1]))
        for common_neighbor in neighbors:
            triangle = tuple(sorted([edge[0], edge[1], common_neighbor]))
            triangle_count[triangle] = triangle_count.get(triangle, 0) + 1

    # 获取所有边的k-truss值
    k_truss_values = list(triangle_count.values())

    return k_truss_values

def compute_distances(graph, subgraph):
    nodes_G_prime = list(subgraph.nodes())
    all_shortest_paths = []
    for i in range(len(nodes_G_prime)):
        for j in range(i + 1, len(nodes_G_prime)):
            shortest_path = nx.shortest_path(graph, source=nodes_G_prime[i], target=nodes_G_prime[j])
            all_shortest_paths.append(shortest_path)

    # 计算平均距离、最小距离和最大距离
    average_distance = sum(len(path) - 1 for path in all_shortest_paths) / len(all_shortest_paths)
    min_distance = min(len(path) - 1 for path in all_shortest_paths)
    max_distance = max(len(path) - 1 for path in all_shortest_paths)
    print(f"Min dis: {min_distance}")
    print(f"Max dis: {max_distance}")
    print(f"Average dis: {average_distance}")
    return average_distance, min_distance, max_distance


def print_truss_info(graph):
    triangle_count = compute_triangle_count(graph)
    print("triangle num: {}".format(triangle_count))
    k_truss_values = compute_k_truss_values(graph)

    if k_truss_values:
        min_k_truss = min(k_truss_values)
        max_k_truss = max(k_truss_values)
        avg_k_truss = sum(k_truss_values) / len(k_truss_values)
        print(f"Min k-truss: {min_k_truss}")
        print(f"Max k-truss: {max_k_truss}")
        print(f"Average k-truss: {avg_k_truss}")
    else:
        print("No k-truss.")


def load_dblp_graphs():
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)
    path=os.getcwd()+".\\Cml-main\\data\\DBLP"

    print(f"Load com_dblp edges")
    if(os.path.exists(path + '//edges.pkl') == False):
        untar_snap_data('dblp')
    with open(path+'//edges.pkl', 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print(f"Load com_dblp cmty")
    with open(path+'//comms.pkl', 'rb') as file:
        com_list = pickle.load(file)
    # print(com_list)


    for comm in com_list:
        subgraph = graph.subgraph(comm)
        is_connected = nx.is_connected(subgraph)
        print("********************************************")
        print("comm with {} nodes is conected?{}".format(len(subgraph.nodes()),is_connected))
        if is_connected:
            print_truss_info(subgraph)
            connected_components = list(nx.connected_components(subgraph))

            # 找到包含给定节点的连通分量
            for component in connected_components:
                if set(comm).issubset(component):
                    # 创建包含给定节点的连通子图
                    connected_subgraph = subgraph.subgraph(component)
                    # 计算连通子图的图直径
                    diameter = nx.diameter(connected_subgraph)
                    # 计算最大、最小和平均度数
                    degrees = dict(connected_subgraph.degree())
                    max_degree = max(degrees.values())
                    min_degree = min(degrees.values())
                    avg_degree = sum(degrees.values()) / len(connected_subgraph)

                    print("Min degree of graph:{}".format(min_degree))
                    print("Max degree of graph:{}".format(max_degree))
                    print("Avg degree of graph:{}".format(avg_degree))
                    print("diameter of graph:{}".format(diameter))
                    if(len(subgraph.nodes())>1):
                        compute_distances(graph,subgraph)
        else:
            connected_components = list(nx.connected_components(subgraph))
            largest_connected_component = max(connected_components, key=len)
            largest_connected_subgraph = graph.subgraph(largest_connected_component)
            print_truss_info(largest_connected_subgraph)
            # print(comm)
            diameter = nx.diameter(largest_connected_subgraph)
            degrees = dict(largest_connected_subgraph.degree())
            max_degree = max(degrees.values())
            min_degree = min(degrees.values())
            avg_degree = sum(degrees.values()) / len(largest_connected_subgraph)

            print("Min degree of graph:{}".format(min_degree))
            print("Max degree of graph:{}".format(max_degree))
            print("Avg degree of graph:{}".format(avg_degree))
            print("diameter of graph:{}".format(diameter))
            if(len(largest_connected_subgraph.nodes())>1):
                compute_distances(graph,largest_connected_subgraph)



def load_email_graphs():
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)
    path=os.getcwd()+"\\Cml-main\\data\\email-Eu"

    print(f"Load email-Eu edges")
    if(os.path.exists(path + '//edges.pkl') == False):
        untar_email_data('email-Eu-core')
    with open(path+'//edges.pkl', 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    
    print(f"Load email-Eu cmty")
    with open(path+'//comms.pkl', 'rb') as file:
        com_list = pickle.load(file)


    for comm in com_list:
        subgraph = graph.subgraph(comm)
        is_connected = nx.is_connected(subgraph)
        print("********************************************")
        print("comm with {} nodes is conected?{}".format(len(subgraph.nodes()),is_connected))
        if is_connected:
            print_truss_info(subgraph)
            connected_components = list(nx.connected_components(subgraph))

            # 找到包含给定节点的连通分量
            for component in connected_components:
                if set(comm).issubset(component):
                    # 创建包含给定节点的连通子图
                    connected_subgraph = subgraph.subgraph(component)
                    # 计算连通子图的图直径
                    diameter = nx.diameter(connected_subgraph)
                    # 计算最大、最小和平均度数
                    degrees = dict(connected_subgraph.degree())
                    max_degree = max(degrees.values())
                    min_degree = min(degrees.values())
                    avg_degree = sum(degrees.values()) / len(connected_subgraph)

                    print("Min degree of graph:{}".format(min_degree))
                    print("Max degree of graph:{}".format(max_degree))
                    print("Avg degree of graph:{}".format(avg_degree))
                    print("diameter of graph:{}".format(diameter))
                    if(len(subgraph.nodes())>1):
                        compute_distances(graph,subgraph)
        else:
            connected_components = list(nx.connected_components(subgraph))
            largest_connected_component = max(connected_components, key=len)
            largest_connected_subgraph = graph.subgraph(largest_connected_component)
            print_truss_info(largest_connected_subgraph)
            # print(comm)
            diameter = nx.diameter(largest_connected_subgraph)
            degrees = dict(largest_connected_subgraph.degree())
            max_degree = max(degrees.values())
            min_degree = min(degrees.values())
            avg_degree = sum(degrees.values()) / len(largest_connected_subgraph)

            print("Min degree of graph:{}".format(min_degree))
            print("Max degree of graph:{}".format(max_degree))
            print("Avg degree of graph:{}".format(avg_degree))
            print("diameter of graph:{}".format(diameter))
            if(len(largest_connected_subgraph.nodes())>1):
                compute_distances(graph,largest_connected_subgraph)


def load_football_graphs():
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)
    path=os.getcwd()+"\\Cml-main\\data\\Football"

    print(f"Load football edges")
    if(os.path.exists(path + '//edges.pkl') == False):
        untar_football_data('football')
    with open(path+'//edges.pkl', 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    
    print(f"Load football cmty")
    with open(path+'//comms.pkl', 'rb') as file:
        com_list = pickle.load(file)


    for comm in com_list:
        subgraph = graph.subgraph(comm)
        is_connected = nx.is_connected(subgraph)
        print("********************************************")
        print("comm with {} nodes is conected?{}".format(len(subgraph.nodes()),is_connected))
        if is_connected:
            print_truss_info(subgraph)
            connected_components = list(nx.connected_components(subgraph))

            # 找到包含给定节点的连通分量
            for component in connected_components:
                if set(comm).issubset(component):
                    # 创建包含给定节点的连通子图
                    connected_subgraph = subgraph.subgraph(component)
                    # 计算连通子图的图直径
                    diameter = nx.diameter(connected_subgraph)
                    # 计算最大、最小和平均度数
                    degrees = dict(connected_subgraph.degree())
                    max_degree = max(degrees.values())
                    min_degree = min(degrees.values())
                    avg_degree = sum(degrees.values()) / len(connected_subgraph)

                    print("Min degree of graph:{}".format(min_degree))
                    print("Max degree of graph:{}".format(max_degree))
                    print("Avg degree of graph:{}".format(avg_degree))
                    print("diameter of graph:{}".format(diameter))
                    if(len(subgraph.nodes())>1):
                        compute_distances(graph,subgraph)
        else:
            connected_components = list(nx.connected_components(subgraph))
            largest_connected_component = max(connected_components, key=len)
            largest_connected_subgraph = graph.subgraph(largest_connected_component)
            print_truss_info(largest_connected_subgraph)
            # print(comm)
            diameter = nx.diameter(largest_connected_subgraph)
            degrees = dict(largest_connected_subgraph.degree())
            max_degree = max(degrees.values())
            min_degree = min(degrees.values())
            avg_degree = sum(degrees.values()) / len(largest_connected_subgraph)

            print("Min degree of graph:{}".format(min_degree))
            print("Max degree of graph:{}".format(max_degree))
            print("Avg degree of graph:{}".format(avg_degree))
            print("diameter of graph:{}".format(diameter))
            if(len(largest_connected_subgraph.nodes())>1):
                compute_distances(graph,largest_connected_subgraph)

def load_facebook_graphs():
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)
    path=os.getcwd()+"\\Cml-main\\data\\Facebook"

    print(f"Load facebook edges")
    if(os.path.exists(path + '//edges.pkl') == False):
        untar_facebook_data('facebook')
    with open(path+'//edges.pkl', 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    
    print(f"Load facebook cmty")
    with open(path+'//comms.pkl', 'rb') as file:
        com_list = pickle.load(file)
    # print(nx.is_connected(graph))

    for comm in com_list:
        subgraph = graph.subgraph(comm)
        is_connected = nx.is_connected(subgraph)
        print("********************************************")
        print("comm with {} nodes is conected?{}".format(len(subgraph.nodes()),is_connected))
        if is_connected:
            print_truss_info(subgraph)
            connected_components = list(nx.connected_components(subgraph))

            # 找到包含给定节点的连通分量
            for component in connected_components:
                if set(comm).issubset(component):
                    # 创建包含给定节点的连通子图
                    connected_subgraph = subgraph.subgraph(component)
                    # 计算连通子图的图直径
                    diameter = nx.diameter(connected_subgraph)
                    # 计算最大、最小和平均度数
                    degrees = dict(connected_subgraph.degree())
                    max_degree = max(degrees.values())
                    min_degree = min(degrees.values())
                    avg_degree = sum(degrees.values()) / len(connected_subgraph)

                    print("Min degree of graph:{}".format(min_degree))
                    print("Max degree of graph:{}".format(max_degree))
                    print("Avg degree of graph:{}".format(avg_degree))
                    print("diameter of graph:{}".format(diameter))
                    if(len(subgraph.nodes())>1):
                        compute_distances(graph,subgraph)
        else:
            connected_components = list(nx.connected_components(subgraph))
            largest_connected_component = max(connected_components, key=len)
            largest_connected_subgraph = graph.subgraph(largest_connected_component)
            print_truss_info(largest_connected_subgraph)
            # print(comm)
            diameter = nx.diameter(largest_connected_subgraph)
            degrees = dict(largest_connected_subgraph.degree())
            max_degree = max(degrees.values())
            min_degree = min(degrees.values())
            avg_degree = sum(degrees.values()) / len(largest_connected_subgraph)

            print("Min degree of graph:{}".format(min_degree))
            print("Max degree of graph:{}".format(max_degree))
            print("Avg degree of graph:{}".format(avg_degree))
            print("diameter of graph:{}".format(diameter))
            if(len(largest_connected_subgraph.nodes())>1):
                compute_distances(graph,largest_connected_subgraph)
                
load_dblp_graphs()
# load_email_graphs()
# load_football_graphs()
# load_facebook_graphs()

