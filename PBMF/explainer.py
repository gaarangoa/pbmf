import networkx as nx
import tensorflow as tf
import pandas as pd
import numpy as np 


from sklearn import metrics
from sklearn.inspection import permutation_importance
from lifelines.utils import concordance_index as cindex
from lifelines import CoxPHFitter

def Grad4Input(model, X, layer=None):
    # Convert the input data to a TensorFlow tensor
    X = tf.constant(X, dtype=tf.float32)
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [i.output for i in model.layers]
    )

    with tf.GradientTape() as tape:
        tape.watch(X)
        latent = grad_model(X)
        outputs = latent[-1]
    
    if layer:
        gradients = tape.gradient(outputs, latent[layer])
        feature_importance = np.abs(np.mean(gradients * latent[layer], axis=0))
    else:
        gradients = tape.gradient(outputs, X)
        feature_importance = np.abs(np.mean(gradients * X, axis=0))
    
    return feature_importance



def score_auc(model, X, y):
    pred = model.predict(X)
    try:
        pred = pred[:, 0]
    except Exception as inst:
        print('Error: {}'.format())

    fpr, tpr, thresholds = metrics.roc_curve(
        y, 
        pred, 
        pos_label=list(set(y))[0]
    )
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc

def score_cindex(model, X, y):
    y = np.array(y)
    pred = model.predict(X)
    try:
        pred = pred[:, 0]
    except:
        pass
    
    idx = (y[:, 3] == 1)
    ci = cindex(y[idx, 0], pred[idx], y[idx, 1])
    
    return ci
    
def score_hr(model, X, y, time, event):
    
    pred = model.predict(X)
    try:
        pred = pred[:, 0]
    except:
        pass
    
    if y.loc[pred > 0, time].mean() > y.loc[pred < 0, time].mean():
        y['risk'] = pred > 0
    else:
        y['risk'] = pred < 0
    
    cph = CoxPHFitter().fit(y.loc[y.treatment == 1, [time, event, 'risk']], time, event)
    
    return cph.summary['exp(coef)'].values[0]




class Explainer():
    '''Deprecated'''
    def __init__(self, model):
        self.model = model
        self.grad_model = tf.keras.models.Model(
            [self.model.inputs], [i.output for i in self.model.layers]
        )
        self.embedding_size = list(self.model.embedding_size)
        self.layers = len(self.embedding_size)
        self.weights = model.get_weights()
    
    def gradCAM(self, data):
        grads = pd.concat([self.get_grads(self.grad_model, i, data) for i in range(len(self.embedding_size))], axis=1)        
        return grads
        
    def get_grads(self, grad_model, layer, data):
        with tf.GradientTape() as tape:
            latent_preds = self.grad_model(data)
            class_channel = latent_preds[-1][:, 0]

        grads = tape.gradient(class_channel, latent_preds[layer])
        grads = grads * latent_preds[layer]
        grads = grads.numpy()

        grads = pd.DataFrame(grads, columns=['Layer:{}_{}'.format(layer, i) for i in range(grads.shape[1])])

        return grads
    
    def plot(self, dgrad, features, ax, **kwargs):
        
        alpha = kwargs.get('alpha', 0.1)
        input_nodes = features
        output_nodes = ['risk']
        hidden_nodes = list(dgrad.columns[dgrad.columns.str.match('Layer:')])
        max_units = 0#np.max(self.model.embedding_size)
        scale_nodes = kwargs.get('scale_nodes', 1)
        scale_edges = kwargs.get('scale_edges', 1)
        positive_color = kwargs.get('positive_color', '#ed3b50')
        negative_color = kwargs.get('negative_color', 'darkblue')
        
        dgrad = pd.DataFrame(dgrad.mean()).T

        G = nx.DiGraph()
        G.add_nodes_from(input_nodes+hidden_nodes+output_nodes)

        pos_x = [[i, np.array(i.replace('Layer:', '').split('_'), dtype=int)] for i in hidden_nodes]
        pos = {i:(xi, yi)  for i, [xi, yi] in pos_x}

        pos.update({i:(-1, ix) for ix,i in enumerate(input_nodes)})
        pos.update({i:(self.layers, ix) for ix,i in enumerate(output_nodes)})

        node_weights = {i:scale_nodes * np.abs(j.values[0]) for i,j in dgrad[hidden_nodes].items()}
        node_weights.update({i: 10 for i in features})
        node_weights.update({i: scale_nodes * np.abs(j.values[0]) for i,j in dgrad[output_nodes].items()})

        node_colors = {i:positive_color if j.values[0] > 0 else negative_color for i,j in dgrad[hidden_nodes].items()}
        node_colors.update({i: positive_color for i in input_nodes})
        node_colors.update({i: positive_color for i in output_nodes})


        edges = []
        edge_weights = {}
        edge_colors = {}
        hidden_nodes_0 = [i for i in hidden_nodes if 'Layer:0_' in i]
        wl = self.weights[0]
        for ix,i in enumerate(input_nodes):
            for j in hidden_nodes_0:
                edges.append((i, j))
                edge_weights[(i, j )] = scale_edges * np.abs(wl[ix, int(j.split('_')[1])])
                edge_colors[(i, j )] = positive_color if wl[ix, int(j.split('_')[1])] > 0 else negative_color

        for l in range(self.layers-1):
            wl = self.weights[2*(l+1)]
            for i in range(self.embedding_size[l]):
                for j in range(self.embedding_size[l+1]):
                    e = (("Layer:{}_{}".format(l, i)), "Layer:{}_{}".format(l+1, j))

                    edge_weights[e] = scale_edges * np.abs(wl[i, j])
                    edge_colors[e] = positive_color if wl[i, j] > 0 else negative_color
                    edges.append(e)

        wl = self.weights[-1]
        for i in range(self.embedding_size[-1]):
            for jx,j in enumerate(output_nodes):
                e = ("Layer:{}_{}".format(self.layers-1, i), j)

                edge_weights[e] = scale_edges * np.abs(wl[i, jx])
                edge_colors[e] = positive_color if wl[i, jx] > 0 else negative_color
                edges.append(e)

        G.add_edges_from(edges)


        # Define the edge widths based on edge weights
        edge_widths = [edge_weights[edge] for edge in G.edges]
        edge_color = [edge_colors[edge] for edge in G.edges]

        # Draw the nodes and edges with node sizes and edge widths
        nx.draw_networkx_nodes(
            G, pos=pos, 
            node_color=[node_colors[node] for node in G.nodes], 
            node_size=[node_weights[node] * 1 for node in G.nodes],
            ax=ax
        )
        nx.draw_networkx_edges(
            G, pos=pos, ax=ax, edge_color=edge_color, 
            width=edge_widths, connectionstyle=kwargs.get('connectionstyle', 'arc3,rad=0.0'), 
            alpha=alpha, arrowstyle='-', style='dashed', arrowsize=5, 
            node_size=100
        )