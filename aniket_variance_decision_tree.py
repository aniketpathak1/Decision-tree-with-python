import pandas as pd
import numpy as np
import sys
import random
import copy
from numpy import log2 as log
import warnings
import ast
warnings.simplefilter("ignore")

def find_entropy(dataset):
        Class = dataset.keys()[-1]   
        entropy = 1.0
        values = dataset[Class].unique()
        for value in values:
                fraction = dataset[Class].value_counts()[value]/len(dataset[Class])
                entropy *= fraction
        return entropy


def find_entropy_2(dataset,attribute):
        Class = dataset.keys()[-1]  
        target_variables = dataset[Class].unique()
        variables = dataset[attribute].unique()
        entropy2 = 0
        for variable in variables:
                entropy = 1.0
                for target_variable in target_variables:
                        num = len(dataset[attribute][dataset[attribute]==variable][dataset[Class] ==target_variable])
                        den = len(dataset[attribute][dataset[attribute]==variable])
                        fraction = num/(den)
                        entropy *= (fraction)
                fraction2 = den/len(dataset)
                entropy2 += -fraction2*entropy
        return abs(entropy2)

def best_attr(dataset):
        attr = []
        for key in dataset.keys()[:-1]:
                attr.append(find_entropy(dataset)-find_entropy_2(dataset,key))
        return dataset.keys()[:-1][np.argmax(attr)]




class Node():
    def __init__(self):
        self.false= None
        self.true = None
        self.tCount = None
        self.label = None
        self.fCount = None
        self.attribute = None
        self.nodeType = None
        self.value = None 
        

    def setNodeValue(self, attribute, nodeType, value = None, tCount = None, fCount = None):
        self.attribute = attribute
        self.nodeType = nodeType
        self.tCount = tCount
        self.value = value
        self.fCount = fCount

class Buildt():
    def __init__(self):
        self.root = Node()
        self.root.setNodeValue('$@$', 'R')

    def decisionTree(self, data, tree):
        countone = data['Class'].sum()
        total = data.shape[0]
        countzero = total - countone        
        if data.shape[1] == 1 or total == countone or total == countzero:
            tree.nodeType = 'L'
            if countzero >= countone:
                tree.label = 0
            else:
                tree.label = 1
            return        
        else:        
            attbest = best_attr(data)
            tree.true = Node()
            tree.false = Node()
            tree.false.setNodeValue(attbest, 'I', 0, data[(data[attbest]==0) & (first['Class']==1) ].shape[0], data[(data[attbest]==0) & (first['Class']==0) ].shape[0])
            tree.true.setNodeValue(attbest, 'I', 1, data[(data[attbest]==1) & (first['Class']==1) ].shape[0], data[(data[attbest]==1) & (first['Class']==0) ].shape[0])
            self.decisionTree( data[data[attbest]==0].drop([attbest], axis=1), tree.false)
            self.decisionTree( data[data[attbest]==1].drop([attbest], axis=1), tree.true) 

    def labelFind(self, data, root):
        if root.label is not None:
            return root.label
        elif data[root.false.attribute][data.index.tolist()[0]] == 1:
            return self.labelFind(data, root.true)
        else:
            return self.labelFind(data, root.false)
        
    def levels(self, node,level):
        if(node.false is None and node.true is not None):
            for i in range(0,level):    
                print("| ", end="")
            level = level + 1
            print("{} = {} : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            self.levels(node.true,level)
        elif(node.true is None and node.false is not None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            self.levels(node.false,level)
        elif(node.true is None and node.false is None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
        else:
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            self.levels(node.false,level)
            self.levels(node.true,level)    

    def treeShow(self, node):
        self.levels(node.false,0)
        self.levels(node.true,0)  
               
def acc(data, tree):
    correctCount = 0
    for i in data.index:
        val = tree.labelFind(data.iloc[i:i+1, :].drop(['Class'], axis=1),tree.root)
        if val == data['Class'][i]:
            correctCount = correctCount + 1
    return correctCount/data.shape[0]*100

if __name__ == "__main__":

    args = str(sys.argv)

    args = ast.literal_eval(args)
    
    if (len(args) < 4):

        print ("only 4 input accepted, refer readme file.")

    else:

        training_set = (sys.argv[1])

        validation_set =(sys.argv[2])

        test_set = (sys.argv[3])

        to_print = str(args[4])

        first = pd.read_csv(training_set)
        first1 = pd.read_csv(test_set)
        fval = pd.read_csv(validation_set)

        dtree =Buildt()
        dtree.decisionTree(first, dtree.root)
        if (to_print.lower()=="yes"):
            print("the learned tree:\n")
            dtree.treeShow(dtree.root)
            print("")
        print("Accuracy of learned tree:\n")
        print("Accuracy percentage on the training dataset = " + str(acc(first,dtree))+"%")
        print("Accuracy percentage on the valdidation dataset = " + str(acc(fval,dtree)) + "%")
        print("Accuracy percentage on the testing dataset = "+ str(acc(first1,dtree))+"%")