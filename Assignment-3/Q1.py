from numpy import linalg
from numpy.core.einsumfunc import _optimal_path
from numpy.core.numeric import normalize_axis_tuple
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import sys
import heapq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class Node:
    def __init__(self, data, par = None, leaf = False, testData = None, validationData = None):
        self.data = data
        self.madeLeaf_during_pruning = False
        self.deleted_during_pruning = False
        self.correct_in_subtree_validation = 0
        self.correct_in_subtree_train = 0
        self.correct_in_subtree_test = 0
        self.val_delta_correct = 0
        self.testData = testData
        self.testCorrect = 0
        self.trainCorrect = 0
        self.validationData = validationData
        self.validationCorrect = 0
        self.par = par
        self.children = []
        self.median = None
        self.isLeaf = False
        self.feature = None
        self.ClFreq = {1:0, 0:0}
        for i in range(data.shape[0]):
            # print(data[i, -1])
            if data[i,-1] == 1:
                self.ClFreq[1] +=1
            if data[i,-1] == 0:
                self.ClFreq[0] +=1
        self.label = None
        if self.ClFreq[1]>self.ClFreq[0]:
            self.label = 1
        else:
            self.label = 0

    def __lt__(self, node):
        return self.correct_in_subtree_validation < node.correct_in_subtree_validation
    

class DecisionTree:
    def __init__(self, categorical, val_to_int, training_raw_Data, one_hot = False, numerical = []):
        self.listNodes = []
        self.listLeaves = []
        self.max_nodes = 800000
        self.minDataSize = 20
        self.num_nodes = 1
        self.hot_encode = one_hot
        self.numericals = numerical
        self.nFeatures = training_raw_Data.shape[1] - 1
        self.categorical = categorical
        self.val_to_int = val_to_int
        self.trainData = self.preprocess(training_raw_Data)
        self.root = Node(self.trainData)
        # print(self.root.ClFreq)
        self.tree = self.buildDecisionTree()


    
    def preprocess(self, data):
        if self.hot_encode:
            return data
        processed = np.zeros(data.shape)
        m,n = data.shape
        for i in range(m):
            for j in range(n):
                if j == n-1:
                    if data[i,j] == "yes":
                        processed[i,j] = 1
                    else:
                        processed[i,j] = 0
                    continue
                if j in self.categorical:
                    if data[i,j] in self.val_to_int[j]:
                        processed[i,j] = self.val_to_int[j][data[i, j]]
                    else:
                        processed[i,j] = 0
                else:
                    processed[i,j] = float(data[i,j])
        return processed

    def numerical_feature_entropy(self, node, feature):
        sortedData = node.data[node.data[:, feature].argsort()]
        nVal = 2
        counts = [0 for i in range(nVal)]
        posCounts = [0 for i in range(nVal)]
        negCounts = [0 for i in range(nVal)]
        m = node.data.shape[0]
        median = sortedData[int(m/2), feature]
        for i in range(node.data.shape[0]):
            if i < int(m/2):
                counts[0] += 1
                if sortedData[i, -1] == 1:
                    posCounts[0] += 1
                else:
                    negCounts[0] += 1
            else:
                counts[1] += 1
                if sortedData[i, -1] == 1:
                    posCounts[1] += 1
                else:
                    negCounts[1] += 1
        entropy = 0
        for i in range(nVal):
            if counts[i] == 0:
                p1 = 0
            else:
                p1 = posCounts[i]/counts[i]
            p0 = 1-p1
            if p1>0:
                entropy -= (counts[i]/m)*(p1*np.log(p1))
            if p0>0:
                entropy -= (counts[i]/m)*(p0*np.log(p0)) 
        return entropy, median

    def categorical_feature_entropy(self, node, feature):
        if self.hot_encode:
            nVal = 2
            counts = [0 for i in range(nVal)]
            posCounts = [0 for i in range(nVal)]
            negCounts = [0 for i in range(nVal)]
            m = node.data.shape[0]
            for i in range(node.data.shape[0]):
                counts[int(node.data[i, feature])] += 1
                if node.data[i, -1] == 1:
                    posCounts[int(node.data[i, feature])] += 1
                else:
                    negCounts[int(node.data[i, feature])] += 1
            entropy = 0
            for i in range(nVal):
                if counts[i] == 0:
                    p1 = 0
                else:
                    p1 = posCounts[i]/counts[i]
                p0 = 1-p1
                if p1>0:
                    entropy -= (counts[i]/m)*(p1*np.log(p1))
                if p0>0:
                    entropy -= (counts[i]/m)*(p0*np.log(p0)) 
            return entropy



        entropy = 0
        nVal = len(self.val_to_int[feature])
        counts = [0 for i in range(nVal)]
        posCounts = [0 for i in range(nVal)]
        negCounts = [0 for i in range(nVal)]
        m = node.data.shape[0]
        for i in range(node.data.shape[0]):
            counts[int(node.data[i, feature])] += 1
            if node.data[i, -1] == 1:
                posCounts[int(node.data[i, feature])] += 1
            else:
                negCounts[int(node.data[i, feature])] += 1

        for i in range(nVal):
            if counts[i] == 0:
                p1 = 0
            else:
                p1 = posCounts[i]/counts[i]
            p0 = 1-p1
            if p1>0:
                entropy -= (counts[i]/m)*(p1*np.log(p1))
            if p0>0:
                entropy -= (counts[i]/m)*(p0*np.log(p0)) 
        return entropy

    def getSplittedData(self, node, feature):
        if self.hot_encode:
            if feature not in self.numericals:
                data = node.data
                childrenData = [data[data[:, feature] == 0], data[data[:, feature] == 1]]
                return childrenData
            else:
                sortedData = node.data[node.data[:, feature].argsort()]
                m = node.data.shape[0]
                childrenData = [sortedData[:int(m/2),:], sortedData[int(m/2):,:]]
                return childrenData

        if feature in self.categorical:
            childrenData = []
            nVal = len(self.val_to_int[feature])
            for i in range(nVal):
                req = node.data[:, feature] == i
                childrenData.append(node.data[req])
            return childrenData
        else:
            sortedData = node.data[node.data[:, feature].argsort()]
            m = node.data.shape[0]
            childrenData = [sortedData[:int(m/2),:], sortedData[int(m/2):,:]]
            return childrenData


    def bestFeatureToSplit(self, node):
        if self.hot_encode:
            bestFeature = -1
            bestEntropy = np.inf
            req_median = None
            for i in range(self.nFeatures):
                if i not in self.numericals:
                    entropy = self.categorical_feature_entropy(node, i)
                    if entropy<bestEntropy:
                        bestEntropy = entropy
                        bestFeature = i
                else:
                    entropy, median = self.numerical_feature_entropy(node, i)
                    if entropy<bestEntropy:
                        bestEntropy = entropy
                        bestFeature = i
                        req_median = median
            
            childrenData = self.getSplittedData(node, bestFeature)
            return bestFeature, childrenData, req_median


        bestFeature = -1
        bestEntropy = np.inf
        req_median = None
        for i in range(self.nFeatures):
            if i in self.categorical:
                entropy= self.categorical_feature_entropy(node, i)
                if entropy<bestEntropy:
                    bestEntropy = entropy
                    bestFeature = i
            else:
                entropy, median = self.numerical_feature_entropy(node, i)
                if entropy<bestEntropy:
                    bestEntropy = entropy
                    bestFeature = i
                    req_median = median
        childrenData = self.getSplittedData(node, bestFeature)
        return bestFeature, childrenData, req_median



    def split(self, node, childrenData):
        children = []
        for i in range(len(childrenData)):
            children.append(Node(childrenData[i], node))
            children[-1].trainCorrect = self.countCorrect(children[-1], training = True)
        return children

    def getCorrectAtSubtrees(self):
        for leaf in self.listLeaves:
            leaf.correct_in_subtree_validation = leaf.validationCorrect
            leaf.correct_in_subtree_test = leaf.testCorrect
            leaf.correct_in_subtree_train = leaf.trainCorrect

            temp = leaf
            while(temp.par != None):
                temp = temp.par
                temp.correct_in_subtree_validation += leaf.correct_in_subtree_validation
                temp.correct_in_subtree_test += leaf.correct_in_subtree_test
                temp.correct_in_subtree_train += leaf.correct_in_subtree_train


    def prune(self, validationData):
        self.testing(validationData, True)
        self.getCorrectAtSubtrees()
        m_validation = self.root.validationData.shape[0]
        m_test = self.root.testData.shape[0]
        m_train = self.root.data.shape[0]
        accuracy_val = [self.root.correct_in_subtree_validation*100/m_validation]
        accuracy_test = [self.root.correct_in_subtree_test*100/m_test]
        accuracy_train = [self.root.correct_in_subtree_train*100/m_train]
        num_nodes = [self.num_nodes]
        l = [(node.correct_in_subtree_validation - node.validationCorrect, node) for node in self.listNodes]
        li = []
        for i in l:
            heapq.heappush(li, i)
        deleted = 0
        while(heapq):
            improvement, node = heapq.heappop(li)
            if node.deleted_during_pruning or node.madeLeaf_during_pruning or node.isLeaf or improvement!=node.correct_in_subtree_validation-node.validationCorrect:
                continue
            if improvement>=0:
                break
            node.madeLeaf_during_pruning = True
            toPropVal = node.validationCorrect - node.correct_in_subtree_validation
            toPropTest = node.testCorrect - node.correct_in_subtree_test
            toPropTrain = node.trainCorrect - node.correct_in_subtree_train
            node.correct_in_subtree_validation = node.validationCorrect
            node.correct_in_subtree_test = node.testCorrect
            node.correct_in_subtree_train = node.trainCorrect
            temp = node
            while(temp.par!= None):
                temp.par.correct_in_subtree_validation += toPropVal
                temp.par.correct_in_subtree_test += toPropTest
                temp.par.correct_in_subtree_train += toPropTrain
                temp = temp.par
                heapq.heappush(li,(temp.correct_in_subtree_validation-temp.validationCorrect, temp))
            q = deque()
            q.append(node)
            while(q):
                temp = q.popleft()
                if temp.isLeaf or temp.deleted_during_pruning:
                    continue
                for child in temp.children:
                    q.append(child)
                    child.deleted_during_pruning = True
                    deleted += 1
            accuracy_val.append(self.root.correct_in_subtree_validation*100/m_validation)
            accuracy_test.append(self.root.correct_in_subtree_test*100/m_test)
            accuracy_train.append(self.root.correct_in_subtree_train*100/m_train)
            num_nodes.append(self.num_nodes-deleted)

        fig, ax = plt.subplots()
        ax.set_xlabel("No. of nodes")
        ax.set_ylabel("Accuracy")
        ax.set_title("Effect of Pruning on Train Data")
        ax.plot(num_nodes, accuracy_train)
        if self.hot_encode:
            plt.savefig("pruning_train_oh.png")
        else:
            plt.savefig("pruning_train.png")
        plt.show()

        fig, ax = plt.subplots()
        ax.set_xlabel("No. of nodes")
        ax.set_ylabel("Accuracy")
        ax.set_title("Effect of Pruning on Test Data")
        ax.plot(num_nodes, accuracy_test)
        if self.hot_encode:
            plt.savefig("pruning_test_oh.png")
        else:
            plt.savefig("pruning_test.png")
        plt.show()

        fig, ax = plt.subplots()
        ax.set_xlabel("No. of nodes")
        ax.set_ylabel("Accuracy")
        ax.set_title("Effect of Pruning on Validation Data")
        ax.plot(num_nodes, accuracy_val)
        if self.hot_encode:
            plt.savefig("pruning_val_oh.png")
        else:
            plt.savefig("pruning_val.png")
        plt.show()

        return (accuracy_train[-1], accuracy_test[-1], accuracy_val[-1])

    def buildDecisionTree(self):
        q = deque()
        q.append(self.root)
        m = self.root.data.shape[0]
        self.root.trainCorrect = self.countCorrect(self.root, training = True)
        node_ar = [1]
        acc_ar = [self.root.trainCorrect*100/m]
        t_correct = self.root.trainCorrect
        while(len(q)!=0 and self.num_nodes<self.max_nodes):
            node = q.popleft()
            self.listNodes.append(node)
            
            if node.ClFreq[node.label] == node.data.shape[0] or node.data.shape[0]<self.minDataSize:
                node.isLeaf = True
                self.listLeaves.append(node)
                continue
            bestFeature, childrenData, median = self.bestFeatureToSplit(node)
            node.median = median
            node.feature = bestFeature
            self.num_nodes += len(childrenData)
            node.children = self.split(node, childrenData)
            t_correct -= node.trainCorrect
            for child in node.children:
                t_correct += child.trainCorrect
                q.append(child)
            acc_ar.append(t_correct*100/m)
            node_ar.append(node_ar[-1]+len(node.children))
        fig, ax = plt.subplots()
        ax.set_xlabel("No. of nodes")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Nodes for Train Data")
        ax.plot(node_ar, acc_ar)
        if self.hot_encode:
            plt.savefig("trainDataAcc_oh.png")
        else:
            plt.savefig("trainDataAcc.png")
        plt.show()
        

    def countCorrect(self, node, validation = False, training = False):
        correct = 0
        if training:
            for i in range(node.data.shape[0]):
                if node.data[i, -1] == node.label:
                    correct+=1
            return correct
        if validation:
            for i in range(node.validationData.shape[0]):
                if node.validationData[i, -1] == node.label:
                    correct+=1
            return correct

        for i in range(node.testData.shape[0]):
            if node.testData[i, -1] == node.label:
                correct+=1
        return correct
    
    def splitTestData(self, node, validation = False):
        if self.hot_encode:
            if validation:
                feature = node.feature
                childrenValidationData = []
                if feature not in self.numericals:
                    childrenValidationData = [node.validationData[node.validationData[:, feature] == 0], node.validationData[node.validationData[:, feature] == 1]]
                else:
                    childrenValidationData.append(node.validationData[node.validationData[:, feature]<node.median])
                    childrenValidationData.append(node.validationData[node.validationData[:, feature]>=node.median])
                return childrenValidationData


            feature = node.feature
            childrenTestData = []
            if feature not in self.numericals:
                data = node.testData
                childrenTestData = [data[data[:, feature] == 0], data[data[:, feature] == 1]]
            else:
                childrenTestData.append(node.testData[node.testData[:, feature]<node.median])
                childrenTestData.append(node.testData[node.testData[:, feature]>=node.median])
            return childrenTestData

        if validation:
            feature = node.feature
            childrenValidationData = []
            if feature in self.categorical:
                for i in range(len(self.val_to_int[feature])):
                    childrenValidationData.append(node.validationData[node.validationData[:, feature] == i])
            else:
                childrenValidationData.append(node.validationData[node.validationData[:, feature]<node.median])
                childrenValidationData.append(node.validationData[node.validationData[:, feature]>=node.median])
            return childrenValidationData

        feature = node.feature
        childrenTestData = []
        if feature in self.categorical:
            for i in range(len(self.val_to_int[feature])):
                childrenTestData.append(node.testData[node.testData[:, feature] == i])
        else:
            childrenTestData.append(node.testData[node.testData[:, feature]<node.median])
            childrenTestData.append(node.testData[node.testData[:, feature]>=node.median])
        return childrenTestData
        
    def testing(self, testData, validation = False):
        if validation:
            self.root.validationData = self.preprocess(testData)
            m = self.root.validationData.shape[0]
        else:
            self.testData = testData
            self.root.testData = self.preprocess(testData)
            m = self.root.testData.shape[0]

        q = deque()
        q.append(self.root)
        num_nodes = 1
        num_nodes_arr = [1]
        correct = self.countCorrect(self.root, validation)
        # correct_gar = self.countCorrect(self.root, validation, training = True)
        # print(correct == correct_gar)
        if validation:
            self.root.validationCorrect = correct
        else:
            self.root.testCorrect = correct
        # print(self.root.data[0], self.root.testData[0])
        accuracy = [correct*100/m]
        while(len(q)!=0):
            node = q.popleft()
            # if(node.trainCorrect != node.testCorrect):
            #     print(node.par.feature in self.numericals)
            #     sys.exit()
            if node.isLeaf == True:
                continue
            childrenTestData = self.splitTestData(node, validation)
            if validation:
                delta_correct = -1*node.validationCorrect
                for i in range(len(node.children)):
                    node.children[i].validationData = childrenTestData[i]
                    node.children[i].validationCorrect = self.countCorrect(node.children[i], validation)
                    delta_correct += node.children[i].validationCorrect
            else:
                delta_correct = -1*node.testCorrect
                for i in range(len(node.children)):
                    node.children[i].testData = childrenTestData[i]
                    node.children[i].testCorrect = self.countCorrect(node.children[i])
                    delta_correct += node.children[i].testCorrect


            # if (delta_correct<0):
            #     print(node)
            #     sys.exit()
            correct += delta_correct
            num_nodes += len(node.children)
            num_nodes_arr.append(num_nodes)
            accuracy.append(correct*100/m)
            for child in node.children:
                q.append(child)

        
        return (num_nodes_arr, accuracy)


def getLabels(data):
    #age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"
    categorical = [1,2,3,4,6,7,8,10,15]
    values = [set([]) for i in range(16)]
    for i in range(data.shape[0]):
        for j in categorical:
            values[j].add(data[i,j])
    val_to_int = []
    for i in range(16):
        tmap = {}
        count = 0
        for label in values[i]:
            tmap[label] = count
            count +=1
        val_to_int.append(tmap)
    return (categorical, val_to_int)



def hot_encode_data_partC(data, categorical, val_to_int):
    dataX, dataY = data[:, :-1], data[:, -1]
    encodedX, encodedY = [], []
    m,n = dataX.shape
    for i in range(m):
        current_row = []
        for j in range(n):
            if j in categorical:
                for k in val_to_int[j]:
                    if dataX[i, j] == k:
                        current_row.append(True)
                    else:
                        current_row.append(False)
            else:
                current_row.append(dataX[i,j])
        encodedX.append(current_row)
        if dataY[i] == 'yes':
            encodedY.append(True)
        else:
            encodedY.append(False)
    return encodedX, encodedY


def hot_encode_data(data, categorical, val_to_int):
    l = [0]
    numerical = []
    for i in range(1,16):
        if (i-1) in categorical:
            l.append(l[-1]+len(val_to_int[i-1]))
        else:
            numerical.append(l[-1])
            l.append(l[-1]+1)
            
    t_num = l[-1] + len(val_to_int[15])
    
    hot_encoded_dataX = np.zeros((data.shape[0], t_num))
    hot_encoded_dataY = np.zeros((data.shape[0],1))

    dataX, dataY = data[:, :-1], data[:, -1]
    m,n = dataX.shape
    for i in range(m):
        for j in range(n):
            if j in categorical:
                hot_encoded_dataX[i, l[j]+val_to_int[j][dataX[i, j]]] = 1
            else:
                hot_encoded_dataX[i, l[j]] = dataX[i, j]
        if data[i, -1] == 'yes':
            hot_encoded_dataY[i,0] = 1
        else:
            hot_encoded_dataY[i,0] = 0
    return hot_encoded_dataX, hot_encoded_dataY, numerical


def partA(trainData, testData, valData):
    print("Part A")
    categorical, val_to_int = getLabels(trainData)
    model = DecisionTree(categorical, val_to_int, trainData)
    # num_nodes_arr, accuracy = model.testing(trainData)
    # fig, ax = plt.subplots()
    # ax.set_xlabel("No. of nodes")
    # ax.set_ylabel("Accuracy")
    # ax.set_title("Variation of accuracy with no. of nodes for Training Data")
    # ax.plot(num_nodes_arr, accuracy)
    # plt.savefig("trainDataAcc.png")
    # plt.show()
    

    num_nodes_arr, accuracy = model.testing(testData)
    fig, ax = plt.subplots()
    ax.set_xlabel("No. of nodes")
    ax.set_ylabel("Accuracy")
    ax.set_title("Variation of accuracy with no. of nodes for Test Data")
    ax.plot(num_nodes_arr, accuracy)
    plt.savefig("testDataAcc.png")
    plt.show()
    

    num_nodes_arr, accuracy  = model.testing(valData)
    fig, ax = plt.subplots()
    ax.set_xlabel("No. of nodes")
    ax.set_ylabel("Accuracy")
    ax.set_title("Variation of accuracy with no. of nodes for Validation Data")
    ax.plot(num_nodes_arr, accuracy)
    plt.savefig("valDataAcc.png")
    plt.show()


    trainEcodedX, trainEncodedY, numerical = hot_encode_data(trainData, categorical, val_to_int)
    testEcodedX, testEncodedY, numerical = hot_encode_data(testData, categorical, val_to_int)
    valEcodedX, valEncodedY, numerical = hot_encode_data(valData, categorical, val_to_int)
    model_hot = DecisionTree(categorical, val_to_int, np.hstack((trainEcodedX, trainEncodedY)), one_hot=True, numerical = numerical)
    
    # num_nodes_arr, accuracy = model_hot.testing(np.hstack((trainEcodedX, trainEncodedY)))
    # fig, ax = plt.subplots()
    # ax.set_xlabel("No. of nodes")
    # ax.set_ylabel("Accuracy")
    # ax.set_title("Accuracy vs Nodes for Train Data (one-hot encoding)")
    # ax.plot(num_nodes_arr, accuracy)
    # plt.savefig("trainDataAcc_oh.png")
    # plt.show()

    num_nodes_arr, accuracy = model_hot.testing(np.hstack((testEcodedX, testEncodedY)))
    fig, ax = plt.subplots()
    ax.set_xlabel("No. of nodes")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Nodes for Test Data (one-hot encoding)")
    ax.plot(num_nodes_arr, accuracy)
    plt.savefig("testDataAcc_oh.png")
    plt.show()

    num_nodes_arr, accuracy = model_hot.testing(np.hstack((valEcodedX, valEncodedY)))
    fig, ax = plt.subplots()
    ax.set_xlabel("No. of nodes")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Nodes for Validation Data (one-hot encoding)")
    ax.plot(num_nodes_arr, accuracy)
    plt.savefig("valDataAcc_oh.png")
    plt.show()


def partB(trainData, testData, valData):
    print("Part B")
    print("Pruning without one-hot encoding")
    categorical, val_to_int = getLabels(trainData)
    model = DecisionTree(categorical, val_to_int, trainData)
    num_nodes_arr, accuracy = model.testing(testData)
    acc_train, acc_test, acc_val = model.prune(valData)
    print("Train: ", acc_train)
    print("Test: ", acc_test)
    print("Validation: ", acc_val)

    print()
    print("Pruning on one-hot encoded data")
    trainEcodedX, trainEncodedY, numerical = hot_encode_data(trainData, categorical, val_to_int)
    testEcodedX, testEncodedY, numerical = hot_encode_data(testData, categorical, val_to_int)
    valEcodedX, valEncodedY, numerical = hot_encode_data(valData, categorical, val_to_int)
    model_hot = DecisionTree(categorical, val_to_int, np.hstack((trainEcodedX, trainEncodedY)), one_hot=True, numerical = numerical)
    num_nodes_arr, accuracy = model_hot.testing(np.hstack((testEcodedX, testEncodedY)))
    acc_train, acc_test, acc_val = model_hot.prune(np.hstack((valEcodedX, valEncodedY)))
    print("Train: ", acc_train)
    print("Test: ", acc_test)
    print("Validation: ", acc_val)

def partC(trainingData, testData, valData):
    # (a) n estimators (50 to 450 in range of 100). 
    # (b) max features (0.1 to 1.0 in range of 0.2).
    # (c) min samples split (2 to 10 in range of 2).
    print("Part C")
    categorical, val_to_int = getLabels(trainingData)

    trainingData = hot_encode_data_partC(trainingData, categorical, val_to_int)
    testData = hot_encode_data_partC(testData, categorical, val_to_int)
    valData = hot_encode_data_partC(valData, categorical, val_to_int)
    n_estimators = [50, 150, 250, 350, 450] 
    max_features = [0.1, 0.3, 0.5, 0.7, 0.9] 
    min_samples_split = [2, 4, 6, 8, 10] 
    opt_oob_score = -float("inf")
    optParameters = []
    best_model = None

    for ne in n_estimators:
        for mf in max_features:
            for mss in min_samples_split:
                model = RandomForestClassifier(n_estimators=ne,max_features=mf,min_samples_split=mss,criterion = "entropy",bootstrap=True,oob_score=True)
                model.fit(trainingData[0], trainingData[1])
                curr_oob_score = model.oob_score_
                print(ne, mf, mss, ':', curr_oob_score)
                if curr_oob_score > opt_oob_score:
                    opt_oob_score = curr_oob_score
                    optParameters = [ne, mf, mss]
                    best_model = model
    print("Best Parameters found are:")
    print("n_estimator: ", optParameters[0], "max_features: ", optParameters[1], "min_sample_split: ", optParameters[2])
    trainScore = best_model.score(trainingData[0], trainingData[1])
    testScore = best_model.score(testData[0], testData[1])
    valScore = best_model.score(valData[0], valData[1])
    oob_score = best_model.oob_score_
    print("Accuracy on train data:", trainScore*100)
    print("Accuracy on test data:", testScore*100)
    print("Accuracy on validation data:", valScore*100)
    print("Out-of-bag accuracy:", oob_score)
    return optParameters

def partD(trainingData, testData, valData):
    print("Part D")
    # opt_n_estimator, opt_max_feature, opt_min_samples_split = partC(trainData, testData, valData)
    opt_n_estimator, opt_max_feature, opt_min_samples_split = [350, 0.7, 8]
    categorical, val_to_int = getLabels(trainingData)
    n_estimators = [50, 150, 250, 350, 450] 
    max_features = [0.1, 0.3, 0.5, 0.7, 0.9] 
    min_samples_split = [2, 4, 6, 8, 10] 
    trainingData = hot_encode_data_partC(trainingData, categorical, val_to_int)
    testData = hot_encode_data_partC(testData, categorical, val_to_int)
    valData = hot_encode_data_partC(valData, categorical, val_to_int)

    change_ne_score_val = []
    change_ne_score_test = []
    for ne in n_estimators:
        model = RandomForestClassifier(n_estimators=ne,max_features=opt_max_feature,min_samples_split=opt_min_samples_split,criterion = "entropy",bootstrap=True,oob_score=True)
        model.fit(trainingData[0], trainingData[1])
        change_ne_score_val.append(model.score(valData[0], valData[1])*100)
        change_ne_score_test.append(model.score(testData[0], testData[1])*100)

    fig, ax = plt.subplots()
    ax.set_xlabel("n_estimator")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on test data on varying n_estimator")
    ax.plot(n_estimators, change_ne_score_test)
    plt.savefig("sensitivity_ne_test.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("n_estimator")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on validation data on varying n_estimator")
    ax.plot(n_estimators, change_ne_score_val)
    plt.savefig("sensitivity_ne_val.png")
    plt.show()

    change_mf_score_val = []
    change_mf_score_test = []
    for mf in max_features:
        model = RandomForestClassifier(n_estimators=opt_n_estimator,max_features=mf,min_samples_split=opt_min_samples_split,criterion = "entropy",bootstrap=True,oob_score=True)
        model.fit(trainingData[0], trainingData[1])
        change_mf_score_val.append(model.score(valData[0], valData[1])*100)
        change_mf_score_test.append(model.score(testData[0], testData[1])*100)

    fig, ax = plt.subplots()
    ax.set_xlabel("max_features")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on test data on varying max_features")
    ax.plot(max_features, change_mf_score_test)
    plt.savefig("sensitivity_mf_test.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("max_features")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on validation data on varying max_features")
    ax.plot(max_features, change_mf_score_val)
    plt.savefig("sensitivity_mf_val.png")
    plt.show()

    change_mss_score_val = []
    change_mss_score_test = []
    for mss in min_samples_split:
        model = RandomForestClassifier(n_estimators=opt_n_estimator,max_features=opt_max_feature,min_samples_split=mss,criterion = "entropy",bootstrap=True,oob_score=True)
        model.fit(trainingData[0], trainingData[1])
        change_mss_score_val.append(model.score(valData[0], valData[1])*100)
        change_mss_score_test.append(model.score(testData[0], testData[1])*100)


    fig, ax = plt.subplots()
    ax.set_xlabel("min_samples_split")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on test data on varying min_samples_split")
    ax.plot(min_samples_split, change_mss_score_test)
    plt.savefig("sensitivity_mss_test.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("max_features")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on validation data on varying min_samples_split")
    ax.plot(min_samples_split, change_mss_score_val)
    plt.savefig("sensitivity_mss_val.png")
    plt.show()
    

if __name__ == '__main__':
    trainFile, testFile, validationFile, partNo = sys.argv[1:]
    trainData = pd.read_csv(trainFile, delimiter=';').values
    testData = pd.read_csv(testFile, delimiter=';').values
    valData = pd.read_csv(validationFile, delimiter=';').values
    print("Starting...\n")
    if partNo == 'a':
        partA(trainData, testData, valData)
    elif partNo == 'b':
        partB(trainData, testData, valData)
    elif partNo == 'c':
        partC(trainData, testData, valData)
    elif partNo == 'd':
        partD(trainData, testData, valData)