from numpy import linalg
from numpy.core.numeric import Inf
import numpy as np
import sys
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
vocab = {}
biVocab = {}
triVocab = {}
biPhi = {}
triPhi = {}
biTheta = {}
triTheta = {}


# function to extract features from the samples
def getFeatures(data, stem, useSummary, remStopWords, type):
    finalData = {"label": data["label"]}
    featureVec_review = word_tokenize(data["reviewText"])

    if useSummary:
        featureVec_summary = word_tokenize(data["summary"])

    if remStopWords:
        featureVec_review = [w for w in featureVec_review if w.lower() not in stop_words]
        if useSummary:
            featureVec_summary = [w for w in featureVec_summary if w.lower() not in stop_words]

    if stem:
        featureVec_review = [stemmer.stem(w) for w in featureVec_review]
        if useSummary:
            featureVec_summary = [stemmer.stem(w) for w in featureVec_summary]

    if type == "bigram":
        featureVec_review = [featureVec_review[i]+" "+featureVec_review[i+1] for i in range(len(featureVec_review)-1)]
        if useSummary:
            featureVec_summary = [featureVec_summary[i]+" "+featureVec_summary[i+1] for i in range(len(featureVec_summary)-1)]

    if type == 'weighted':
        featureVec_bigram = [featureVec_review[i]+" "+featureVec_review[i+1] for i in range(len(featureVec_review)-1)]
        featureVec_trigram = [featureVec_review[i]+" "+featureVec_review[i+1]+" "+featureVec_review[i+2] for i in range(len(featureVec_review)-2)]
        finalData["bigrams"] = featureVec_bigram
        finalData["trigrams"] = featureVec_trigram
    
    finalData["featureVec_review"] = featureVec_review
    if useSummary:
        finalData["featureVec_summary"] = featureVec_summary
    return finalData

# function to parse a file
def parse(path):
    file1 = open(path, "r")
    for l in file1:
        yield eval(l)


# function to read data from a file and preprocess the data
def getData(trainFile, useSummary, stem, remStopWords, type, getFeat = True):
    dataset = []
    for data in parse(trainFile):
        d = {"reviewText": data["reviewText"], "label": int(data["overall"])}
        if useSummary:
            d["summary"] = data["summary"]
        if getFeat:
            d = getFeatures(d, stem, useSummary, remStopWords, type)
        #if (len(dataset)%1000==0):
        #    print(len(dataset))
        dataset.append(d)
    return dataset

# creates a vocabulary list from the given data
def getVocab(dataset, useSummary=False, type = "unigram"):
    wordSet = set({})
    for i in range(len(dataset)):
        for w in dataset[i]["featureVec_review"]:
            wordSet.add(w)
        if useSummary:
            for w in dataset[i]["featureVec_summary"]:
                wordSet.add(w)
    i = 1
    global vocab
    for w in wordSet:
        vocab[w] = i
        i+=1
    
    if type == "weighted":
        wordSet2 = set({})
        wordSet3 = set({})
        for i in range(len(dataset)):
            for w in dataset[i]["bigrams"]:
                wordSet2.add(w)
            for w in dataset[i]["trigrams"]:
                wordSet3.add(w)
        global biVocab, triVocab
        i = 1
        for w in wordSet2:
            biVocab[w] = i
            i+=1
        i = 1
        for w in wordSet3:
            triVocab[w] = i
            i+=1
        
    print("Vocab Created..")
    return
        
# function to train the model to get the parameters theta and phi
def train(trainData, useSummary, type):
    phi = np.zeros((6,))
    phi[0] = 1
    V = len(vocab)
    theta = np.ones((6, V+1))
    wordCount = np.array([V for i in range(6)]).reshape((6,1))
    for i in range(len(trainData)):
        y = trainData[i]['label']
        phi[y]+=1
        wordCount[y,0] += len(trainData[i]['featureVec_review'])
        for j in trainData[i]['featureVec_review']:
            theta[y, vocab[j]]+=1
        if useSummary:
            wordCount[y,0] += len(trainData[i]['featureVec_summary'])
            for j in trainData[i]['featureVec_summary']:
                theta[y, vocab[j]]+=1
    theta/=wordCount
    phi/=len(trainData)

    if (type == "weighted"):
        global biPhi, biTheta, triPhi, triTheta
        biPhi = np.zeros((6,))
        biPhi[0] = 1
        V = len(biVocab)
        biTheta = np.ones((6, V+1))
        wordCount = np.array([V for i in range(6)]).reshape((6,1))
        for i in range(len(trainData)):
            y = trainData[i]['label']
            biPhi[y]+=1
            wordCount[y,0] += len(trainData[i]['bigrams'])
            for j in trainData[i]['bigrams']:
                biTheta[y, biVocab[j]]+=1
        biTheta/=wordCount
        biPhi/=len(trainData)
        biTheta, biPhi = np.log(biTheta), np.log(biPhi)

        triPhi = np.zeros((6,))
        triPhi[0] = 1
        V = len(triVocab)
        triTheta = np.ones((6, V+1))
        wordCount = np.array([V for i in range(6)]).reshape((6,1))
        for i in range(len(trainData)):
            y = trainData[i]['label']
            triPhi[y]+=1
            wordCount[y,0] += len(trainData[i]['trigrams'])
            for j in trainData[i]['trigrams']:
                triTheta[y, triVocab[j]]+=1
        triTheta/=wordCount
        triPhi/=len(trainData)
        triTheta, triPhi = np.log(triTheta), np.log(triPhi)

    print("Parameters found...")
    return (phi, theta)

# function to predict label of a given sample
def predict(data, params, useSummary, type, weight = 1):
    predClass = 0
    prob = -Inf
    logPhi, logTheta = params
    for i in range(1, 6):
        tprob = logPhi[i]
        for w in data['featureVec_review']:
            if w not in vocab:
                tprob += np.log((1/len(vocab)))
            else:
                tprob += logTheta[i, vocab[w]]
        if type == "weighted":
            biProb = 0
            for w in data['bigrams']:
                if w in biVocab:
                    biProb += biTheta[i, biVocab[w]]
            triProb = 0
            for w in data['trigrams']:
                if w in triVocab:
                    triProb += triTheta[i, triVocab[w]]
            tprob+= biProb/10+triProb
            

        if useSummary:
            prob_summ = logPhi[i]
            for w in data['featureVec_summary']:
                if w not in vocab:
                    prob_summ += np.log((1/len(vocab)))
                else:
                    prob_summ += logTheta[i, vocab[w]]
            tprob += (prob_summ)/weight
        
        if tprob>=prob:
            prob = tprob
            predClass = i
    return predClass

# function to test the model on a given dataset
def test(testData, useSummary, params, type):
    prediction = []
    givenClass = []
    phi, theta = params
    log_params = np.log(phi), np.log(theta)
    for i in range(len(testData)):
        y = testData[i]['label']
        givenClass.append(y)
        prediction.append(predict(testData[i], log_params, useSummary, type))
    accuracy = 0
    for i in range(len(prediction)):
        if prediction[i]==givenClass[i]:
            accuracy+=1
    accuracy/=len(prediction)
    return(accuracy*100, prediction, givenClass)

# function to get confusion matrix from
def getConfusionMatrix(predicted, actual):
    confusion = np.zeros((6,6), dtype=np.int32)  #[actual, predicted]
    for i in range(len(predicted)):
        confusion[actual[i], predicted[i]]+=1
    return confusion

# function to get random testing result
def randomTesting(dataset):
    m = len(dataset)
    prediction = []
    givenClass = []
    for i in range(m):
        y = dataset[i]['label']
        givenClass.append(y)
        prediction.append(random.randint(1,5))
    accuracy = 0
    for i in range(len(dataset)):
        if prediction[i]==givenClass[i]:
            accuracy+=1
    accuracy/=len(dataset)
    print()
    print ("Accuracy on Random Prediction: ", accuracy*100)
    return prediction, givenClass

# function to get majority testing result
def majorityTesting(dataset, testData):
    count = [0 for i in range(6)]
    for i in range(len(dataset)):
        count[dataset[i]['label']]+=1
    major_count = 0
    major = 0
    for i in range(1,6):
        if count[i]>major_count:
            major_count = count[i]
            major = i
    givenClass = []
    prediction = [major for i in range(len(testData))]
    for i in range(len(testData)):
        y = testData[i]['label']
        givenClass.append(y)
    accuracy = 0
    for i in range(len(testData)):
        if prediction[i] == givenClass[i]:
            accuracy+=1
    accuracy/=len(testData)
    print()
    print ("Accuracy on Majority Prediction: ", accuracy*100)
    return prediction, givenClass

# function to get F1 score
def getF1_score(confMat):
    totalActuals = np.sum(confMat, axis = 1, keepdims=True)
    totalPredicted = np.sum(confMat, axis = 0, keepdims=True)
    for i in range(6):
        if totalActuals[i,0] == 0:
            totalActuals[i,0] = 1
        if totalPredicted[0,i] == 0:
            totalPredicted[0,i] = 1
    precision = confMat/totalPredicted
    recall = confMat/totalActuals
    micro_f1 = np.zeros((1,5))
    for i in range(1,6):
        if precision[i,i]+recall[i,i] !=0:
            micro_f1[0,i-1] = 2*precision[i,i]*recall[i,i]/(precision[i,i]+recall[i,i])
    print("Micro F1-scores: ", micro_f1)
    macro_f1 = np.sum(micro_f1)/5
    print("Macro F1-score: ", macro_f1)


def main_function(trainFile, testFile, type, useSummary, stem, remStopWords, confusionMat, randomPredict, majorityPredict, getF1):
    start = time.time()
    trainData = getData(trainFile, useSummary, stem, remStopWords, type)
    getVocab(trainData, useSummary, type)
    end = time.time()
    print("Time taken to pre-process the data (in sec): ", end-start)
    start = time.time()
    model = train(trainData, useSummary, type)
    end = time.time()
    print("Time taken to train the model (in sec): ", end-start)
    # print()
    # print("Testing on Training Data...")
    # accuracy, prediction, actual = test(trainData, useSummary, model, type)
    # print("Accuracy on Training Data:", accuracy)
    # print("F1-score on Training Data:")
    # conf = getConfusionMatrix(prediction, actual)
    # getF1_score(conf)

    print()
    testData = getData(testFile, useSummary, stem, remStopWords, type)
    print("Testing on Test-Data...")
    accuracy, prediction, actual = test(testData, useSummary, model, type)
    print("Accuracy on Test-Data:", accuracy)
    print("F1-score on Test Data:")
    conf = getConfusionMatrix(prediction, actual)
    if getF1:
        getF1_score(conf)

    if confusionMat:
        print()
        print("Confusion Matrix Obtained:")
        print(conf[1:, 1:])
    
# main function to handle different operations in different problems
def main(trainFile, testFile, type, useSummary, stem, remStopWords, confusionMat, randomPredict, majorityPredict, getF1):
    if randomPredict or majorityPredict:
        testData = getData(testFile, useSummary, stem, remStopWords, type, False)
        print()
        print("Random Prediction: ")
        prediction, actual = randomTesting(testData)
        conf = getConfusionMatrix(prediction, actual)
        getF1_score(conf)

        print()
        print("Major Prediction: ")
        trainData = getData(trainFile, useSummary, stem, remStopWords, type, False)
        prediction, actual = majorityTesting(trainData, testData)
        conf = getConfusionMatrix(prediction, actual)
        getF1_score(conf)
        return
    if type == "weighted_&_bigram":
        print("\nUsing Bigram Features: ")
        main_function(trainFile, testFile, "bigram", useSummary, stem, remStopWords, confusionMat, randomPredict, majorityPredict, getF1)
        global vocab
        vocab = {}
        print("\n\nUsing Weighted Unigram + Bigram + Trigram Features: ")
        main_function(trainFile, testFile, "weighted", useSummary, stem, remStopWords, confusionMat, randomPredict, majorityPredict, getF1)
    elif type =="unigram":
        main_function(trainFile, testFile, type, useSummary, stem, remStopWords, confusionMat, randomPredict, majorityPredict, getF1)
    elif type =="bigram":
        print("\nUsing Bigram Features: ")
        main_function(trainFile, testFile, "bigram", useSummary, stem, remStopWords, confusionMat, randomPredict, majorityPredict, getF1)
    elif type =="weighted":
        print("\nUsing Weighted Unigram + Bigram + Trigram Features: ")
        main_function(trainFile, testFile, "weighted", useSummary, stem, remStopWords, confusionMat, randomPredict, majorityPredict, getF1)


# entry point for the file
if __name__ == '__main__':
    arg = sys.argv[1:]
    trainFile, testFile, part = arg[0], arg[1], arg[2]
    # change value based on question numbers
    type = "unigram"   #simple, bigram,
    useSummary = False
    stem = False
    remStopWords = False
    confusionMat = False
    randomPredict = False
    majorityPredict = False
    getF1 = True
    if part == 'b':
        randomPredict = True
        majorityPredict = True
    elif part == 'c':
        confusionMat = True
    elif part == 'd' or part == 'f':
        stem = True
        remStopWords = True
    elif part == 'e':
        stem = True
        remStopWords = True
        type = "weighted_&_bigram"
    elif part == 'g':
        stem = True
        remStopWords = True
        useSummary = True
    main(trainFile, testFile, type, useSummary, stem, remStopWords, confusionMat, randomPredict, majorityPredict, getF1)