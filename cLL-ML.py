#!/usr/bin/env python
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from pandas import DataFrame, read_table
import pandas as pd
import collections
import random
from collections import Counter
import json
import os
import math
import sys
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sklearn
import argparse
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from scipy import spatial

#This is the variable that defines how many positive instances a token must have before it is deemed useful.
#NOTE: this is different from how many times it appeared for a particular instance. For the training that appears to
#be 1.
MIN_POS_INSTS = 3
#This controls the minimum number of times a token has to appear in descriptions for an instance before the instance
#is deemed to be a positive example of this token
MIN_TOKEN_PER_INST = 5

parser = argparse.ArgumentParser()
parser.add_argument('--resDir',help='path to result directory',required=True)
parser.add_argument('--cat', help='type for learning', choices=['all','rgb','shape','object'],required=True)
parser.add_argument('--pre', help='the file with the preprocessed data', required=True)
parser.add_argument('--cutoff',choices=['0.1','0.15','0.25','0.5','0.75'],help='the cutoff for what portion of negative examples to use', default='0.25')

args = parser.parse_args()

resultDir = args.resDir
preFile = args.pre
kinds = np.array([args.cat])
NEG_SAMPLE_PORTION = float(args.cutoff)
if args.cat == 'all':
	kinds = np.array(['rgb','shape','object'])
execType = 'random'

execPath = './'
dPath = "../"
#dsPath = dPath + "ImgDz/"
dsPath = dPath + "GLD-features/visual_features/"
fAnnotation = execPath + "list_of_instances.conf"

dgAbove = 80

ds = ""
cDf = ""
nDf = ""
tests = ""


"""generalObjs = ['potatoe','cylinder','square', 'cuboid', 'sphere', 'halfcircle','circle','rectangle','cube','triangle','arch','semicircle','halfcylinder','wedge','block','apple','carrot','tomato','lemon','cherry','lime', 'banana','corn','hemisphere','cucumber','cabbage','ear','potato', 'plantain','eggplant']

generalShapes = ['spherical', 'cylinder', 'square', 'rounded', 'cylindershaped', 'cuboid', 'rectangleshape','arcshape', 'sphere', 'archshaped', 'cubeshaped', 'curved' ,'rectangular', 'triangleshaped', 'halfcircle', 'globular','halfcylindrical', 'circle', 'rectangle', 'circular', 'cube', 'triangle', 'cubic', 'triangular', 'cylindrical','arch','semicircle', 'squareshape', 'arched','curve', 'halfcylinder', 'wedge', 'cylindershape', 'round', 'block', 'cuboidshaped']
"""

def fileAppend(fName, sentence):
  """""""""""""""""""""""""""""""""""""""""
	Function to write results/outputs to a log file
		Args: file descriptor, sentence to write
		Returns: Nothing
  """""""""""""""""""""""""""""""""""""""""
  with open(fName, "a") as myfile:
    myfile.write(sentence)
    myfile.write("\n")

######## Negative Example Generation##########
class LabeledLineSentence(object):
    def __init__(self,docLists,docLabels):
        self.docLists = docLists
        self.docLabels = docLabels

    def __iter__(self):
        for index, arDoc in enumerate(self.docLists):
            yield LabeledSentence(arDoc, [self.docLabels[index]])

    def to_array(self):
        self.sentences = []
        for index, arDoc in enumerate(self.docLists):
            self.sentences.append(LabeledSentence(arDoc, [self.docLabels[index]]))
        return self.sentences

    def sentences_perm(self):
        from random import shuffle
        shuffle(self.sentences)
        return self.sentences

class NegSampleSelection:
   """ Class to bundle negative example generation functions and variables. """
   __slots__ = ['docs']
   docs = {}
   def __init__(self,docs):
      """""""""""""""""""""""""""""""""""""""""
                Initialization function for NegSampleSelection class
                Args: Documents dictionary where key is object instance and value
                      is object annotation
                Returns: Nothing
      """""""""""""""""""""""""""""""""""""""""
      docs = collections.OrderedDict(sorted(docs.items()))
      self.docs = docs

   def sentenceToWordLists(self):
      docLists = []
      docs = self.docs
      for key in docs.keys():
         sent = docs[key]
         wLists = sent.split(" ")
         docLists.append(wLists)
      return docLists

   def sentenceToWordDicts(self):
      docs = self.docs
      docDicts = {}
      for key in docs.keys():
         sent = docs[key]
         wLists = sent.split(" ")
         docDicts[key] = wLists
      return docDicts

   def square_rooted(self,x):
      return round(math.sqrt(sum([a*a for a in x])),3)

   def cosine_similarity(self,x,y):
      numerator = sum(a*b for a,b in zip(x,y))
      denominator = self.square_rooted(x)*self.square_rooted(y)
      return round(numerator/float(denominator),3)

   def generateNegatives(self):
      docs = self.docs
      #print '*********************Generate Negs****',docs
      docNames = docs.keys()
      docLists = self.sentenceToWordLists()
      docDicts = self.sentenceToWordDicts()
      docLabels = []
      for key in docNames:
        ar = key.split("/")
        docLabels.append(ar[1])
      sentences = LabeledLineSentence(docLists,docLabels)
      model = Doc2Vec(min_count=1, window=10, size=2000, sample=1e-4, negative=5, workers=8)

      model.build_vocab(sentences.to_array())
      token_count = sum([len(sentence) for sentence in sentences])
      for epoch in range(10):
          model.train(sentences.sentences_perm(),total_examples = token_count,epochs=model.iter)
          model.alpha -= 0.002 # decrease the learning rate
          model.min_alpha = model.alpha # fix the learning rate, no deca
          model.train(sentences.sentences_perm(),total_examples = token_count,epochs=model.iter)

      degreeMap = {}
      for i , item1 in enumerate(docLabels):
         fDoc = model.docvecs[docLabels[i]]
         cInstMap = {}
         cInstance = docNames[i]
         for j,item2 in enumerate(docLabels):
            tDoc = model.docvecs[docLabels[j]]
            cosineVal = max(-1.0,min(self.cosine_similarity(fDoc,tDoc),1.0))

            try:
            	cValue = math.degrees(math.acos(cosineVal))
            except:
                print("ERROR: invalid cosine value")
                print cosineVal
                print fDoc
                print tDoc
                exit()
            tInstance = docNames[j]
            cInstMap[tInstance] = cValue
         degreeMap[cInstance] = cInstMap
      negInstances = {}
      for k in np.sort(degreeMap.keys()):
        v = degreeMap[k]
        ss = sorted(v.items(), key=lambda x: x[1])
        sentAngles = ""
        for item in ss:
          if item[0] != k:
             sentAngles += item[0]+"-"+str(item[1])+","
        sentAngles = sentAngles[:-1]
        negInstances[k] = sentAngles
      return negInstances

############Negative Example Generation --- END ########

class Category:
   """ Class to bundle our dataset functions and variables category wise. """
   __slots__ = ['catNums', 'name']
   catNums = np.array([], dtype='object')

   def __init__(self, name):
      """""""""""""""""""""""""""""""""""""""""
		Initialization function for category class
		Args: category name
		Returns: Nothing
      """""""""""""""""""""""""""""""""""""""""
      self.name = name

   def getName(self):
      """""""""""""""""""""""""""""""""""""""""
	Function to get the category name
		Args: category class instance
		Returns: category name
      """""""""""""""""""""""""""""""""""""""""
      return self.name

   def addCategoryInstances(self,*num):
      """""""""""""""""""""""""""""""""""""""""
      Function to add a new instance number to the category
         Args: category class instance
         Returns: None
      """""""""""""""""""""""""""""""""""""""""
      self.catNums = np.unique(np.append(self.catNums,num))


   def chooseOneInstance(self):
      """""""""""""""""""""""""""""""""""""""""
      Function to select one random instance from this category for testing
         Args: category class instance
         Returns: Randomly selected instance name
      """""""""""""""""""""""""""""""""""""""""
      r = random.randint(0,self.catNums.size - 1)
      instName = self.name + "/" + self.name + "_" + self.catNums[r]
      return instName


class Instance(Category):
	""" Class to bundle instance wise functions and variables """
	__slots__ = ['name','catNum','tokens','negs','gT']
	gT = {}
	tokens = np.array([])
	name = ''
	def __init__(self, name,num):
		"""""""""""""""""""""""""""""""""""""""""
		Initialization function for Instance class
         Args: instance name, category number of this instance
         Returns: Nothing
        """""""""""""""""""""""""""""""""""""""""
		self.name = name
		self.catNum = num

	def getName(self):
		"""""""""""""""""""""""""""""""""""""""""

	Function to get the instance name
		Args: Instance class instance
		Returns: instance name
	"""""""""""""""""""""""""""""""""""""""""
		return self.name



	def getFeatures(self,kind):
		"""""""""""""""""""""""""""""""""""""""""
		Function to find the complete dataset file path (.../arch/arch_1/arch_1_rgb.log)
		where the visual feaures are stored, read the features from the file, and return

		Args: Instance class instance, type of features(rgb, shape, or object)
		Returns: feature set
        """""""""""""""""""""""""""""""""""""""""
		instName = self.name
		instName.strip()
		ar1 = instName.split("/")
		path1 = "/".join([dsPath,instName])
		path  = path1 + "/" + ar1[1] + "_" + kind + ".log"
		#print '******************PATH to featurs ',path
		featureSet = read_table(path,sep=',',  header=None)
		#print len(featureSet.values)
		#print 'FEATURES SET*********************',featureSet.values
		return featureSet.values

	def addNegatives(self, negs):
		"""""""""""""""""""""""""""""""""""""""""
		Function to add negative instances

		Args: Instance class instance, array of negative instances
		Returns: None
        """""""""""""""""""""""""""""""""""""""""
		add = lambda x : np.unique(map(str.strip,x))
		self.negs = add(negs)

	def getNegatives(self):
		"""""""""""""""""""""""""""""""""""""""""
		Function to get the list of negative instances

		Args: Instance class instance
		Returns: array of negative instances
        """""""""""""""""""""""""""""""""""""""""
		return self.negs

	def addTokens(self,tkn):
		"""""""""""""""""""""""""""""""""""""""""
		Function to add a word (token) describing this instance to the array of tokens
         Args: Instance class instance, word
         Returns: None
        """""""""""""""""""""""""""""""""""""""""
		self.tokens = np.append(self.tokens,tkn)

	def getTokens(self):
		"""""""""""""""""""""""""""""""""""""""""
		Function to get array of tokens which humans used to describe this instance
         Args: Instance class instance
         Returns: array of words (tokens)
        """""""""""""""""""""""""""""""""""""""""
		return self.tokens

	def getY(self,token,kind):
		"""""""""""""""""""""""""""""""""""""""""
        Function to find if a token is a meaningful representation for this instance for testing. In other words, if the token is described for this instance in learning phase, we consider it as a meaningful label.
         Args: Instance class instance, word (token) to verify, type of testing
         Returns: 1 (the token is a meaningful label) / 0 (the token is not a  meaningful label)

		NOTE: the result of this function is not actually used anywhere. The gold labels for testing
		are determined during test time. Thus this function is defunct.
		"""""""""""""""""""""""""""""""""""""""""
		"""if token in list(self.tokens):
			if kind == "rgb":
				if token in list(generalColors):
					return 1
			elif kind == "shape":
				if token in list(generalShapes):
					return 1
			else:
				if token in list(generalObjs):
					return 1
		"""
		return 0

class Token:

   """ Class to bundle token (word) related functions and variables """
   __slots__ = ['name', 'posInstances', 'negInstances']
   posInstances = np.array([], dtype='object')
   negInstances = np.array([], dtype='object')

   def __init__(self, name):
        """""""""""""""""""""""""""""""""""""""""
		Initialization function for Token class
         Args: token name ("red")
         Returns: Nothing
        """""""""""""""""""""""""""""""""""""""""
        self.name = name

   def getTokenName(self):
       """""""""""""""""""""""""""""""""""""""""
	Function to get the label from class instance
		Args: Token class instance
		Returns: token (label, for ex: "red")
       """""""""""""""""""""""""""""""""""""""""
       return self.name

   def extendPositives(self,instName):
      """""""""""""""""""""""""""""""""""""""""
		Function to add postive instance (tomato/tomato_1) for this token (red)

		Args: token class instance, positive instance
		Returns: None
      """""""""""""""""""""""""""""""""""""""""
      self.posInstances = np.append(self.posInstances,instName)

   def getPositives(self):
      """""""""""""""""""""""""""""""""""""""""
		Function to get all postive instances of this token

		Args: token class instance
		Returns: array of positive instances (ex: tomato/tomato_1, ..)
      """""""""""""""""""""""""""""""""""""""""
      return self.posInstances

   def extendNegatives(self,*instName):
      """""""""""""""""""""""""""""""""""""""""
		Function to add negative instances for this token

		Args: Instance class instance, array of negative instances
		Returns: None
      """""""""""""""""""""""""""""""""""""""""
      self.negInstances = np.unique(np.append(self.negInstances,instName))

   def getNegatives(self):
      """""""""""""""""""""""""""""""""""""""""
		Function to get all negative instances of this token (ex, "red")

		Args: token class instance
		Returns: array of negative instances (ex: arch/arch_1, ..)
      """""""""""""""""""""""""""""""""""""""""
      return self.negInstances

   def clearNegatives(self):
      self.negInstances = np.array([])


   def getTrainFiles(self,insts,kind):
      """""""""""""""""""""""""""""""""""""""""
        This function is to get all training features for this particular token
		>> Find positive instances described for this token
		>> if the token is used less than 3 times, remove it from execution
		>> fetch the feature values from the physical dataset location
		>> find negative instances and fetch the feature values from the physical location
		>> balance the number positive and negative feature samples

             Args: token class instance, complete Instance list, type for learning/testing
             Returns: training features (X) and values (Y)
      """""""""""""""""""""""""""""""""""""""""
      instances = insts.to_dict()
      pS = Counter(self.posInstances)
      #NOTE: this is not how many times a token was used at all, but how many positive instances it has
      #This means a token count be used 100 times for a particular instance but still not make the cut.
      #print '**************************TOK NAME **',self.name
      #print '**************************POS Instances**',Counter(self.posInstances)
      #print '**************************NEG Instances**',Counter(self.negInstances)
      
      if len(np.unique(self.posInstances)) <= MIN_POS_INSTS:
	  return np.array([]),np.array([])
      #print self.name,":",self.posInstances
      features = np.array([])
      negFeatures = np.array([])
      y = np.array([])
      if self.posInstances.shape[0] == 0 or self.negInstances.shape[0] == 0 :
         return (features,y)

      if self.posInstances.shape[0] > 0 :
        features = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.posInstances)

      if self.negInstances.shape[0] > 0:
        negFeatures = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.negInstances if len(inst) > 1)
        """ if length of positive samples are more than the length of negative samples,
        duplicate negative instances to balance the count"""
        if len(features) > len(negFeatures):
          c = int(len(features) / len(negFeatures))
          negFeatures = np.tile(negFeatures,(c,1))

      if self.posInstances.shape[0] > 0 and self.negInstances.shape[0] > 0 :
       """ if length of positive samples are less than the length of negative samples,
        duplicate positive samples to balance the count"""
       if len(negFeatures) > len(features):
          c = int(len(negFeatures) / len(features))
          features = np.tile(features,(c,1))
      """ find trainY for our binary classifier: 1 for positive samples,
      0 for negative samples"""
      print '**************************LENS POS*****',len(features)
      print '**************************LENS NEGS*****',len(negFeatures)
      y = np.concatenate((np.full(len(features),1),np.full(len(negFeatures),0)))
      if self.negInstances.shape[0] > 0:
        features = np.vstack([features,negFeatures])
      return(features,y)


class DataSet:
   """ Class to bundle data set related functions and variables """
   __slots__ = ['dsPath', 'annotationFile']

   def __init__(self, path,anFile):
      """""""""""""""""""""""""""""""""""""""""
		Initialization function for Dataset class
         Args:
		path - physical location of image dataset
		anFile - 6k amazon mechanical turk description file
         Returns: Nothing
      """""""""""""""""""""""""""""""""""""""""
      self.dsPath = path
      self.annotationFile = anFile

   def findCategoryInstances(self):
      """""""""""""""""""""""""""""""""""""""""
        Function to find all categories and instances in the dataset
		>> Read the amazon mechanical turk annotation file,
		>> Find all categories (ex, tomato), and instances (ex, tomato_1, tomato_2..)
		>> Create Category class instances and Instance class instances

             Args:  dataset instance
             Returns:  Category class instances, Instance class instances
      """""""""""""""""""""""""""""""""""""""""
      nDf = read_table(self.annotationFile,sep=',',  header=None)
      nDs = nDf.values
      #print nDs
      categories = {}
      instances = {}
      for (k1,v1) in nDs:
          instName = k1.strip()
          #print k1
          #print v1
          #print "instname",instName
          (cat,inst) = instName.split("/")
          #(_,num) = inst.split("_")
          (_,num) = inst.rsplit("_",1)
          #print num
          if cat not in categories.keys():
             categories[cat] = Category(cat)
          categories[cat].addCategoryInstances(num)
          if instName not in instances.keys():
             instances[instName] = Instance(instName,num)
      instDf = pd.DataFrame(instances,index=[0])
      catDf =  pd.DataFrame(categories,index=[0])
      return (catDf,instDf)


   def splitTestInstances(self,cDf):
      """""""""""""""""""""""""""""""""""""""""
        Function to find one instance from all categories for testing
		>> We use 4-fold cross validation here
		>> We try to find a random instance from all categories for testing

             Args:  dataset instance, all Category class instances
             Returns:  array of randomly selected instances for testing
      """""""""""""""""""""""""""""""""""""""""
      cats = cDf.to_dict()
      tests = np.array([])
      for cat in np.sort(cats.keys()):
         obj = cats[cat]
         #print tests
         tests = np.append(tests,obj[0].chooseOneInstance())
      tests = np.sort(tests)
      #print '****************************TESTS instances:',tests
      return tests

   def getDataSet(self,cDf,nDf,tests,fName):
      """""""""""""""""""""""""""""""""""""""""
        Function to add amazon mechanical turk description file,
        find all tokens, find positive and negative instances for all tokens

             Args:  dataset instance, array of Category class instances,
		array of Instance class instances, array of instance names to test,
		file name for logging
             Returns:  array of Token class instances
      """""""""""""""""""""""""""""""""""""""""
      instances = nDf.to_dict()
      """ read the amazon mechanical turk description file line by line,
      separating by comma [ line example, 'arch/arch_1, yellow arch' """
      df = read_table(self.annotationFile, sep=',',  header=None)
      tokenDf = {}
      cDz = df.values
      """ column[0] would be arch/arch_1 and column[1] would be 'yellow arch' """
      docs = {}
      for column in df.values:
        ds = column[0]
        if ds in docs.keys():
           sent = docs[ds]
           sent += " " + column[1]
           docs[ds] = sent
        else:
           docs[ds] = column[1]

      #print '*****************DOCS',docs
      for inst in docs.keys():
          #get the counts for tokens and filter those < MIN_TOKEN_PER_INST
          token_counts = pd.Series(docs[inst].split(" ")).value_counts()
          #print '***************************BEFORE FILTER Token counts',token_counts

          token_counts = token_counts[token_counts >= MIN_TOKEN_PER_INST]
          dsTokens = token_counts.index.tolist()
          #print '***************************Token counts',token_counts
            
          instances[inst][0].addTokens(dsTokens)
          #print 'tests***************************',tests
          #print "Curr INST",inst
            ## change to consider only descriptions from training set
          if inst not in tests:
            #print "DS",ds
            iName = instances[ds][0].getName()
            for annotation in dsTokens:
                if annotation not in tokenDf.keys():
                 # creating Token class instances for all tokens (ex, 'yellow' and 'arch')
                    tokenDf[annotation] = Token(annotation)
                # add 'arch/arch_1' as a positive instance for token 'yellow'
                #print "Extending positives  --anno",annotation
                #print "Extending positives  --inst",inst
                tokenDf[annotation].extendPositives(inst)
      """
        if ds not in tests:
         iName = instances[ds][0].getName()
         for annotation in dsTokens:
             if annotation not in tokenDf.keys():
                 # creating Token class instances for all tokens (ex, 'yellow' and 'arch')
                 tokenDf[annotation] = Token(annotation)
             # add 'arch/arch_1' as a positive instance for token 'yellow'
             tokenDf[annotation].extendPositives(ds) """
      tks = pd.DataFrame(tokenDf,index=[0])
      sent = "Tokens :: "+ " ".join(tokenDf.keys())
      fileAppend(fName,sent)
      negSelection = NegSampleSelection(docs)
      negExamples = negSelection.generateNegatives()
      #print 'Returned Negs ', negExamples

      """ find negative instances for all tokens.
      """
      for tk in tokenDf.keys():
         poss = list(set(tokenDf[tk].getPositives()))
         negs = []
         #this keeps track of how strongly negative each instance is for this token
         negCandidateScores = {}

         for ds in poss:
             if isinstance(negExamples[ds], str):
                  negatives1 = negExamples[ds].split(",")
                  localNegCandScores = {}
                  for instNeg in negatives1:
                       s1 = instNeg.split("-")
                       #filter out instances that see the token in their descriptions
                       #also filter out instances that are in the test split
                       if s1[0] in tests or tk in docs[s1[0]].split(" "):
                            continue

                       localNegCandScores[s1[0]] = float(s1[1])
                  #sort the local dictionary by closeness and select the back 2/3 of that list
                  scores_sorted = list(sorted(localNegCandScores.iteritems(), key= lambda x: x[1], reverse = False))
                  scores_sorted = scores_sorted[len(scores_sorted)//3:]
                  #now update the main dictionary
		  for (inst,val) in scores_sorted:
                       if inst in negCandidateScores:
			    negCandidateScores[inst] += val
                       else:
                            negCandidateScores[inst] = val


         #out of the options left choose the N most negative
         num_to_choose = int(math.ceil(float(len(negCandidateScores.keys()))*NEG_SAMPLE_PORTION))
         #TESTING: no more than twice as many negative examples as positive
         #num_to_choose = min(len(poss)*2,num_to_choose)
         choices_sorted = list(sorted(negCandidateScores.iteritems(), key= lambda x: x[1], reverse = True))
         choices = [negInst for negInst,negVal in choices_sorted[:num_to_choose]]


         print "For token",tk,"with",len(poss),"positive examples","choosing",num_to_choose,"examples","out of",len(negCandidateScores.keys())

         #print "Original negatives:",negs
         #print "New negatives:", choices
         negsPart = choices
         tokenDf[tk].extendNegatives(negsPart)
      #exit()

      return tks


def getTestFiles(insts,kind,tests,token):
   """""""""""""""""""""""""""""""""""""""""
   Function to get all feature sets for testing and dummy 'Y' values
		Args:  Array of all Instance class instances, type of testing
				(rgb, shape, or object) , array of test instance names,
				token (word) that is testing
        Returns:  Feature set and values for testing
   """""""""""""""""""""""""""""""""""""""""
   instances = insts.to_dict()
   features = []
   y = []
   for nInst in tests:
      y1 = instances[nInst][0].getY(token,kind)
      fs  = instances[nInst][0].getFeatures(kind)
      features.append(list(fs))
      y.append(list(np.full(len(fs),y1)))
   return(features,y)

def getNonTestFiles(insts,kind,tests,token):
   """""""""""""""""""""""""""""""""""""""""
   Function to get all feature sets for training data. This is used for
   testing on the training data (as a preliminary step to filter out tokens
   that are not meaningful like 'the')
                Args:  Array of all Instance class instances, type of testing
                                (rgb, shape, or object) , array of test instance names,
                                token (word) that is testing
        Returns:  Feature set
   """""""""""""""""""""""""""""""""""""""""
   instances = insts.to_dict()
   features = []
   trainNames = []
   for nInst in instances.keys():
      if nInst not in tests:
	fs  = instances[nInst][0].getFeatures(kind)
        trainNames.append(nInst)
	features.append(list(fs))

   return(features, trainNames)

def findTrainTestFeatures(insts,tkns,tests):
  """""""""""""""""""""""""""""""""""""""""
  Function to iterate over all tokens, find train and test features for execution
	Args:  Array of all Instance class instances,
		array of all Token class instances,
		array of test instance names
    Returns:  all train test features, values, type of testing
  """""""""""""""""""""""""""""""""""""""""
  tokenDict = tkns.to_dict()
  for token in np.sort(tokenDict.keys()):
  ##CHANGE
  #for token in ['marker','syring','pencil']: ##
     objTkn = tokenDict[token][0]
     for kind in kinds:
#     for kind in ['rgb']:
        (features,y) = objTkn.getTrainFiles(insts,kind)
        (testFeatures,testY) = getTestFiles(insts,kind,tests,token)
        (trainForTestingFeatures, trainNames) = getNonTestFiles(insts,kind,tests,token)
        if len(features) == 0 :
            continue;
        yield (token,kind,features,y,testFeatures,testY,trainForTestingFeatures,trainNames)


def callML(resultDir,insts,tkns,tests,algType,resfname):
  """ generate a CSV result file with all probabilities
	for the association between tokens (words) and test instances"""
  print resultDir
  print 'create csv'
  confFile = open(resultDir + '/groundTruthPrediction.csv','w')
  headFlag = 0
  fldNames = np.array(['Token','Type'])
  confWriter = csv.DictWriter(confFile, fieldnames=fldNames)

  """ Generate another CSV file with the results of applying the classifiers back on
      the training data. This is used for token filtering """
  trainConfFile = open(resultDir + '/groundTruthPredictionTrain.csv','w')
  trainHeadFlag = 0
  trainFldNames = set()

  """ Trying to add correct object name of test instances in groundTruthPrediction csv file
	ex, 'tomato/tomato_1 - red tomato' """
  featureSet = read_table(fAnnotation,sep=',',  header=None)
  featureSet = featureSet.values
  fSet = dict(zip(featureSet[:,0],featureSet[:,1]))
  testTokens = []

  """ fine tokens, type to test, train/test features and values """
  for (token,kind,X,Y,tX,tY,trX, trNames) in findTrainTestFeatures(insts,tkns,tests):
   #continue
   if token not in testTokens:
      testTokens.append(token)
   #print testTokens
   #print len(testTokens)
   print "Token : " + token + ", Kind : " + kind
   """ binary classifier Logisitc regression is used here """
   polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
   sgdK = linear_model.LogisticRegression(C=10**5,random_state=0)
   #   pipeline2_2 = Pipeline([("polynomial_features", polynomial_features),
   #                         ("logistic", sgdK)])
   pipeline2_2 = Pipeline([("logistic", sgdK)])

   pipeline2_2.fit(X,Y)
   fldNames = np.array(['Token','Type'])
   confD = {}
   confDict = {'Token' : token,'Type' : kind}
   """ testing all images category wise and saving the probabilitties in a Map
		for ex, for category, tomato, test all images (tomato image 1, tomato image 2...)"""
   for ii in range(len(tX)) :
      testX = tX[ii]

      testY = tY[ii]
      tt = tests[ii]

      predY = []
      tProbs = []
      probK = pipeline2_2.predict_proba(testX)
      tProbs = probK[:,1]
      predY = tProbs

      for ik in range(len(tProbs)):
          fldNames = np.append(fldNames,str(ik) + "-" + tt)
          confD[str(ik) + "-" + tt] = str(fSet[tt])

      for ik in range(len(tProbs)):
          confDict[str(ik) + "-" + tt] = str(tProbs[ik])

   if headFlag == 0:
      headFlag = 1
      """ saving the header of CSV file """
      confWriter = csv.DictWriter(confFile, fieldnames=fldNames)
      confWriter.writeheader()

      confWriter.writerow(confD)
   """ saving probabilities in CSV file """
   confWriter.writerow(confDict)

   #now generate the probability of each training data being an example of this token and kind
   trainProbDict = {"Token":token,"Type":kind}
   trainConfD = {}

   for trI in range(len(trX)):
      trainIX = trX[trI]
      trName = trNames[trI]
      #print trName, trainIX

      probK = pipeline2_2.predict_proba(trainIX)
      tProbs = probK[:,1]

      for ik in range(len(tProbs)):
          trainFldNames.add(str(ik) + "-" + trName)
          trainConfD[str(ik) + "-" + trName] = fSet[trName]

      for ik in range(len(tProbs)):
          trainProbDict[str(ik) + "-" + trName] = str(tProbs[ik])

   #should be unable to even predict the training data
   if trainHeadFlag == 0:
      trainHeadFlag = 1
      print "writing the header len:",2+len(trainFldNames)
      """ saving the header of CSV file """
      trainConfWriter = csv.DictWriter(trainConfFile, fieldnames=['Token','Type']+list(trainFldNames))
      trainConfWriter.writeheader()

      trainConfWriter.writerow(trainConfD)
   """ saving probabilities in CSV file """
   trainConfWriter.writerow(trainProbDict)


  confFile.close()
  trainConfFile.close()

def execution(resultDir,ds,cDf,nDf,tests):

    os.mkdir(resultDir)
    os.mkdir(resultDir+"/NoOfDataPoints")
    resultDir1 = resultDir + "/NoOfDataPoints/6000"
    os.mkdir(resultDir1)

    fResName = resultDir1 + "/results.txt"
    sent = "Test Instances :: " + " ".join(tests)
    fileAppend(fResName,sent)
    """ read amazon mechanical turk file, find all tokens
    get positive and negative instance for all tokens """
    tokens = ds.getDataSet(cDf,nDf,tests,fResName)
    """ Train and run binary classifiers for all tokens, find the probabilities
	for the associations between all tokens and test instances,
	and log the probabilitties """
    callML(resultDir1,nDf,tokens,tests,0,fResName)


if __name__== "__main__":
  print "START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  anFile =  execPath + preFile
  fResName = ""
  #os.system("mkdir -p " + resultDir)
  """ creating a Dataset class Instance with dataset path, amazon mechanical turk description file"""
  ds = DataSet(dsPath,anFile)
  """ find all categories and instances in the dataset """
  (cDf,nDf) = ds.findCategoryInstances()
  """ find all test instances. We are doing 4- fold cross validation """
  #print cDf
  tests = ds.splitTestInstances(cDf)
  print "ML START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  execution(resultDir,ds,cDf,nDf,tests)
  print "ML END :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
