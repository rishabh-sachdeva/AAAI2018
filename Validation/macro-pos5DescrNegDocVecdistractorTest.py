# This script tests and validates the performance of visual classifiers beyond category,
# by selecting right positive instances from the pool of positive and negative instance object images.
#
# Positive instances --> images of instances of which the visual classifier token is used at least 6 times to describe.
# Negative instances -- > the intersection of negative sampling and Doc2Vec negative instances of all positive instances

# Arguments: If your result directory is 'test/NoOfDataPoints/6000', then
# python macro-pos5DescrNegDocVecdistractorTest.py test/NoOfDataPoints/ 'rgb' 'rgb'
# or
# python macro-pos5DescrNegDocVecdistractorTest.py test/NoOfDataPoints/ 'shape' 'shape'
# or
# python macro-pos5DescrNegDocVecdistractorTest.py test/NoOfDataPoints/ 'object' 'object'
#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
import random
import util
from collections import Counter
import json
import sys
import csv
import os
import collections
import os.path

posInsts = {}
negInsts = {}
objInstances = {}
objNames = {}
classifierProbs = {}

argvs = sys.argv
if len(argvs) == 1:
	exit(0)

fld = str(argvs[1])

argProb = 0.50
tID = ""

if len(argvs) > 2:
	tID =  str(argvs[2])

cID = ""
if len(argvs) > 3:
   cID = str(argvs[3])

#tID = "rgb"
#tID = "shape"
#tID = "object"
posName = "posInstances.conf"
negName = "negInstances.conf"

pos = fld + "/evalHelpFiles/" + tID + "posInstances.conf"
neg = fld + "/evalHelpFiles/" + tID + "negInstances.conf"

#meaningfulWords = ["wedge", "cylinder", "square", "yellow","carrot", "tomato", "curved", "archshaped","lime", "blue", "eggplant", "purple","cuboid", "prism", "orange", "plantain", "white", "semicylinder", "banana", "red", "cube", "triangle", "semicircle", "cylindrical", "corn", "triangular", "cucumber", "brinjal", "lemon", "cabbage", "arch", "circle",  "plum", "potato", "rectangular", "green", "eggplant",  "rectangle"]

#rgbWords  = ["yellow","blue", "purple","orange", "white", "red", "green"]

#shapeWords  = ["wedge", "cylinder", "square",  "curved", "archshaped","cuboid",  "semicylinder",  "cube", "triangle", "semicircle", "cylindrical", "triangular","arch", "circle",  "rectangular",  "rectangle"]

#objWords = ["cylinder", "carrot", "tomato", "lime", "cuboid", "prism", "orange", "plantain", "semicylinder", "banana",  "cube", "triangle", "corn", "cucumber", "brinjal", "lemon", "cabbage", "arch",  "plum", "eggplant"]

#shapeWords = list(set(shapeWords) - set(["archshaped", "cylindrical", "curved", "semicylinder"]))
#shapeWords = list(set(shapeWords) - set(['semicircle']))
#shapeWords = list(set(shapeWords) - set(['semicircle', 'curved', 'archshaped','wedge','semicylinder']))
#shapeWords = list(set(shapeWords) - set(['semicircle', 'archshaped', 'cuboid', 'cylindrical', 'curved', 'semicylinder']))
#objWords = list(set(objWords) - set(['plantain', 'prism', 'semicylinder', 'brinjal']))
#objWords = list(set(objWords) - set(['plantain', 'prism', 'brinjal']))

####TFIDF MeaningfulWords - Above 50.0####
meaningfulWords = ['cylinder', 'apple', 'yellow', 'carrot', 'lime', 'blue', 'lemon', 'purple', 'orange', 'banana', 'red', 'cube', 'triangle', 'corn', 'triangular', 'cucumber', 'half', 'cabbage', 'ear', 'tomato', 'potato', 'rectangular', 'cob', 'green', 'eggplant']
rgbWords  = ['yellow','blue','purple', 'orange','red','green']
shapeWords  = ['cylinder','cube', 'triangle','triangular','rectangular']
objWords = ['cylinder', 'apple','carrot', 'lime','lemon','orange', 'banana','cube', 'triangle', 'corn','cucumber', 'half', 'cabbage', 'ear', 'tomato', 'potato', 'cob','eggplant']

objWords = list(set(objWords) - set(['ear','half']))
tobeTestedTokens = rgbWords
if tID == "":
 tobeTestedTokens.extend(shapeWords)
 tobeTestedTokens.extend(objWords)
elif tID == "shape":
   tobeTestedTokens = shapeWords
elif tID == "object":
   tobeTestedTokens = objWords

neverAppeared = tobeTestedTokens
predfName = fld + "/groundTruthPrediction.csv"
predfileName = fld
def getPosNegInstances():
	resDictFile = []
	with open(pos) as csvfile:
		readFile = csv.DictReader(csvfile)
		for row in readFile:
			resDictFile.append(row)
			color = row['token']
			objs = row['objects']
			insts = objs.split("-")
			newInsts = []
			for instance in insts:
				if instance in objInstances.keys():
					newInsts.append(instance)
					posInsts[color] = newInsts

	resDictFile = []
	with open(neg) as csvfile:
  	  readFile = csv.DictReader(csvfile)
  	  for row in readFile:
  	  	  resDictFile.append(row)
  	  	  color = row['token']
  	  	  objs = row['objects']
  	  	  insts = objs.split("-")
  	  	  newInsts = []
  	  	  for instance in insts:
  	  	  	  if instance in objInstances.keys():
  	  	  	  	  newInsts.append(instance)
  	  	  	  	  negInsts[color] = newInsts


def getTestInstancesAndClassifiers(fName):
#print fName
	global objInstances,objNames,classifierProbs
	head = 0
	objInstances = {}
	objNames = {}
	classifierProbs = {}
	with open(fName) as csvfile:
		readFile = csv.DictReader(csvfile)
		for row in readFile:
			if head == 0:
				temp = row.keys()

				temp.remove('Type')
				temp.remove('Token')
				for inst in temp:
					ar = inst.split("-")
					if ar[1] in objInstances.keys():
						objInstances[ar[1]].append(inst)
					else :
						objInstances[ar[1]] = [inst]
						objNames[ar[1]] = row[inst]
			else :
                                if cID == "":
					#if row['Token'] in posInsts.keys():
					if row['Token'] in classifierProbs.keys():
						classifierProbs[row['Token']].append(row)
					else:
						classifierProbs[row['Token']] = [row]
				else:
					if row['Type'] == cID:
						if row['Token'] in classifierProbs.keys():
							classifierProbs[row['Token']].append(row)
						else:
							classifierProbs[row['Token']] = [row]

			head = head + 1

def writePosNeg():
 testObjs = objInstances.keys()
 descObjs = util.getDocsForTest(testObjs)
# descObjs = util.getDocuments()
 objTokens = util.sentenceToWordDicts(descObjs)
 tknsGlobal = set()
 posTokens = {}
 negSampleTokens = {}
 mostImpTokens = {}
 if len(objTokens.keys()) == 0:
   return 1
 for (key,value) in objTokens.items():
    cValue = Counter(value)
    mostImpTokens[key] = []
    for (k1,v1) in cValue.items():
     if v1 > 10:
        mostImpTokens[key].append(k1)
     if v1 > 10:
       if k1 in meaningfulWords:
          tknsGlobal.add(k1)
          if k1 in posTokens.keys():
             kk1 = posTokens[k1]
             kk1.append(key)
             posTokens[k1] = kk1
          else:
             posTokens[k1] = [key]
 posTokens = collections.OrderedDict(sorted(posTokens.items()))
 os.system("mkdir -p " + fld + "/evalHelpFiles/")
 f = open(fld + "/evalHelpFiles/" + posName, "w")
 title = "token,objects\n"
 f.write(title)
 for k,v in posTokens.items():
    ll  = str(k) + ","
    ll += "-".join(v)
    ll += "\n"
    f.write(ll)
 f.close()
 for kTkn in posTokens.keys():
    negSampleTokens[kTkn] = []
    for (key,value) in objTokens.items():
       if kTkn not in value:
          negSampleTokens[kTkn].append(key)
 negTokens = {}
 negsD = util.doc2Vec(descObjs)
 for kTkn in posTokens.keys():
   negTokens[kTkn] = negSampleTokens[kTkn]
   posV = posTokens[kTkn]
   for v in posV:
      negDocVec = negsD[v]
      negTokens[kTkn] = list(set(negTokens[kTkn]).intersection(set(negDocVec)))

 negTokens = collections.OrderedDict(sorted(negTokens.items()))
 f = open(fld + "/evalHelpFiles/" + negName, "w")
 f.write(title)
 for k,v in negTokens.items():
    ll  = str(k) + ","
    ll += "-".join(v)
    ll += "\n"
    f.write(ll)
 f.close()

 kWord = ["rgb","shape","object"]
 for wd in kWord:
   f = open(fld + "/evalHelpFiles/" + wd + posName, "w")
   f1 = open(fld + "/evalHelpFiles/" +wd + negName, "w")
   sWords = []
   f.write(title)
   f1.write(title)
   if wd == "rgb":
      sWords = rgbWords
   elif wd == "shape":
      sWords = shapeWords
   elif wd == "object":
      sWords = objWords
   for k,v in posTokens.items():
    if k in sWords:
     if len(v) > 0:
       ll  = str(k) + ","
       ll += "-".join(v)
       ll += "\n"
       f.write(ll)
     v = negTokens[k]
     if len(v) > 0:
       ll  = str(k) + ","
       ll += "-".join(v)
       ll += "\n"
       f1.write(ll)
   f.close()
   f1.close()
 return 0

def getTestImages(testPosInsts,testNegInsts,posNo,negNo):
	posId = []
	negId = []
	testInstances = {}
	relevantInst = []
	if (len(testPosInsts) > 0) or (len(testNegInsts) > 0):
                if len(testPosInsts) > 0:
		   posId = [(i + 1) %  len(testPosInsts) for i in range(posNo)]
                if len(testNegInsts) > 0:
		   negId = [(inn + 1) % len(testNegInsts) for inn in range(negNo)]

		for id in posId:
			tmp = testPosInsts[id]
			imgs = objInstances[tmp]
			p1 = random.sample(range(len(imgs)), k=1)
			testInstances[imgs[p1[0]]] = tmp + " (" + objNames[tmp] + ")"
			relevantInst.append(imgs[p1[0]])

		for id in negId:
			tmp = testNegInsts[id]
			imgs = objInstances[tmp]
			p1 = random.sample(range(len(imgs)), k=1)
			testInstances[imgs[p1[0]]] = tmp + " (" + objNames[tmp] + ")"
	return (relevantInst,testInstances)

def selectCorrectImage(c,testInstances):
	probs = classifierProbs[c]
	argMax = argProb
	selInst = []
	for ik in range(len(probs)):
		probInst = probs[ik]
		for inst in testInstances.keys():
			if float(probInst[inst]) > argMax:
				#  argMax = float(probs[inst])
				selInst.append(inst)
	return list(set(selInst))

def getMatchNumbers(relevantInst,selInst,testInstances):
	tNo = float(len(testInstances))
	tP = 0.0
	fN = 0.0
	fP = 0.0
        tps = []
        if len(relevantInst) > 0:
 	   tps = list(set(relevantInst).intersection(set(selInst)))
	tP = float(len(tps))
	fP = float(len(selInst) - tP)
	fN = float(len(relevantInst) - tP)
	tN = tNo - tP - fN - fP
	return (tP,fN,fP,tN)

def getStats(tP,fN,fP,tN):
	acc = (tP + tN)/(tP + fN + fP + tN)
	if (tP + fP) > 0.0:
		prec = tP / (tP + fP)
	else:
		prec = 0.0
	if (tP + fN) > 0.0:
		rec = tP / (tP + fN)
	else:
		rec = 0.0
	f1s = 0.0
	if (prec + rec) != 0.0:
		f1s = 2.0 * prec * rec / (prec + rec)
	return (acc,prec,rec,f1s)


rootDir = fld
resultFolder = fld

resFileName = resultFolder + tID + "_" + str(cID) + "_overall-learnPerformance.csv"
perfFile1 = open(resFileName,'w')
fieldnames = np.array(['Test Batch'])
fieldnames = np.append(fieldnames,['True Positive','True Negative','False Positive','False Negative'])
fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Score'])
writer1 = csv.DictWriter(perfFile1, fieldnames=fieldnames)
writer1.writeheader()

noIndArs = []
accArs = []
f1sArs = []
precArs = []
recArs = []
#for fNo in np.arange(100,2000,100):
#for fNo in np.arange(10,200,10):
#for fNo1 in range(100):
fFldrs = []
for o in os.listdir(rootDir):
  if os.path.isdir(os.path.join(rootDir,o)):
      fFldrs.append(int(o))
fFldrs = np.sort(fFldrs)
for fNo in fFldrs:
#for fNo in [6050]:
    fName1 = predfileName + "/" + str(fNo)
#    fName1 = predfileName
    resultFolder = fName1
    fld = resultFolder
    fName = fName1 + "/groundTruthPrediction.csv"
    predfName = fName
    pos = fName1 + "/evalHelpFiles/" + tID + "posInstances.conf"
    neg = fName1 + "/evalHelpFiles/" + tID + "negInstances.conf"
    resultFileName = resultFolder + "/" + str(fNo) + "-" + str(tID) + "_" + str(cID) + "_learnPerformance.csv"
    perfFile = open(resultFileName,'w')
    fieldnames = np.array(['Classifier','Test Object Images','Ground Truth','Selected by Classifier'])
    fieldnames = np.append(fieldnames,['True Positive','True Negative','False Positive','False Negative'])
    fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Score'])
    writer = csv.DictWriter(perfFile, fieldnames=fieldnames)
    writer.writeheader()

    getTestInstancesAndClassifiers(fName)
    ret = writePosNeg()
    if ret == 1:
     noIndArs.append(str(fNo))
     accArs.append(str(0))
     f1sArs.append(str(0))
     precArs.append(str(0))
     recArs.append(str(0))
     continue
    getPosNegInstances()
    testTokens = list(set(posInsts.keys()).union(set(negInsts.keys())))
#    print testTokens
#    testTokens = list(set(posInsts.keys()).intersection(set(negInsts.keys())))
    testTokens = list(set(testTokens).intersection(set(classifierProbs.keys())))
#    print testTokens
    accFldr= []
    f1sFldr = []
    precFldr = []
    recFldr= []
    noTokens = list(set(tobeTestedTokens) - set(testTokens))
    neverAppeared =list(set(noTokens).intersection(set(neverAppeared)))
    for cc in range(len(noTokens)):
        accFldr.append(0.0)
        f1sFldr.append(0.0)
        precFldr.append(0.0)
        recFldr.append(0.0)
    for c in testTokens:
  	  dictRes = {'Classifier' : " "}
  	  writer.writerow(dictRes)
          testPosInsts = []
          testNegInsts = []
          accT = 0.0
          f1T = 0.0
          precT = 0.0
          recT = 0.0
	  if c in posInsts.keys():
           if len(posInsts[c]) > 0:
  	        testPosInsts = list(set(posInsts[c]).intersection(set(objInstances.keys())))
          if c in negInsts.keys():
           if len(negInsts[c]) > 0:
              testNegInsts = list(set(negInsts[c]).intersection(set(objInstances.keys())))
          if len(testPosInsts) > 0 or len(testNegInsts) > 0:
                  accTkn = []
                  f1sTkn = []
                  precTkn = []
                  recTkn = []
  	  	  for tms in range(10):
  	  	  	  posNo = random.sample(range(3), k=1)
                          totNo = random.sample([4,5,6], k=1)
  	  	  	  negNo = totNo[0] - posNo[0] - 1

  	  	  	  (relevantInst,testInstances) = getTestImages(testPosInsts,testNegInsts,posNo[0] + 1, negNo)
  	  	  	  selInst = selectCorrectImage(c,testInstances)

  	  	  	  (tP,fN,fP,tN) = getMatchNumbers(relevantInst,selInst,testInstances)
  	  	  	  (acc,prec,rec,f1s) = getStats(tP,fN,fP,tN)
  	  	  	  tmpObj = ""
  	  	  	  for v in testInstances.values():
  	  	  	  	  tmpObj += str(v) + "      "
  	  	  	  relInsts = ""
  	  	  	  for ik in relevantInst:
  	  	  	  	  relInsts += str(testInstances[ik]) + " "
  	  	  	  selInsts = ""

  	  	  	  for ik in selInst:
  	  	  	  	  selInsts += str(testInstances[ik]) + " "
  	  	  	  dictRes = {'Classifier' : str(c),'Test Object Images' : tmpObj, 'Ground Truth' : str(relInsts)}
  	  	  	  dictRes.update({'Selected by Classifier' : str(selInsts), 'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
			  dictRes.update({'True Positive' : str(tP),'True Negative' : str(tN) ,'False Positive' : str(fP),'False Negative' : str(fN)})

	 		  writer.writerow(dictRes)
                          accTkn.append(acc)
                          f1sTkn.append(f1s)
                          precTkn.append(prec)
                          recTkn.append(rec)
                  accT = np.mean(accTkn)
                  f1T = np.mean(f1sTkn)
                  precT = np.mean(precTkn)
                  recT = np.mean(recTkn)
          dictRes = {'Classifier' : 'Total - ' + str(c),'Accuracy' : str(accT),'Precision' : str(precT) ,'Recall' : str(recT),'F1-Score' : str(f1T)}
          writer.writerow(dictRes)
          accFldr.append(accT)
          f1sFldr.append(f1T)
          precFldr.append(precT)
          recFldr.append(recT)
    if len(accFldr) > 0:
      acc = np.mean(accFldr)
      prec = np.mean(precFldr)
      rec = np.mean(recFldr)
      f1s = np.mean(f1sFldr)
    else:
      acc = 0.0
      prec = 0.0
      rec = 0.0
      f1s = 0.0

    dictRes = {'Test Batch' : str(fNo)}
    dictRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
    writer1.writerow(dictRes)
#    print dictRes
    perfFile.close()
    noIndArs.append(str(fNo))
    accArs.append(str(acc))
    f1sArs.append(str(f1s))
    precArs.append(str(prec))
    recArs.append(str(rec))

#dictRes = {'Test Batch' : " "}
#writer1.writerow(dictRes)
#(acc,prec,rec,f1s) = getStats(otP,ofN,ofP,otN)
#dictRes = {'Test Batch' : 'Total'}
#dictRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
#dictRes.update({'True Positive' : str(otP),'True Negative' : str(otN) ,'False Positive' : str(ofP),'False Negative' : str(ofN)})
#writer1.writerow(dictRes)

#dRes = {'Test Batch' : 'Total'}
#dRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})

#print dRes
perfFile1.close()
print "Folder Number: ",
print ", ".join(noIndArs)
print "Accuracy: ",
print ", ".join(accArs)
print "F1-Score: ",
print ", ".join(f1sArs)
print "Precision: ",
print ", ".join(precArs)
print "Recall: ",
print ", ".join(recArs)
