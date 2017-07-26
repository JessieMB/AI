import math
from pts_loader import load
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
      
def eyeLength(point8x, point8y, point13x, point13y):
        distance = math.sqrt((math.pow((point8x - point13x),2)) + (math.pow((point8y - point13y),2)))
        return distance

def eyeLengthRatio(point9x,point9y, point10x, point10y, eyeLengthSize):
        distance = math.sqrt((math.pow((point9x - point10x),2)) + (math.pow((point9y - point10y),2)))
        distanceRatio = distance / eyeLengthSize
        return distanceRatio
    
def eyeDistance(point0x, point0y, point1x, point1y, eyeLength):
        centerDistance = math.sqrt((math.pow((point0x - point1x),2)) + (math.pow((point0y - point1y),2)))
        distanceRatio = centerDistance / eyeLength
        return distanceRatio
    
def jawDistance(point20x, point20y, point21x, point21y):
        distanceRatio = math.sqrt((math.pow((point20x - point21x),2)) + (math.pow((point20y - point21y),2)))
        return distanceRatio 
    
def noseDistance(point15x, point15y, point16x, point16y, jawLength):
        noseRatio = math.sqrt((math.pow((point15x - point16x),2)) + (math.pow((point15y - point16y),2)))
        distanceRatio = noseRatio / jawLength
        return distanceRatio

def lipSize(point2x, point2y, point3x, point3y, point17x, point17y, point18x, point18y):
        lipsLength = math.sqrt((math.pow((point2x - point3x),2)) + (math.pow((point2y - point3y),2)))
        lipsWidth = math.sqrt((math.pow((point17x - point18x),2)) + (math.pow((point17y - point18y),2)))   
        distanceRatio = lipsLength / lipsWidth
        return distanceRatio
    
def lipLength(point2x, point2y, point3x, point3y, jawLength):
        lipsLength = math.sqrt((math.pow((point2x - point3x),2)) + (math.pow((point2y - point3y),2)))
        distanceRatio = lipsLength / jawLength
        return distanceRatio
    
def eyeBrows(point4x, point4y, point5x, point5y, point6x, point6y, point7x, point7y, eyesLength):
        ratio45 = math.sqrt((math.pow((point4x - point5x),2)) + (math.pow((point4y - point5y),2)))
        ratio56 = math.sqrt((math.pow((point5x - point6x),2)) + (math.pow((point5y - point6y),2)))   
        #Clause to determine which ratio should be used
        if (ratio56 > ratio45):
                browRatio = ratio56
        else:
                browRatio = ratio45
    
        distanceRatio = (browRatio / eyesLength)
    
        return distanceRatio
    
def aggressive(point10x, point10y, point19x, point19y, jawLength):
        agLength = math.sqrt((math.pow((point10x - point19x),2)) + (math.pow((point10y - point19y),2)))
        distanceRatio = agLength / jawLength
        return distanceRatio

def analyze(fIn):              
        path = fIn
        points = load(path)
    
        xList = [coord[0] for coord in points]
        yList = [coord[1] for coord in points]
    
        point0x = xList[0]
        point0y = yList[0]
        point1x = xList[1]
        point1y = yList[1]
        point2x = xList[2]
        point2y = yList[2]
        point3x = xList[3]
        point3y = yList[3]
        point4x = xList[4]
        point4y = yList[4]
        point5x = xList[5]
        point5y = yList[5]
        point6x = xList[6]
        point6y = yList[6]
        point7x = xList[7]
        point7y = yList[7]
        point8x = xList[8]
        point8y = yList[8]
        point9x = xList[9]
        point9y = yList[9]
        point10x = xList[10]
        point10y = yList[10]
        point11x = xList[11]
        point11y = yList[11]        
        point12x = xList[12]
        point12y = yList[12]        
        point13x = xList[13]
        point13y = yList[13]        
        point14x = xList[14]
        point14y = yList[14]        
        point15x = xList[15]
        point15y = yList[15]        
        point16x = xList[16]
        point16y = yList[16]        
        point17x = xList[17]
        point17y = yList[17]        
        point18x = xList[18]
        point18y = yList[18]        
        point19x = xList[19]
        point19y = yList[19]        
        point20x = xList[20]
        point20y = yList[20]        
        point21x = xList[21]
        point21y = yList[21]        
    
        count = 1
        #print("X Coordinates")
        for line in xList:
                #print(count, line)
                count = count +1         
    
        count = 1    
        #print("")
        #print("Y Coordinates")        
        for line in yList:
                #print(count, line)
                count = count +1   
    
        eyesLength = eyeLength(point8x, point8y, point13x, point13y)
        eyeLengthRat = eyeLengthRatio(point9x, point9y, point10x, point10y, 
                                     eyesLength)

        #print("The eye length ratio is " + str(eyeLengthRat))
    
        eyeDistanceRatio = eyeDistance(point0x, point0y, point1x, point1y, eyesLength)  
    
        #print("The eye distance ratio is " + str(eyeDistanceRatio))
    
        jawLength = jawDistance(point20x, point20y, point21x, point21y)
        
        noseRatio = noseDistance(point15x, point15y, point16x, point16y, 
                                 jawLength)
    
        #print("The nose ratio is " + str(noseRatio))
    
        lipSizeRatio = lipSize(point2x, point2y, point3x, point3y, point17x, 
                               point17y, 
                               point18x, 
                               point18y)
    
        #print("The lip size ratio is " + str(lipSizeRatio))
    
        lipLengthRatio = lipLength(point2x, point2y, point3x, point3y, 
                                   jawLength)
    
        #print("The lip length ratio is " + str(lipLengthRatio))
    
        eyeBrowRatio = eyeBrows(point4x, point4y, point5x, point5y, point6x, 
                                point6y, 
                                point7x, 
                                point7y, 
                                eyesLength)
    
        #print("The eyebrow length ratio is " + str(eyeBrowRatio))
    
        aggressiveRatio = aggressive(point10x, point10y, point19x, point19y, 
                                     jawLength)
    
        #print("The aggressive ratio is " + str(aggressiveRatio))
    
        return([eyeLengthRat, eyeDistanceRatio, noseRatio, lipSizeRatio, lipLengthRatio,eyeBrowRatio,aggressiveRatio])    
    
def loadFiles():
        
        #Load training data
        man1 = ("m-001-01.pts")
        man2 = ("m-002-01.pts")
        man3 = ("m-003-01.pts")
        man4 = ("m-004-01.pts")
        man5 = ("m-005-01.pts")
        woman1 = ("w-001-01.pts")
        woman2 = ("w-002-01.pts")
        woman3 = ("w-003-01.pts")
        woman4 = ("w-004-01.pts")
        woman5 = ("w-005-01.pts")
        
        #Load testing data
        
        man1b = ("m-001-05.pts")
        man2b = ("m-002-05.pts")
        man3b = ("m-003-05.pts")
        man4b = ("m-004-05.pts")
        man5b = ("m-005-05.pts")
        woman1b = ("w-001-05.pts")
        woman2b = ("w-002-05.pts")
        woman3b = ("w-003-05.pts")
        woman4b = ("w-004-05.pts")
        woman5b = ("w-005-05.pts")        
        
        trainList = [man1,man2,man3,man4,man5,woman1,woman2,woman3,woman4,woman5]
        testList = [man1b,man2b,man3b,man4b,man5b,woman1b,woman2b,woman3b,woman4b,woman5b]
        
        X = []
        Y = [x for x in range(10)]
        neigh = KNeighborsClassifier(n_neighbors=1)
        prediction = []
        tests = []
        
        for person in trainList:
                trainResults = (analyze(person))
                X.append(trainResults)
                #print(neigh.predict(results))
        neigh.fit(X, Y)
        for person in testList:
                genuineTest = (analyze(person))
                tests.append(genuineTest)
                prediction.append(int(neigh.predict(genuineTest)))                
        
        for x in genuineTest:
                print(genuineTest, x)
        #for x in range(10):
        
        print(X)       
        print(prediction)    
        print(tests)
        trueY = [x for x in range(10)]
        conMatrix = sklearn.metrics.confusion_matrix(trueY, prediction)
        acc = sklearn.metrics.accuracy_score(trueY,prediction)
        enumeration = ['Man 1','Man 2', 'Man 3', 'Man 4', 'Man 5','Woman 1', 'Woman 2', 'Woman 3', 'Woman 4', 'Woman 5']
        print("Test Data (Predicted): " + str(prediction) + "\n")
        print("Accuracy rate: " + str(acc) + "\n")
        print("Confusion Matrix : ")
        print("\n")
        print(str(conMatrix) + "\n\n")
        print("\n")
        print(str(sklearn.metrics.classification_report(trueY,prediction, target_names=enumeration)))
                      
loadFiles()