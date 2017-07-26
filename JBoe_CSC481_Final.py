import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.metrics

def main():
    
    #Load Training Data -- 45 titles of 9 different genres, 
    
    #5 Blues, 5 Classic Rock, 5 Classical, 5 Country, 5 Electronic, 5 Metal,
    #5 Hip Hop, 5 Jazz, 5 Pop
    
    #4 Features: Note Density(Avg. # of Notes Per Second, Initial Tempo, Bass Register Importance,
    #Amount of Arpeggiation) -- All previously normalized
    
    trainList = [[-0.766678522671417,-1.4134894457127,0.0122606256864645,0.43151719310051],
                [-1.25246164516747,-1.0661631348805,1.66340797727991,1.21419919717718],
                [-1.40405821644502,-1.0661631348805,0.403817590836005,-1.09638443503824],
                [-1.3577680200252,0.971484555335123,0.649660918402909,-0.553046197585348],
                [-1.12520463926371,-1.11247330965812,2.62171635167137,0.652347745189381],
                [2.43551771310301,0.369452283225963,0.434310848102859,-0.134141978619881],
                [0.477945144638293,-0.232579988883197,0.0594181118767843,-0.0658562454473348],
                [0.870921477457971,-0.394665600604894,-0.400726511974132,0.479514784984597],
                [1.45707350937701,-0.440975775382522,-0.308871070886021,1.06130302355337],
                [0.317025254101156,0.276831933670708,0.0302952889775195,-0.313278979682085],
                [0.344564567366374,-0.811457173603543,0.0058742100751454,-0.772179500134836],
                [-1.43344191386319,-0.950387697936426,0.189561226701406,-1.63399440732091],
                [-0.509888362597354,0.624158244502915,-1.64915124629092,-2.57380317251345],
                [-0.688984501422195,-0.440975775382522,-0.573876378367778,-0.911725215749629],
                [1.0931214550796,0.415762458003591,-0.239951378380251,-0.647473539619574],
                [0.305762422927743,0.0915912345601968,0.515319672402441,0.0357254444966523],
                [0.0624347875482635,-0.37151051321608,-0.0786751179102309,-0.50150779894908],
                [-1.15105910134567,-1.25140383399101,0.486438451417823,-1.57481116359108],
                [-0.705237572473972,1.89768805088768,1.29234485530951,0.154049060043043],
                [-0.044785112264018,-0.440975775382522,0.14184574295472,0.528489953548707],
                [0.890374866596739,0.23052175889308,-0.728919490615075,-0.395906062340086],
                [0.338609204986195,0.0915912345601968,0.00240196126908647,0.675915446802265],
                [0.702159802983765,0.0452810597825691,1.56736511491168,-0.454043070733069],
                [1.57109652803916,0.161056496726638,-1.09636595497638,1.14348078111971],
                [-0.37196172410778,0.554692982336474,-0.368561234334016,0.907415824160339],
                [0.600507499201812,1.4345863031114,1.00839825378751,0.955005214604942],
                [1.10244499892636,2.17554909955344,1.81454080734429,0.363035159499528],
                [0.153359658817574,-1.18193857182456,1.6354924054874,-0.00454647255469575],
                [0.916855697221869,1.24934560400089,1.93419178349293,0.244811833289804],
                [-0.0918102995735832,-0.927232610547612,0.127335162294101,0.0839276733003873],
                [1.37119861987878,-0.811457173603543,-1.52856876509287,0.67106206208497],
                [0.93323082818666,-0.603061387104218,-0.400227918868385,1.8010528221493],
                [1.07383037479694,-0.950387697936426,-0.573990235637437,1.29493370762047],
                [-0.583104951959534,-0.834612260992357,-1.43566596814499,1.40195387262669],
                [-1.19986159572819,-1.29771400876863,-0.12304791014598,1.49648769523065],
                [-0.372297246211269,1.66613717699954,-1.11177695290247,-0.910589434995976],
                [0.399464128365263,-0.834612260992357,0.160241864875368,-0.855590862944964],
                [-1.14389931582131,-0.649371561881846,-1.38617972788128,-2.7441128533059],
                [-0.771977276287568,0.253676846281894,-0.83487270827995,0.255986573229018],
                [-1.36950601865057,-1.64504031960084,0.0851051444620558,-2.01914851836497],
                [-0.233730884591378,-0.927232610547612,0.400787709093632,0.196017786581242],
                [0.613634713378625,-0.209424901494383,0.171104729047728,-0.688776703429684],
                [-0.964928464375092,1.48089647788903,-1.63760172273251,-1.13536431429426],
                [0.262997873689561,-0.880922435769985,0.45336491398644,0.160631572105278],
                [2.46876985514246,-1.11247330965812,-1.8129457869236,1.20203427847653]]
    
    testList = [[-1.42550367089293,-0.741991911437102,1.03907139337304,-0.524103659257335],
                [-1.4713199318185,-0.834612260992357,0.610848921601832,-1.07153109849005],
                [0.0483469943547123,1.06410490489038,1.41483496062355,-0.736087518180374],
                [-0.973697981011053,0.693623506669357,-0.180733411152442,-1.16782656220974],
                [-1.0181376270085,-1.0661631348805,-0.664143752552901,0.270549075112729],
                [0.0740472545469077,-0.255735076272011,0.582212294684255,-0.128785564777901],
                [0.71758688500379,0.554692982336474,0.133199237498739,0.581912344088212],
                [-0.137417909220219,1.71244735177717,0.60518579551654,-0.154413314075152],
                [0.271454224529076,-1.25140383399101,0.613087382121505,-0.772701130690326],
                [0.864020291135146,-0.209424901494383,-0.191872343613909,0.217888209405153],
                [-0.810894246904719,1.15672525444563,-0.172523673097193,-1.04070772100896],
                [3.20706768407871,1.71244735177717,-0.643132149404525,-0.357730150445054],
                [-0.250691095660947,-0.533596124937777,-1.30676064627265,-2.08822794623238],
                [-0.795521491862785,0.137901409337824,0.150083129455051,-1.40763364196608],
                [0.372076487690884,-0.487285950160149,-0.849538792435035,-1.26744421775474],
                [-0.65660384990208,-1.29771400876863,-0.180568359917071,1.42353891497765],
                [-0.77988391630925,-0.37151051321608,1.21149957238249,-0.365286553173383],
                [-1.30604172925065,-0.139959639327942,-0.229290080643909,-1.25699057202116],
                [-1.04110227888866,-1.01985296010287,-0.330316732364811,1.54321130870156],
                [0.61373886561153,-0.00102911499505855,-0.2808348691897,0.53675115196638],
                [0.49875146997193,-0.0936494645503139,-0.832360238451796,0.651893184309962],
                [2.82271114057672,-0.139959639327942,0.0440903522786204,0.402991197507736],
                [0.0463327895684618,0.137901409337824,-0.523226280915796,1.40875258840225],
                [-0.0296713299119058,0.323142108448335,-1.02406080673175,0.319768352634448],
                [0.781983086248411,1.9439982256653,0.181776380407751,1.84028755471701],
                [0.782278454549117,-0.533596124937777,-0.276684483711903,-0.534461148257612],
                [-0.0876172574377816,0.554692982336474,1.72524522236284,0.0156455721085213],
                [0.203902110434939,1.24934560400089,1.79628504094392,0.437922107143642],
                [1.32483987974607,2.63865084732972,1.71067873124182,0.660006834959962],
                [0.851227474200866,2.17554909955344,1.95118989770912,0.772316815484984],
                [-0.403488104826915,-0.139959639327942,-2.05730647943417,1.34945888234721],
                [-1.15063158429676,-0.950387697936426,0.336974769326176,-1.58007717324087],
                [-0.532153920088487,1.01779473011275,-1.56126596914715,-0.00905096105451718],
                [-0.67501661729555,-0.139959639327942,-0.952051167678154,-0.0422238467014761],
                [-0.310849370568786,0.0915912345601968,-0.471342095626615,1.41103820392797],
                [-0.208964531651521,0.323142108448335,-0.361903894340191,-0.275842503944826],
                [-1.04654416417428,-0.464130862771335,-0.949945286117449,0.840457233938729],
                [-0.951584146504103,1.29565577877852,0.323269958277516,-0.313860836548317],
                [0.184463802998577,-0.186269814105569,-0.672037942893625,0.240047252985395],
                [-0.0698549841159306,2.54603049777446,-0.956718963479093,-0.654542118681162],
                [0.191338979070559,0.253676846281894,-0.723803925093155,-0.137064058860264],
                [0.358325281596958,-0.139959639327942,-0.277426681630464,-0.0153182994681455],
                [-1.43216073444622,-0.163114726716755,1.53782098070242,1.1224554353098],
                [-0.536067966587311,-0.139959639327942,1.0180011125623,1.47349893131731],
                [0.666671687756801,-1.4134894457127,-1.89213174857686,0.956901667933616]]
    
    X = []
    Y = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8]
    
    shuffle = list(zip(testList,Y))
    testList, Y = zip(*shuffle)
    
    neigh = KNeighborsClassifier(n_neighbors=10) #With 10 neighbors, accuracy rate jumps from 0.28 to 0.35
    svc = SVC(random_state=2,decision_function_shape='ovr')
    mlp = MLPClassifier(alpha=1)
    gaussian = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    dT = DecisionTreeClassifier()
    rF = RandomForestClassifier()
    nB = GaussianNB()
    aB = AdaBoostClassifier()
    qD = QuadraticDiscriminantAnalysis()
    
    kNNprediction = []
    svcPrediction = []
    mlpPrediction = []
    gaussianPrediction = []
    dTPrediction = []
    rFPrediction = []
    nBPrediction = []
    aBPrediction = []
    qDPrediction = []
    
    tests = []
    
    for song in trainList:
        X.append(song)
    neigh.fit(X,Y)
    svc.fit(X,Y)
    mlp.fit(X,Y)
    gaussian.fit(X, Y)
    dT.fit(X,Y)
    rF.fit(X,Y)
    nB.fit(X,Y)
    aB.fit(X,Y)
    qD.fit(X,Y)
    
    for song in testList:
        tests.append(song)
        kNNprediction.append(int(neigh.predict(song)))
        svcPrediction.append(int(svc.predict(song)))
        mlpPrediction.append(int(mlp.predict(song)))
        gaussianPrediction.append(gaussian.predict(song))
        dTPrediction.append(dT.predict(song))
        rFPrediction.append(rF.predict(song))
        nBPrediction.append(nB.predict(song))
        aBPrediction.append(aB.predict(song))
        qDPrediction.append(qD.predict(song))
    conMatrixKNN = sklearn.metrics.confusion_matrix(Y, kNNprediction)
    conMatrixSVC = sklearn.metrics.confusion_matrix(Y,svcPrediction)
    conMatrixMLP = sklearn.metrics.confusion_matrix(Y,mlpPrediction)
    conMatrixGaussian = sklearn.metrics.confusion_matrix(Y,gaussianPrediction)
    conMatrixDT = sklearn.metrics.confusion_matrix(Y,dTPrediction)
    conMatrixRF = sklearn.metrics.confusion_matrix(Y, rFPrediction)
    conMatrixNB = sklearn.metrics.confusion_matrix(Y, nBPrediction)
    conMatrixAB = sklearn.metrics.confusion_matrix(Y, aBPrediction)
    conMatrixQD = sklearn.metrics.confusion_matrix(Y, qDPrediction)
       
    
    accKNN = sklearn.metrics.accuracy_score(Y,kNNprediction)
    accSVC = sklearn.metrics.accuracy_score(Y,svcPrediction)
    accMLP = sklearn.metrics.accuracy_score(Y,mlpPrediction)
    accGaussian = sklearn.metrics.accuracy_score(Y, gaussianPrediction)
    accDT = sklearn.metrics.accuracy_score(Y, dTPrediction)
    accRF = sklearn.metrics.accuracy_score(Y, rFPrediction)
    accNB = sklearn.metrics.accuracy_score(Y, nBPrediction)
    accAB = sklearn.metrics.accuracy_score(Y, aBPrediction)
    accQD = sklearn.metrics.accuracy_score(Y, qDPrediction)
    
    enumeration= ['Blues','Blues','Blues','Blues','Blues','Classic Rock','Classic Rock',
                  'Classic Rock','Classic Rock','Classic Rock','Classical','Classical',
                  'Classical','Classical','Classical', 'Country','Country','Country',
                  'Country','Country','Electronic','Electronic','Electronic','Electronic',
                  'Electronic','Metal','Metal','Metal','Metal','Metal','Hip-Hop','Hip-Hop',
                  'Hip-Hop','Hip-Hop','Hip-Hop','Jazz','Jazz','Jazz','Jazz','Jazz',
                  'Pop','Pop','Pop','Pop','Pop']
    
    print("KNN Accuracy rate: " + str(accKNN) + "\n")    
    print("SVC Accuracy rate: " + str(accSVC) + "\n")
    print("MLP Accuracy Rate: " + str(accMLP) + "\n")
    print("Gaussian Accuracy Rate: " + str(accGaussian) + "\n")
    print("Decision Tree Accuracy Rate: " + str(accDT) + "\n")
    print("Random Forest Accuracy Rate: " + str(accRF) + "\n")
    print("Naive Bayes Accuracy Rate: " + str(accNB) + "\n")
    print("AdaBoost Accuracy Rate: " + str(accAB) + "\n")
    print("Quadratic Discriminant Analysis Accuracy Rate: " + str(accMLP) + "\n")
    
    
    #print("KNN Test Data (Predicted): " + str(kNNprediction) + "\n")
    #print("SVC Test Data (Predicted): " + str(svcPrediction) + "\n")    
    #print("MLP Test Data (Predicted): " + str(mlpPrediction) + "\n")
    #print("Gaussian Test Data (Predicted): " + str(gaussianPrediction) + "\n")
    #print("Decision Tree Test Data (Predicted): " + str(dTPrediction) + "\n")
    #print("Random Forest Test Data (Predicted): " + str(rFPrediction) + "\n")
    #print("Naive Bayes Test Data (Predicted): " + str(nBPrediction) + "\n")
    #print("AdaBoost Test Data (Predicted): " + str(aBPrediction) + "\n")
    #print("Quadratic Discriminant Analysis Test Data (Predicted): " + str(qDPrediction) + "\n")
    
    
    #print("KNN Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixKNN) + "\n\n")
    #print("\n KNN Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,kNNprediction, target_names=enumeration)))
    #("\n")
    #print("SVC Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixSVC) + "\n\n")
    #print("\n SVC Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,svcPrediction, target_names=enumeration)))
    #print("MLP Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixMLP) + "\n\n")
    #print("\n Multilayer Perceptron Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,mlpPrediction, target_names=enumeration)))
    #print("Gaussian Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixGaussian) + "\n\n")
    #print("\n Gaussian Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,gaussianPrediction, target_names=enumeration)))
    #print("Decision Tree Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixDT) + "\n\n")
    #print("\n Decision Tree Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,dTPrediction, target_names=enumeration)))
    #print("Random Forest Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixRF) + "\n\n")
    #print("\n Random Forest Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,rFPrediction, target_names=enumeration)))
    #print("AdaBoost Classifier Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixAB) + "\n\n")
    #print("\n AdaBoost Classifier Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,aBPrediction, target_names=enumeration)))
    #print("Naive Bayes Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixNB) + "\n\n")
    #print("\n Naive Bayes Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,nBPrediction, target_names=enumeration)))
    #print("Quadratic Discriminant Analysis Confusion Matrix : ")
    #print("\n")
    #print(str(conMatrixQD) + "\n\n")
    #print("\n Quadratic Discriminant Analysis Classification Report")
    #print(str(sklearn.metrics.classification_report(Y,qDPrediction, target_names=enumeration)))
    
    
     
    

if __name__ == "__main__":
    main()