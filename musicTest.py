import music21

def main():
    
    metal = ['testPreacher.xml','testamentOver.xml','metallicaBattery.xml', 'metallicaMotorbreath.xml','blackHeaven.xml','blackNIB.xml','carcassDeath.xml','carcassHeartwork.xml','slayerRainingBlood.xml','slayerWar.xml']
    blues = ['muddyMannish.xml','muddyI.xml','tajFishing.xml','bessieStormy.xml','bessieBlackwater.xml','johnsonSweet.xml','johnsonKindhearted.xml',
                'hookerBoom.xml','buddyEveryday.xml','buddyStone.xml']
    classicRock = ['blueDont.xml','bostonMore.xml','deepHush.xml','deepSmoke.xml','eaglesTake.xml','journeyDont.xml','ledRock.xml',
             'ledWhole.xml','queenBohemian.xml','styxCome.xml']    
    classical = ['vivaldiPrimavera.xml','vivaldiBaroque.xml','paganiniMoto.xml','paganiniCaprice.xml','mozartFigaro.xml','beethovenMoonlight.xml','beethoven5th.xml',
             'bachViolin.xml','mozartRondo.xml','bachFugueD.xml']
    country = ['brooksBoot.xml','garthFriends.xml','garthShameless.xml','johnnyFolson.xml','johnnyFire.xml','dixieTrouble.xml','merleCarolyn.xml',
                   'willieBlue.xml','willieCrazy.xml','shaniaMan.xml']    
    electronic = ['aviciiLevels.xml','bennyIllusion.xml','bennySatisfaction.xml','calvinIm.xml','calvinOutside.xml','deadGhosts.xml','deadStrobe.xml',
               'swedishAntidote.xml','skrillexScary.xml','skrillexBang.xml']      
    jazz = ['djangoMinor.xml','pacoZyryab.xml','dukeSatin.xml','dukeThings.xml','ellaFly.xml','louisOne.xml','louisWorld.xml',
                  'wesWhat.xml','wesCalifornia.xml','milesSo.xml']
    hipHop = ['tupacCal.xml','fiftyHate.xml','fiftyPIMP.xml','dmxRuff.xml','dmxX.xml','jayzBig.xml','biggieBig.xml',
            'wutangCream.xml','wutangGravel.xml','rundmcIts.xml']
    pop = ['taylorWe.xml','taylorLove.xml','britneyBaby.xml','oasisWonderwall.xml','georgeCareless.xml','madonnaMaterial.xml','madonnaVirgin.xml',
              'gagaPoker.xml','gagaBad.xml','bieberBaby.xml']
    
    #Add training data
    trainData = []
    for file in blues: 
        trainData.append(music21.converter.parse(file))
    for file in classicRock:
        trainData.append(music21.converter.parse(file))
    for file in classical:
        trainData.append(music21.converter.parse(file))
    for file in country:
        trainData.append(music21.converter.parse(file))
    for file in electronic: 
        trainData.append(music21.converter.parse(file))
    for file in metal:
        trainData.append(music21.converter.parse(file))
    for file in hipHop:
        trainData.append(music21.converter.parse(file))
    for file in jazz:
        trainData.append(music21.converter.parse(file))  
    for file in pop:
        trainData.append(music21.converter.parse(file))          
    # This is where we comment in features, one at a time, to gather the data.
    # This is much too data intensive to handle in aggregate and took hours individually
    # Getting all song data for one of the features takes about 45 minutes per run
    
    dataList = []
    for element in trainData:
        print(element.metadata.title)
        #p = music21.analysis.discrete.Ambitus() #pitch measures object
        #fe = music21.features.jSymbolic.AverageMelodicIntervalFeature(
            #element)
        #ff = music21.features.jSymbolic.AverageNoteDurationFeature(
            #element)
        #fi = music21.features.native.MajorTriadSimultaneityPrevalence(
            #element)
        #fj = music21.features.native.IncorrectlySpelledTriadPrevalence
 
        #fi = music21.features.native.DiminishedSeventhSimultaneityPrevalence(
            #element) not providing useful data
     
        fi = music21.features.jSymbolic.NoteDensityFeature(element)
        
        #fj = music21.features.jSymbolic.MostCommonMelodicIntervalFeature(
            #element)
        #fk = music21.features.jSymbolic.InitialTempoFeature(element) #Tempo 
      
        i = fi.extract()
        dataList.append(i.vector)
        print(i.description)
        print(i.vector)
    print(dataList)

    
if __name__ == "__main__":
    main()