/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import development.DataSets;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import utilities.*;
import weka.classifiers.meta.RotationForest;

/**
 *
 * @author phillipperks
 */
public class ActivityRecognitionProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        ///gpfs/home/xju14zpu/AllExperiments/MVMotionUni.arff
        Instances all = ClassifierTools.loadData("\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\MVMotionUni.arff");
        String path="\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\MVMotionMulti.arff";
        Instances multiVariate = ClassifierTools.loadData(path);
        int totalFolds = 30;
        
//        int classes []  = new int [4];
//        int count = 0;
//        for(Instance i: all){
//            count++;
//            classes[(int)i.classValue()]++;
//        }
//        System.out.println("Count:\t" + count);
//        System.out.println("Smash:\t" + classes[0]);
//        System.out.println("Clear:\t" + classes[1]);
//        System.out.println("Forehand:\t" + classes[2]);
//        System.out.println("Backhand:\t" + classes[3]);
        
        double aveBayes = 0;
        double aveKNN =  0;
        double aveJ48 = 0;
        double aveRF [] = new double [2];
        double aveEuclid[] = new double [2];
        double aveRotation [] = new double [2];
        double aveSMO [] = new double [2];
        double aveANN [] = new double [2];
        
        double aveKNN3 =  0;
        double aveKNN5 =  0;
        double aveKNN7 =  0;
        double aveKNN9 =  0;
        double aveKNN11 =  0;
        double aveKNN13 =  0;
        double aveKNN15 =  0;
        double aveKNN17 =  0;
        double aveKNN19 =  0;
        double aveKNN21 =  0;
        double aveKNN23 =  0;
        double aveKNN25 =  0;
        double aveKNN27 =  0;
        double aveKNN29 =  0;
        double aveKNN31 =  0;
        double aveKNN33 =  0;
        double aveKNN35 =  0;
        double aveKNN37 =  0;
        double aveKNN39 =  0;
        double aveKNN41 =  0;
        double aveKNN43 =  0;
        double aveKNN45 =  0;
        double aveKNN47 =  0;
        double aveKNN49 =  0;
        
        double aveRF10 [] = new double [2];
        double aveRF50 [] = new double [2];
        double aveRF100 [] = new double [2];
        double aveRF150 [] = new double [2];
        double aveRF200 [] = new double [2];
        double aveRF250 [] = new double [2];
        double aveRF300 [] = new double [2];
        double aveRF350 [] = new double [2];
        double aveRF400 [] = new double [2];
        double aveRF450 [] = new double [2];
        double aveRF500 [] = new double [2];
        
        double aveRotation10 [] = new double [2];
        double aveRotation50 [] = new double [2];
        double aveRotation100 [] = new double [2];
        double aveRotation150 [] = new double [2];
        double aveRotation200 [] = new double [2];
        double aveRotation250 [] = new double [2];
        double aveRotation300 [] = new double [2];
        double aveRotation350 [] = new double [2];
        double aveRotation400 [] = new double [2];
        double aveRotation450 [] = new double [2];
        double aveRotation500 [] = new double [2];
        
        double aveBasicDTW [] = new double [2];
        double aveDTWI [] = new double [2];
        double aveDTWD [] = new double [2];
        double aveDTWA [] = new double [2];
        double aveEnhancedDTWA [] = new double [3];
        double aveEnhancedDTWI [] = new double [2];
        
        
        double aveKNNDTWI3 [] = new double [2];
        double aveKNNDTWI5 [] = new double [2];
        double aveKNNDTWI7 [] = new double [2];
        double aveKNNDTWI9 [] = new double [2];
        double aveKNNDTWI11 [] = new double [2];
        double aveKNNDTWI13 [] = new double [2];
        double aveKNNDTWI15 [] = new double [2];
        double aveKNNDTWI17 [] = new double [2];
        double aveKNNDTWI19 [] = new double [2];
        double aveKNNDTWI21 [] = new double [2];
        double aveKNNDTWI23 [] = new double [2];
        double aveKNNDTWI25 [] = new double [2];
        double aveKNNDTWI27 [] = new double [2];
        double aveKNNDTWI29 [] = new double [2];
        double aveKNNDTWI31 [] = new double [2];
        double aveKNNDTWI33 [] = new double [2];
        double aveKNNDTWI35 [] = new double [2];
        double aveKNNDTWI37 [] = new double [2];
        double aveKNNDTWI39 [] = new double [2];
        double aveKNNDTWI41 [] = new double [2];
        double aveKNNDTWI43 [] = new double [2];
        double aveKNNDTWI45 [] = new double [2];
        double aveKNNDTWI47 [] = new double [2];
        double aveKNNDTWI49 [] = new double [2];
        

        String [][] dataPaths = new String [11][2];
//        dataPaths [0][0] = "Basic DTW";
//        dataPaths[0][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/BASICDTW/basicDTWTest";
//        dataPaths [1][0] = "DTWA";
//        dataPaths[1][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/DTWA/DTWATest";
//        dataPaths [2][0] = "DTWD";
//        dataPaths[2][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/DTWD/DTWDTest";
//        dataPaths [3][0] = "DTWI";
//        dataPaths[3][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/DTWI/DTWITest";
//        dataPaths [4][0] = "Enhanced DTWI";
//        dataPaths[4][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/EnhancedDTWI/EnhancedDTWITest";
//        dataPaths [5][0] = "Random Forest";
//        dataPaths[5][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/RANDOMFOREST/randomForestTest";

//        dataPaths [0][0] = "Bayes";
//        dataPaths[0][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST/bayesTest";
//        dataPaths [1][0] = "Euclidean";
//        dataPaths[1][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/EUCLIDEANDISTANCE/euclideanTest";
//        dataPaths [2][0] = "KNN";
//        dataPaths[2][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/KNN/knnTest";
//        dataPaths [3][0] = "DTWI";
//        dataPaths[3][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/J48/J48Test";
//        dataPaths [4][0] = "Random Forest";
//        dataPaths[4][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/RANDOMFOREST/randomForestTest";
//        dataPaths [5][0] = "Rotation Forest";
//        dataPaths[5][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST/rotationTest";
//        dataPaths [6][0] = "SVM";
//        dataPaths[6][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/SVM/SVMTest";
//        dataPaths [7][0] = "ANN";
//        dataPaths[7][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ANN/ANNTest";

        dataPaths [0][0] = "10";
        dataPaths[0][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST10/rotationTest";
        dataPaths [1][0] = "50";
        dataPaths[1][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST50/rotationTest";
        dataPaths [2][0] = "100";
        dataPaths[2][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST100/rotationTest";
        dataPaths [3][0] = "150";
        dataPaths[3][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST150/rotationTest";
        dataPaths [4][0] = "200";
        dataPaths[4][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST200/rotationTest";
        dataPaths [5][0] = "250";
        dataPaths[5][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST250/rotationTest";
        dataPaths [6][0] = "300";
        dataPaths[6][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST300/rotationTest";
        dataPaths [7][0] = "350";
        dataPaths[7][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST350/rotationTest";
        dataPaths [8][0] = "400";
        dataPaths[8][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST400/rotationTest";
        dataPaths [9][0] = "450";
        dataPaths[9][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST450/rotationTest";
        dataPaths [10][0] = "500";
        dataPaths[10][1] = "/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/ROTATIONFOREST500/rotationTest";

//        Results.writeTable(dataPaths, "Rotation-Forest");
//        Results.writeNLLTable(dataPaths, "Further Experiments NLL");
        //System.out.println("NLL\t" + Results.NLL("/Users/phillipperks/Desktop/3rd-Year-Project/FinalResults/EnhancedDTWI/EnhancedDTWITest", 30));
        for(int fold=0; fold<totalFolds; fold++){
            System.out.println("fold:\t" + fold);
            Instances[] data = InstanceTools.resampleInstances(all, fold, .5);

            Instances train = data[0];
            Instances test = data[1];
            
            Instances[] multiData = InstanceTools.resampleInstances(multiVariate, fold, .5);
            Instances multiTrain = multiData[0];
            Instances  multiTest = multiData[1];
            
            
            //Further Experiments
        //String [][] dataPaths = new String [5][2];
        dataPaths [0][0] = "Basic DTW";
        dataPaths[0][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\BASICDTW\\basicDTWTest";
        dataPaths [1][0] = "DTWD";
        dataPaths[1][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\DTWD\\DTWDTest";
        dataPaths [2][0] = "DTWI";
        dataPaths[2][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\DTWI\\DTWITest";
        dataPaths [3][0] = "DTWA";
        dataPaths[3][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\DTWA\\DTWATest";
        dataPaths [4][0] = "Enhanced DTWI";
        dataPaths[4][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\EnhancedDTWI\\EnhancedDTWITest";
//        dataPaths [5][0] = "RANDOMFOREST250";
//        dataPaths[5][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\RANDOMFOREST250\\randomForestTest";
//        dataPaths [6][0] = "RANDOMFOREST300";
//        dataPaths[6][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\RANDOMFOREST300\\randomForestTest";
//        dataPaths [7][0] = "RANDOMFOREST350";
//        dataPaths[7][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\RANDOMFOREST350\\randomForestTest";
//        dataPaths [8][0] = "RANDOMFOREST400";
//        dataPaths[8][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\RANDOMFOREST400\\randomForestTest";
//        dataPaths [9][0] = "RANDOMFOREST450";
//        dataPaths[9][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\RANDOMFOREST450\\randomForestTest";
//        dataPaths [10][0] = "RANDOMFOREST500";
//        dataPaths[10][1] = "\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Project\\Quick\\FinalResults\\RANDOMFOREST500\\randomForestTest";
        
        
        
        Results.writeTable(dataPaths, "Time Series");
        
//        for(int fold=0; fold<totalFolds; fold++){
//            Instances[] data = InstanceTools.resampleInstances(all, fold, .5);
//
//            Instances train = data[0];
//            Instances test = data[1];
//            
//            Instances[] multiData = InstanceTools.resampleInstances(multiVariate, fold, .5);
//            Instances multiTrain = multiData[0];
//            Instances  multiTest = multiData[1];
//            
//            
//            //Further Experiments
//            aveEuclid = runEuclideanDistance(aveEuclid,test, train, fold);
//            aveBayes = runBayes(aveBayes,test, train, fold);
//            aveKNN = runKNN(aveKNN,test, train, fold);
//            aveJ48 = runJ48(aveJ48,test, train, fold);
//            aveRF = runRandomForest(aveRF,test, train, fold);
//            aveRotation = runRotationForest(aveRotation,test, train, fold);

//            aveSMO = runSMO(aveSMO, test, train, fold);

//            aveANN = runANN(aveANN, test, train, fold);
//
//
//            aveKNN3 = runKNN(aveKNN,test, train, fold, 3);
//            aveKNN5 = runKNN(aveKNN,test, train, fold, 5);
//            aveKNN7 = runKNN(aveKNN,test, train, fold, 7);
//            aveKNN9 = runKNN(aveKNN,test, train, fold, 9);
//            aveKNN11 = runKNN(aveKNN,test, train, fold, 11);
//            aveKNN13 = runKNN(aveKNN,test, train, fold, 13);
//            aveKNN15 = runKNN(aveKNN,test, train, fold, 15);
//            aveKNN17 = runKNN(aveKNN,test, train, fold, 17);
//            aveKNN19 = runKNN(aveKNN,test, train, fold, 19);
//            aveKNN21 = runKNN(aveKNN,test, train, fold, 21);
//            aveKNN23 = runKNN(aveKNN,test, train, fold, 23);
//            aveKNN25 = runKNN(aveKNN,test, train, fold, 25);
//            aveKNN27 = runKNN(aveKNN,test, train, fold, 27);
//            aveKNN29 = runKNN(aveKNN,test, train, fold, 29);
//            aveKNN31 = runKNN(aveKNN,test, train, fold, 31);
//            aveKNN33 = runKNN(aveKNN,test, train, fold, 33);
//            aveKNN35 = runKNN(aveKNN,test, train, fold, 35);
//            aveKNN37 = runKNN(aveKNN,test, train, fold, 37);
//            aveKNN39 = runKNN(aveKNN,test, train, fold, 39);
//            aveKNN41 = runKNN(aveKNN,test, train, fold, 41);
//            aveKNN43 = runKNN(aveKNN,test, train, fold, 43);
//            aveKNN45 = runKNN(aveKNN,test, train, fold, 45);
//            aveKNN47 = runKNN(aveKNN,test, train, fold, 47);
//            aveKNN49 = runKNN(aveKNN,test, train, fold, 49);
//
//            aveRF10 = runRandomForest(aveRF,test, train, fold, 10);
//            aveRF50 = runRandomForest(aveRF,test, train, fold, 50);
//            aveRF100 = runRandomForest(aveRF,test, train, fold, 100);
//            aveRF150 = runRandomForest(aveRF,test, train, fold, 150);
//            aveRF200 = runRandomForest(aveRF,test, train, fold, 200);
//            aveRF250 = runRandomForest(aveRF,test, train, fold, 250);
//            aveRF300 = runRandomForest(aveRF,test, train, fold, 300);
//            aveRF350 = runRandomForest(aveRF,test, train, fold, 350);
//            aveRF400 = runRandomForest(aveRF,test, train, fold, 400);
//            aveRF450 = runRandomForest(aveRF,test, train, fold, 450);
//            aveRF500 = runRandomForest(aveRF,test, train, fold, 500);
//
////            aveRotation10 = runRotationForest(aveRotation,test, train, fold, 10);
////            aveRotation50 = runRotationForest(aveRotation,test, train, fold, 50);
////            aveRotation100 = runRotationForest(aveRotation,test, train, fold, 100);
////            aveRotation150 = runRotationForest(aveRotation,test, train, fold, 150);
////            aveRotation200 = runRotationForest(aveRotation,test, train, fold, 200);
////            aveRotation250 = runRotationForest(aveRotation,test, train, fold, 250);
////            aveRotation300 = runRotationForest(aveRotation,test, train, fold, 300);
////            aveRotation350 = runRotationForest(aveRotation,test, train, fold, 350);
////            aveRotation400 = runRotationForest(aveRotation,test, train, fold, 400);
////            aveRotation450 = runRotationForest(aveRotation,test, train, fold, 450);
////            aveRotation500 = runRotationForest(aveRotation,test, train, fold, 500);
//
////            
////            //time sereies experimnets
//            aveBasicDTW = runBasicDTW(aveBasicDTW,test, train, fold);
//            aveDTWI = runDTWI(aveDTWI, multiTest, multiTrain, fold);
//            aveDTWD = runDTWD(aveDTWD, multiTest, multiTrain, fold);
//            aveDTWA = runDTWA(aveDTWA, multiTest, multiTrain, fold);
            aveEnhancedDTWA = runEnhancedDTWA(aveEnhancedDTWA, multiTest, multiTrain, fold);
//            aveEnhancedDTWI = runEnhancedDTWI(aveEnhancedDTWI, multiTest, multiTrain, fold);
//
//
////            aveKNNDTWI3 = runKNNDTWI(aveKNNDTWI3, multiTest, multiTrain, fold, 3);
////            aveKNNDTWI5 = runKNNDTWI(aveKNNDTWI5, multiTest, multiTrain, fold, 5);
////            aveKNNDTWI7 = runKNNDTWI(aveKNNDTWI7, multiTest, multiTrain, fold, 7);
////            aveKNNDTWI9 = runKNNDTWI(aveKNNDTWI9, multiTest, multiTrain, fold, 9);
////            aveKNNDTWI11 = runKNNDTWI(aveKNNDTWI11, multiTest, multiTrain, fold, 11);
////            aveKNNDTWI13 = runKNNDTWI(aveKNNDTWI7, multiTest, multiTrain, fold, 13);
////            aveKNNDTWI15 = runKNNDTWI(aveKNNDTWI13, multiTest, multiTrain, fold, 15);
////            aveKNNDTWI17 = runKNNDTWI(aveKNNDTWI17, multiTest, multiTrain, fold, 17);
////            aveKNNDTWI19 = runKNNDTWI(aveKNNDTWI19, multiTest, multiTrain, fold, 19);
////            aveKNNDTWI21 = runKNNDTWI(aveKNNDTWI21, multiTest, multiTrain, fold, 21);
////            aveKNNDTWI23 = runKNNDTWI(aveKNNDTWI23, multiTest, multiTrain, fold, 23);
////            aveKNNDTWI25 = runKNNDTWI(aveKNNDTWI25, multiTest, multiTrain, fold, 25);
////            aveKNNDTWI27 = runKNNDTWI(aveKNNDTWI27, multiTest, multiTrain, fold, 27);
////            aveKNNDTWI29 = runKNNDTWI(aveKNNDTWI29, multiTest, multiTrain, fold, 29);
////            aveKNNDTWI31 = runKNNDTWI(aveKNNDTWI31, multiTest, multiTrain, fold, 31);
////            aveKNNDTWI33 = runKNNDTWI(aveKNNDTWI33, multiTest, multiTrain, fold, 33);
////            aveKNNDTWI35 = runKNNDTWI(aveKNNDTWI35, multiTest, multiTrain, fold, 35);
////            aveKNNDTWI37 = runKNNDTWI(aveKNNDTWI37, multiTest, multiTrain, fold, 37);
////            aveKNNDTWI39 = runKNNDTWI(aveKNNDTWI39, multiTest, multiTrain, fold, 39);
////            aveKNNDTWI41 = runKNNDTWI(aveKNNDTWI41, multiTest, multiTrain, fold, 41);
////            aveKNNDTWI43 = runKNNDTWI(aveKNNDTWI43, multiTest, multiTrain, fold, 43);
////            aveKNNDTWI45 = runKNNDTWI(aveKNNDTWI45, multiTest, multiTrain, fold, 45);
////            aveKNNDTWI47 = runKNNDTWI(aveKNNDTWI47, multiTest, multiTrain, fold, 47);
////            aveKNNDTWI49 = runKNNDTWI(aveKNNDTWI49, multiTest, multiTrain, fold, 49);
//
//
//            
//
//            
//            
        }
//
//        System.out.println("Euclidean Distance Accuracy: " + aveEuclid[0]/totalFolds);
//        System.out.println("Euclidean Distance Sport Accuracy: " + aveEuclid[1]/totalFolds);
//        System.out.println("Average Bayes: " + aveBayes/totalFolds);
//        System.out.println("");
//        System.out.println("Average KNN: " + aveKNN/totalFolds);
//        System.out.println("Average J48: " + aveJ48/totalFolds);
//        System.out.println("");
//        System.out.println("Rotational Forest Accuracy: " + aveRotation[0]/totalFolds);
//        System.out.println("Rotational Forest Sport Accuracy: " + aveRotation[1]/totalFolds);
//        System.out.println("");
//        System.out.println("Random Forest Accuracy: " + aveRF[0]/totalFolds);
//        System.out.println("Random Forest Sport Accuracy: " + aveRF[1]/totalFolds);
//        System.out.println("");
//        System.out.println("SMO Accuracy: " + aveSMO[0]/totalFolds);
//        System.out.println("SMO Sport Accuracy: " + aveSMO[1]/totalFolds);
//        System.out.println("");
//        System.out.println("ANN Accuracy: " + aveANN[0]/totalFolds);
//        System.out.println("ANN Sport Accuracy: " + aveANN[1]/totalFolds);
//        System.out.println("");
//        
//        System.out.println("Average KNN3:  " + aveKNN3/totalFolds);
//        System.out.println("Average KNN5:  " + aveKNN5/totalFolds);
//        System.out.println("Average KNN7:  " + aveKNN7/totalFolds);
//        System.out.println("Average KNN9:  " + aveKNN9/totalFolds);
//        System.out.println("Average KNN11: " + aveKNN11/totalFolds);
//        System.out.println("Average KNN13: " + aveKNN13/totalFolds);
//        System.out.println("Average KNN15: " + aveKNN15/totalFolds);
//        System.out.println("Average KNN17: " + aveKNN17/totalFolds);
//        System.out.println("Average KNN19: " + aveKNN19/totalFolds);
//        System.out.println("Average KNN21: " + aveKNN21/totalFolds);
//        System.out.println("Average KNN23: " + aveKNN23/totalFolds);
//        System.out.println("Average KNN25: " + aveKNN25/totalFolds);
//        System.out.println("Average KNN27: " + aveKNN27/totalFolds);
//        System.out.println("Average KNN29: " + aveKNN29/totalFolds);
//        System.out.println("Average KNN31: " + aveKNN31/totalFolds);
//        System.out.println("Average KNN33: " + aveKNN33/totalFolds);
//        System.out.println("Average KNN35: " + aveKNN35/totalFolds);
//        System.out.println("Average KNN37: " + aveKNN37/totalFolds);
//        System.out.println("Average KNN39: " + aveKNN39/totalFolds);
//        System.out.println("Average KNN41: " + aveKNN41/totalFolds);
//        System.out.println("Average KNN43: " + aveKNN43/totalFolds);
//        System.out.println("Average KNN45: " + aveKNN45/totalFolds);
//        System.out.println("Average KNN47: " + aveKNN47/totalFolds);
//        System.out.println("Average KNN49: " + aveKNN49/totalFolds);
//        
//        System.out.println("Random Forest 10 Accuracy:  " + aveRF10[0]/totalFolds);
//        System.out.println("Random Forest 10 Sport Accuracy:  " + aveRF10[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 50 Accuracy:  " + aveRF50[0]/totalFolds);
//        System.out.println("Random Forest 50 Sport Accuracy:  " + aveRF50[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 100 Accuracy: " + aveRF100[0]/totalFolds);
//        System.out.println("Random Forest 100 Sport Accuracy: " + aveRF100[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 150 Accuracy: " + aveRF150[0]/totalFolds);
//        System.out.println("Random Forest 150 Sport Accuracy: " + aveRF150[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 200 Accuracy: " + aveRF200[0]/totalFolds);
//        System.out.println("Random Forest 200 Sport Accuracy: " + aveRF200[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 250 Accuracy: " + aveRF250[0]/totalFolds);
//        System.out.println("Random Forest 250 Sport Accuracy: " + aveRF250[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 300 Accuracy: " + aveRF300[0]/totalFolds);
//        System.out.println("Random Forest 300 Sport Accuracy: " + aveRF300[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 350 Accuracy: " + aveRF350[0]/totalFolds);
//        System.out.println("Random Forest 350 Sport Accuracy: " + aveRF350[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 400 Accuracy: " + aveRF400[0]/totalFolds);
//        System.out.println("Random Forest 400 Sport Accuracy: " + aveRF400[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 450 Accuracy: " + aveRF450[0]/totalFolds);
//        System.out.println("Random Forest 450 Sport Accuracy: " + aveRF450[1]/totalFolds);System.out.println("");
//        System.out.println("Random Forest 500 Accuracy: " + aveRF500[0]/totalFolds);
//        System.out.println("Random Forest 500 Sport Accuracy: " + aveRF500[1]/totalFolds);System.out.println("");
//        
////        System.out.println("Rotational Forest 10 Accuracy: " + aveRotation10[0]/totalFolds);
////        System.out.println("Rotational Forest 10 Sport Accuracy: " + aveRotation10[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 50 Accuracy: " + aveRotation50[0]/totalFolds);
////        System.out.println("Rotational Forest 50 Sport Accuracy: " + aveRotation50[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 100 Accuracy: " + aveRotation100[0]/totalFolds);
////        System.out.println("Rotational Forest 100 Sport Accuracy: " + aveRotation100[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 150 Accuracy: " + aveRotation150[0]/totalFolds);
////        System.out.println("Rotational Forest 150 Sport Accuracy: " + aveRotation150[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 200 Accuracy: " + aveRotation200[0]/totalFolds);
////        System.out.println("Rotational Forest 200 Sport Accuracy: " + aveRotation200[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 250 Accuracy: " + aveRotation250[0]/totalFolds);
////        System.out.println("Rotational Forest 250 Sport Accuracy: " + aveRotation250[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 300 Accuracy: " + aveRotation300[0]/totalFolds);
////        System.out.println("Rotational Forest 300 Sport Accuracy: " + aveRotation300[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 350 Accuracy: " + aveRotation350[0]/totalFolds);
////        System.out.println("Rotational Forest 350 Sport Accuracy: " + aveRotation350[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 400 Accuracy: " + aveRotation400[0]/totalFolds);
////        System.out.println("Rotational Forest 400 Sport Accuracy: " + aveRotation400[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 450 Accuracy: " + aveRotation450[0]/totalFolds);
////        System.out.println("Rotational Forest 450 Sport Accuracy: " + aveRotation450[1]/totalFolds);System.out.println("");
////        System.out.println("Rotational Forest 500 Accuracy: " + aveRotation500[0]/totalFolds);
////        System.out.println("Rotational Forest 500 Sport Accuracy: " + aveRotation500[1]/totalFolds);System.out.println("");
//        
//        System.out.println("Basic DTW Accuracy: " + aveBasicDTW[0]/totalFolds);
//        System.out.println("Basic DTW Sport Accuracy: " + aveBasicDTW[1]/totalFolds);
//        System.out.println("");
//        System.out.println("DTWD Accuracy: " + aveDTWD[0]/totalFolds);
//        System.out.println("DTWD Sport Accuracy: " + aveDTWD[1]/totalFolds);
//        System.out.println("");
//        System.out.println("DTWI Accuracy: " + aveDTWI[0]/totalFolds);
//        System.out.println("DTWI Sport Accuracy: " + aveDTWI[1]/totalFolds);
//        System.out.println("");

        System.out.println("DTWA Accuracy: " + aveDTWA[0]/totalFolds);
        System.out.println("DTWA Sport Accuracy: " + aveDTWA[1]/totalFolds);
            System.out.println("");
        System.out.println("Enhanced DTWA Accuracy: " + aveEnhancedDTWA[0]/totalFolds);
        System.out.println("Enhanced DTWA Sport Accuracy: " + aveEnhancedDTWA[1]/totalFolds);
        System.out.println("Enhanced DTWA Balanced Accuracy: " + aveEnhancedDTWA[2]/totalFolds);
//        System.out.println("");
//        System.out.println("Enhanced DTWI Accuracy: " + aveEnhancedDTWI[0]/totalFolds);
//        System.out.println("Enhanced DTWI Sport Accuracy: " + aveEnhancedDTWI[1]/totalFolds);
//        System.out.println("");
//        

//        System.out.println("DTWA Accuracy: " + aveDTWA[0]/totalFolds);
//        System.out.println("DTWA Sport Accuracy: " + aveDTWA[1]/totalFolds);
//        System.out.println("");
//        System.out.println("Enhanced DTWI Accuracy: " + aveEnhancedDTWI[0]/totalFolds);
//        System.out.println("Enhanced DTWI Sport Accuracy: " + aveEnhancedDTWI[1]/totalFolds);
//        System.out.println("");
////        

//        System.out.println("KNN_DTWI 3 Accuracy: " + aveKNNDTWI3[0]/totalFolds);
//        System.out.println("KNN_DTWI 3 Sport Accuracy: " + aveKNNDTWI3[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 5 Accuracy: " + aveKNNDTWI5[0]/totalFolds);
//        System.out.println("KNN_DTWI 5 Sport Accuracy: " + aveKNNDTWI5[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 7 Accuracy: " + aveKNNDTWI7[0]/totalFolds);
//        System.out.println("KNN_DTWI 7 Sport Accuracy: " + aveKNNDTWI7[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 9 Accuracy: " + aveKNNDTWI9[0]/totalFolds);
//        System.out.println("KNN_DTWI 9 Sport Accuracy: " + aveKNNDTWI9[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 11 Accuracy: " + aveKNNDTWI11[0]/totalFolds);
//        System.out.println("KNN_DTWI 11 Sport Accuracy: " + aveKNNDTWI11[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 13 Accuracy: " + aveKNNDTWI13[0]/totalFolds);
//        System.out.println("KNN_DTWI 13 Sport Accuracy: " + aveKNNDTWI13[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 15 Accuracy: " + aveKNNDTWI15[0]/totalFolds);
//        System.out.println("KNN_DTWI 15 Sport Accuracy: " + aveKNNDTWI15[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 17 Accuracy: " + aveKNNDTWI17[0]/totalFolds);
//        System.out.println("KNN_DTWI 17 Sport Accuracy: " + aveKNNDTWI17[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 19 Accuracy: " + aveKNNDTWI19[0]/totalFolds);
//        System.out.println("KNN_DTWI 19 Sport Accuracy: " + aveKNNDTWI19[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 21 Accuracy: " + aveKNNDTWI21[0]/totalFolds);
//        System.out.println("KNN_DTWI 21 Sport Accuracy: " + aveKNNDTWI21[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 23 Accuracy: " + aveKNNDTWI23[0]/totalFolds);
//        System.out.println("KNN_DTWI 23 Sport Accuracy: " + aveKNNDTWI23[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 25 Accuracy: " + aveKNNDTWI25[0]/totalFolds);
//        System.out.println("KNN_DTWI 25 Sport Accuracy: " + aveKNNDTWI25[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 27 Accuracy: " + aveKNNDTWI27[0]/totalFolds);
//        System.out.println("KNN_DTWI 27 Sport Accuracy: " + aveKNNDTWI27[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 29 Accuracy: " + aveKNNDTWI29[0]/totalFolds);
//        System.out.println("KNN_DTWI 29 Sport Accuracy: " + aveKNNDTWI29[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 31 Accuracy: " + aveKNNDTWI31[0]/totalFolds);
//        System.out.println("KNN_DTWI 31 Sport Accuracy: " + aveKNNDTWI31[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 33 Accuracy: " + aveKNNDTWI33[0]/totalFolds);
//        System.out.println("KNN_DTWI 33 Sport Accuracy: " + aveKNNDTWI33[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 35 Accuracy: " + aveKNNDTWI35[0]/totalFolds);
//        System.out.println("KNN_DTWI 35 Sport Accuracy: " + aveKNNDTWI35[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 37 Accuracy: " + aveKNNDTWI37[0]/totalFolds);
//        System.out.println("KNN_DTWI 37 Sport Accuracy: " + aveKNNDTWI37[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 39 Accuracy: " + aveKNNDTWI39[0]/totalFolds);
//        System.out.println("KNN_DTWI 39 Sport Accuracy: " + aveKNNDTWI39[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 41 Accuracy: " + aveKNNDTWI41[0]/totalFolds);
//        System.out.println("KNN_DTWI 41 Sport Accuracy: " + aveKNNDTWI41[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 43 Accuracy: " + aveKNNDTWI43[0]/totalFolds);
//        System.out.println("KNN_DTWI 43 Sport Accuracy: " + aveKNNDTWI43[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 45 Accuracy: " + aveKNNDTWI45[0]/totalFolds);
//        System.out.println("KNN_DTWI 45 Sport Accuracy: " + aveKNNDTWI45[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 47 Accuracy: " + aveKNNDTWI47[0]/totalFolds);
//        System.out.println("KNN_DTWI 47 Sport Accuracy: " + aveKNNDTWI47[1]/totalFolds);
//        System.out.println("");
//        System.out.println("KNN_DTWI 49 Accuracy: " + aveKNNDTWI49[0]/totalFolds);
//        System.out.println("KNN_DTWI 49 Sport Accuracy: " + aveKNNDTWI49[1]/totalFolds);
//
//        
    }
    
    public static double [] runKNNDTWI(double aveDTWI[], Instances test, Instances train, int fold, int k) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new KNN_DTWI(k), test, train);
        dtw.confusionMatrix();
        dtw.writeCsvFile("KNNDTWITest" + fold, "KNNDTWI" + k);
        aveDTWI[0] += dtw.getAccuracy();
        aveDTWI[1] += dtw.getSportAccuracy();
        return aveDTWI;
    }
    
    public static double [] runEnhancedDTWI(double aveDTWI[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new Enhanced_DTWI(), test, train);
        dtw.confusionMatrix();
        dtw.writeCsvFile("EnhancedDTWITest" + fold, "EnhancedDTWI");
        aveDTWI[0] += dtw.getAccuracy();
        aveDTWI[1] += dtw.getSportAccuracy();
        return aveDTWI;
    }
    
    public static double [] runEnhancedDTWI(double aveDTWI[], Instances test, Instances train, int fold, int warp_Size) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new Enhanced_DTWI(warp_Size), test, train);
        dtw.confusionMatrix();
        dtw.writeCsvFile("EnhancedDTWITest" + fold, "EnhancedDTWI");
        aveDTWI[0] += dtw.getAccuracy();
        aveDTWI[1] += dtw.getSportAccuracy();
        return aveDTWI;
    }
    
    public static double [] runEnhancedDTWA(double aveDTWA[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new DTWA(), test, train);
        dtw.confusionMatrix();
        //dtw.writeCsvFile("DTWATest" + fold, "EnhancedDTWA");
        aveDTWA[0] += dtw.getAccuracy();
        aveDTWA[1] += dtw.getSportAccuracy();
        aveDTWA[2] += dtw.getBalancedAccuracy();
        return aveDTWA;
    }
    
    public static double [] runDTWA(double aveDTWA[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new Basic_DTWA(), test, train);
        dtw.confusionMatrix();
        dtw.writeCsvFile("DTWATest" + fold, "DTWA");
        aveDTWA[0] += dtw.getAccuracy();
        aveDTWA[1] += dtw.getSportAccuracy();
        return aveDTWA;
    }
    
    public static double [] runDTWD(double aveDTWD[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new DTWD(), test, train);
        dtw.confusionMatrix();
        dtw.writeCsvFile("DTWDTest" + fold, "DTWD");
        aveDTWD[0] += dtw.getAccuracy();
        aveDTWD[1] += dtw.getSportAccuracy();
        
        return aveDTWD;
    }
    
    public static double [] runDTWI(double aveDTWI[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new DTWI(), test, train);
        dtw.confusionMatrix();
        dtw.writeCsvFile("DTWITest" + fold, "DTWI");
        aveDTWI[0] += dtw.getAccuracy();
        aveDTWI[1] += dtw.getSportAccuracy();
        return aveDTWI;
    }
   
    
    public static double [] runBasicDTW(double aveBasicDTW[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new Basic_DTW(), test, train);
        dtw.confusionMatrix();
        dtw.writeCsvFile("basicDTWTest" + fold, "BASICDTW");
        aveBasicDTW[0] += dtw.getAccuracy();
        aveBasicDTW[1] += dtw.getSportAccuracy();
        return aveBasicDTW;
    }
    
    public static double [] runEuclideanDistance(double aveEuclid[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper euclidean = new ClassifierWrapper(new EuclideanDistance(), test, train);
        euclidean.confusionMatrix();
        euclidean.writeCsvFile("euclideanTest" + fold, "EUCLIDEANDISTANCE");
        aveEuclid[0] += euclidean.getAccuracy();
        aveEuclid[1] += euclidean.getSportAccuracy();
        return aveEuclid;
    }
    
    public static double [] runRotationForest(double aveRotation[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper rotation = new ClassifierWrapper(new RotationForest(), test, train);
        rotation.confusionMatrix();
        rotation.writeCsvFile("rotationTest" + fold, "ROTATIONFOREST");
        aveRotation[0] += rotation.getAccuracy();
        aveRotation[1] += rotation.getSportAccuracy();
        return aveRotation;
    }
    
        public static double [] runRotationForest(double aveRotation[], Instances test, Instances train, int fold, int numTrees) throws Exception{
        weka.classifiers.meta.RotationForest rf = new weka.classifiers.meta.RotationForest();
        rf.setNumIterations(numTrees);
        ClassifierWrapper rotation = new ClassifierWrapper(rf, test, train);
        rotation.confusionMatrix();
        rotation.writeCsvFile("rotationTest" + fold, "ROTATIONFOREST" + numTrees);
        aveRotation[0] += rotation.getAccuracy();
        aveRotation[1] += rotation.getSportAccuracy();
        return aveRotation;
    }
        public static double [] runRandomForest(double aveRF[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper randomForest = new ClassifierWrapper(new weka.classifiers.trees.RandomForest(),test,train);
        randomForest.confusionMatrix();
        randomForest.writeCsvFile("randomForestTest" + fold, "RANDOMFOREST");
        aveRF[0] += randomForest.getAccuracy();
        aveRF[1] += randomForest.getSportAccuracy();
        return aveRF;
    }
    
    public static double [] runRandomForest(double aveRF[], Instances test, Instances train, int fold, int numTrees) throws Exception{
        weka.classifiers.trees.RandomForest rf = new weka.classifiers.trees.RandomForest();
        rf.setNumTrees(numTrees);
        ClassifierWrapper randomForest = new ClassifierWrapper(rf,test,train);
        randomForest.confusionMatrix();
        randomForest.writeCsvFile("randomForestTest" + fold, "RANDOMFOREST" + numTrees);
        aveRF[0] += randomForest.getAccuracy();
        aveRF[1] += randomForest.getSportAccuracy();
        return aveRF;
    }
    
    public static double [] runSMO(double aveSVM[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper svm = new ClassifierWrapper(new weka.classifiers.functions.SMO(), test, train);
        svm.confusionMatrix();
        svm.writeCsvFile("SVMTest" + fold, "SVM");
        aveSVM[0] += svm.getAccuracy();
        aveSVM[1] += svm.getSportAccuracy();
        return aveSVM;
    }
    
     public static double [] runANN(double aveANN[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper ann = new ClassifierWrapper(new weka.classifiers.functions.MultilayerPerceptron(), test, train);
        ann.confusionMatrix();
        ann.writeCsvFile("ANNTest" + fold, "ANN");
        aveANN[0] += ann.getAccuracy();
        aveANN[1] += ann.getSportAccuracy();
        return aveANN;
    }
     
    public static double runBayes(double aveBayes, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper bayes = new ClassifierWrapper(new weka.classifiers.bayes.NaiveBayes(), test, train);
        bayes.confusionMatrix();
        bayes.writeCsvFile("bayesTest" + fold, "BAYES");
        aveBayes += bayes.getAccuracy();
        return aveBayes;
    }
    
    public static double runKNN(double aveKNN, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper knn = new ClassifierWrapper(new weka.classifiers.lazy.IBk(),test,train);
        knn.confusionMatrix();
        knn.writeCsvFile("knnTest" + fold, "KNN");
        aveKNN += knn.getAccuracy();
        return aveKNN;
    }
    
    public static double runKNN(double aveKNN, Instances test, Instances train, int fold, int k) throws Exception{
        ClassifierWrapper knn = new ClassifierWrapper(new weka.classifiers.lazy.IBk(k),test,train);
        knn.confusionMatrix();
        knn.writeCsvFile("knnTest" + fold, "KNN" + k);
        aveKNN += knn.getAccuracy();
        return aveKNN;
    }
    
    public static double runJ48(double aveJ48, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper J48 = new ClassifierWrapper(new weka.classifiers.trees.J48(),test,train);
        J48.confusionMatrix();
        J48.writeCsvFile("J48Test" + fold, "J48");
        aveJ48 += J48.getAccuracy();
        return aveJ48;
    }
    
    
    
    public static Instances loadData(String filePath){
        String dataLocation=filePath;
        Instances i = null;
        try{
            FileReader reader = new FileReader(dataLocation);
            i = new Instances(reader);
            i.setClassIndex(i.numAttributes()-1);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        return i;
    }
    
    public static void contingencyTable(Classifier tr, Instances test) throws Exception{
        int classes = test.numClasses();
        int[][] matrix = new int [classes][classes];
        for(Instance i: test){
            matrix[(int)tr.classifyInstance(i)][(int)i.classValue()] += 1;
       }
        System.out.println("\t\t\t ACTUAL");
        System.out.print("\t\t\t");
        for(int i=0; i<classes; i++){
            System.out.print("[" + i + "]\t");
        }
        System.out.println("");
        System.out.println("Predicted");
        for(int i=0; i<classes; i++){
            
            System.out.print("\t\t[" + i + "]\t");
            for(int j=0; j<classes; j++){
                
                System.out.print(matrix[i][j] + "\t");
            }
            System.out.println("");
        }
        
        
    }
}
