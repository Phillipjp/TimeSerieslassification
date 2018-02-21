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
        
        Instances all = ClassifierTools.loadData("/Users/phillipperks/Desktop/3rd-Year-Project/ARFF_Files/Cross Validation/Combination/MVMotionUni.arff");
        String path="/Users/phillipperks/Desktop/3rd-Year-Project/ARFF_Files/Cross Validation/Combination/MVMotionMulti.arff";
        Instances multiVariate = ClassifierTools.loadData(path);
        int totalFolds = 30;
        
        double aveBayes = 0;
        double aveKNN =  0;
        double aveJ48 = 0;
        double aveRF = 0;
        double aveEuclid = 0;
        double aveRotation = 0;
        double aveBasicDTW = 0;
        double aveDTWI = 0;
        double aveDTWD = 0;
        double aveEnhancedDTWI = 0;
        
        
        for(int fold=0; fold<totalFolds; fold++){
            Instances[] data = InstanceTools.resampleInstances(all, fold, .5);

            Instances train = data[0];
            Instances test = data[1];
            
            Instances[] multiData = InstanceTools.resampleInstances(multiVariate, fold, .5);
            Instances multiTrain = multiData[0];
            Instances  multiTest = multiData[1];
            
//            aveEuclid = runEuclideanDistance(aveEuclid,test, train, fold);
//            aveBayes = runBayes(aveBayes,test, train, fold);
//            aveKNN = runKNN(aveKNN,test, train, fold, 3 , "KNN3");
//            aveJ48 = runJ48(aveJ48,test, train, fold);
            aveRotation = runRotationalForest(aveRotation,test, train, fold);
            aveRF = runRandomForest(aveRF,test, train, fold, 100, "RANDOMFOREST100");
            //aveBasicDTW = runBasicDTW(aveBasicDTW,test, train, fold);
            aveDTWI = runDTWI(aveDTWI, multiTest, multiTrain, fold);
            //aveDTWD = runDTWD(aveDTWD, multiTest, multiTrain, fold);
//            for(int i=1; i<32;i++){
//                aveEnhancedDTWI[i] = runEnhancedDTWI(aveEnhancedDTWI[i], multiTest, multiTrain, fold, i);
//            }
            aveEnhancedDTWI = runEnhancedDTWI(aveEnhancedDTWI, multiTest, multiTrain, fold);

            
            
        }

        System.out.println("Average Euclidean: " + aveEuclid/totalFolds);
        System.out.println("Average Bayes: " + aveBayes/totalFolds);
        System.out.println("Average KNN: " + aveKNN/totalFolds);
        System.out.println("Average J48: " + aveJ48/totalFolds);
        System.out.println("Average Rotational Forest: " + aveRotation/totalFolds);
        System.out.println("Average Random Forest: " + aveRF/totalFolds);
        System.out.println("Average Basic DTW: " + aveBasicDTW/totalFolds);
        System.out.println("Average DTWI: " + aveDTWI/totalFolds);
        System.out.println("Average DTWD: " + aveDTWD/totalFolds);
        System.out.println("Average Enhanced DTWI: " + aveEnhancedDTWI/totalFolds);
//        for (int i = 0; i < 32; i++) {
//            System.out.println("Average Enhanced DTWI " + i + " : " + aveEnhancedDTWI[i]/totalFolds);
//        }
        
    }
    
    public static double runDTWD(double aveDTWD, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new DTWD(), test, train);
        dtw.classifyAllInstances();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
        aveDTWD += dtw.getAccuracy();
        return aveDTWD;
    }
    
    public static double runDTWI(double aveDTWI, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new DTWI(), test, train);
        dtw.classifyAllInstances();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
        aveDTWI += dtw.getAccuracy();
        return aveDTWI;
    }
    
    public static double runEnhancedDTWI(double aveDTWI, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new Enhanced_DTWI(), test, train);
        dtw.classifyAllInstances();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
        aveDTWI += dtw.getAccuracy();
        return aveDTWI;
    }
    public static double runEnhancedDTWI(double aveDTWI, Instances test, Instances train, int fold, int warp_Size) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new Enhanced_DTWI(warp_Size), test, train);
        dtw.classifyAllInstances();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
        aveDTWI += dtw.getAccuracy();
        return aveDTWI;
    }
    
    public static double runBasicDTW(double aveBasicDTW, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new Basic_DTW(), test, train);
        dtw.classifyAllInstances();
        dtw.writeCsvFile("basicDTWTest" + fold, "BASICDTW");
        aveBasicDTW += dtw.getAccuracy();
        return aveBasicDTW;
    }
    
    public static double runEuclideanDistance(double aveEuclid, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper euclidean = new ClassifierWrapper(new EuclideanDistance(), test, train);
        euclidean.classifyAllInstances();
        euclidean.writeCsvFile("euclideanTest" + fold, "EUCLIDEANDISTANCE");
        aveEuclid += euclidean.getAccuracy();
        return aveEuclid;
    }
    
    public static double runRotationalForest(double aveRotation, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper rotation = new ClassifierWrapper(new RotationForest(), test, train);
        rotation.classifyAllInstances();
        rotation.writeCsvFile("rotationTest" + fold, "ROTATIONALFOREST");
        aveRotation += rotation.getAccuracy();
        return aveRotation;
    }
    
    public static double runRandomForest(double aveRF, Instances test, Instances train, int fold, int numTrees, String testName) throws Exception{
        weka.classifiers.trees.RandomForest rf = new weka.classifiers.trees.RandomForest();
        rf.setNumTrees(numTrees);
        ClassifierWrapper randomForest = new ClassifierWrapper(rf,test,train);
        randomForest.classifyAllInstances();
        randomForest.writeCsvFile("randomForestTest" + fold, "RANDOMFOREST/" + testName);
        aveRF += randomForest.getAccuracy();
        return aveRF;
    }
    
    public static double runBayes(double aveBayes, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper bayes = new ClassifierWrapper(new weka.classifiers.bayes.NaiveBayes(), test, train);
        bayes.classifyAllInstances();
        bayes.writeCsvFile("bayesTest" + fold, "BAYES");
        aveBayes += bayes.getAccuracy();
        return aveBayes;
    }
    
    public static double runKNN(double aveKNN, Instances test, Instances train, int fold, int k, String testName) throws Exception{
        ClassifierWrapper knn = new ClassifierWrapper(new weka.classifiers.lazy.IBk(k),test,train);
        knn.classifyAllInstances();
        knn.writeCsvFile("knnTest" + fold, "KNN/" + testName);
        aveKNN += knn.getAccuracy();
        return aveKNN;
    }
    
    public static double runJ48(double aveJ48, Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper J48 = new ClassifierWrapper(new weka.classifiers.trees.J48(),test,train);
        J48.classifyAllInstances();
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
