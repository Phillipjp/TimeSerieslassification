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
        
        Instances all = ClassifierTools.loadData("/Users/phillipperks/Desktop/3rd Year Project/ARFF_Files/Cross Validation/Combination/MVMotionUni.arff");
        
        int totalFolds = 30;
        
        double aveBayes = 0;
        double aveKNN =  0;
        double aveJ48 = 0;
        double aveRF = 0;
        double aveEuclid = 0;
        double aveRotation = 0;
        
        for(int fold=0; fold<totalFolds; fold++){
            Instances[] data = InstanceTools.resampleInstances(all, fold, .5);

            Instances train = data[0];
            Instances test = data[1];
            
            aveEuclid = runEuclideanDistance(aveEuclid,test, train, fold);
            aveBayes = runBayes(aveBayes,test, train, fold);
            aveKNN = runKNN(aveKNN,test, train, fold, 3 , "KNN3");
            aveJ48 = runJ48(aveJ48,test, train, fold);
            aveRotation = runRotationalForest(aveRotation,test, train, fold);
            aveRF = runRandomForest(aveRF,test, train, fold, 100, "RANDOMFOREST100");

            
            
        }

        System.out.println("Average Euclidean: " + aveEuclid/totalFolds);
        System.out.println("Average Bayes: " + aveBayes/totalFolds);
        System.out.println("Average KNN: " + aveKNN/totalFolds);
        System.out.println("Average J48: " + aveJ48/totalFolds);
        System.out.println("Average Rotational Forest: " + aveRotation/totalFolds);
        System.out.println("Average Random Forest: " + aveRF/totalFolds);
        
        
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
