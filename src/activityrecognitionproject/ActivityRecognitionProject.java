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
        double aveRF [] = new double [2];
        double aveEuclid[] = new double [2];
        double aveRotation [] = new double [2];
        double aveSMO [] = new double [2];
        double aveANN [] = new double [2];
        double aveBasicDTW [] = new double [2];
        double aveDTWI [] = new double [2];
        double aveDTWD [] = new double [2];
        double aveEnhancedDTWI [] = new double [2];
        double aveKNNDTWI3 [] = new double [2];
        double aveKNNDTWI5 [] = new double [2];
        double aveKNNDTWI7 [] = new double [2];
        
        
        
        for(int fold=0; fold<totalFolds; fold++){
            Instances[] data = InstanceTools.resampleInstances(all, fold, .5);

            Instances train = data[0];
            Instances test = data[1];
            
            Instances[] multiData = InstanceTools.resampleInstances(multiVariate, fold, .5);
            Instances multiTrain = multiData[0];
            Instances  multiTest = multiData[1];
            
 //          aveEuclid = runEuclideanDistance(aveEuclid,test, train, fold);
//            aveBayes = runBayes(aveBayes,test, train, fold);
//            aveKNN = runKNN(aveKNN,test, train, fold, 3 , "KNN3");
//            aveJ48 = runJ48(aveJ48,test, train, fold);
//            aveRotation = runRotationalForest(aveRotation,test, train, fold);
//            aveRF = runRandomForest(aveRF,test, train, fold, 100, "RANDOMFOREST100");
//            aveBasicDTW = runBasicDTW(aveBasicDTW,test, train, fold);
            //aveDTWI = runDTWI(aveDTWI, multiTest, multiTrain, fold);
  //          aveDTWD = runDTWD(aveDTWD, multiTest, multiTrain, fold);
            aveEnhancedDTWI = runEnhancedDTWI(aveEnhancedDTWI, multiTest, multiTrain, fold);
//            aveKNNDTWI3 = runKNNDTWI(aveKNNDTWI3, multiTest, multiTrain, fold, 3);
//            aveKNNDTWI5 = runKNNDTWI(aveKNNDTWI5, multiTest, multiTrain, fold, 5);
//            aveKNNDTWI7 = runKNNDTWI(aveKNNDTWI7, multiTest, multiTrain, fold, 7);
//           aveSMO = runSMO(aveSMO, multiTest, multiTrain, fold);
//            aveANN = runANN(aveANN, test, train, fold);
            

            
            
        }

        System.out.println("Euclidean Distance Accuracy: " + aveEuclid[0]/totalFolds);
        System.out.println("Euclidean Distance Sport Accuracy: " + aveEuclid[1]/totalFolds);
        System.out.println("Average Bayes: " + aveBayes/totalFolds);
        System.out.println("");
        System.out.println("Average KNN: " + aveKNN/totalFolds);
        System.out.println("Average J48: " + aveJ48/totalFolds);
        System.out.println("");
        System.out.println("Rotational Forest Accuracy: " + aveRotation[0]/totalFolds);
        System.out.println("Rotational Forest Sport Accuracy: " + aveRotation[1]/totalFolds);
        System.out.println("");
        System.out.println("Random Forest Accuracy: " + aveRF[0]/totalFolds);
        System.out.println("Random Forest Sport Accuracy: " + aveRF[1]/totalFolds);
        System.out.println("");
        System.out.println("SMO Accuracy: " + aveSMO[0]/totalFolds);
        System.out.println("SMO Sport Accuracy: " + aveSMO[1]/totalFolds);
        System.out.println("");
        System.out.println("ANN Accuracy: " + aveANN[0]/totalFolds);
        System.out.println("ANN Sport Accuracy: " + aveANN[1]/totalFolds);
        System.out.println("");
        System.out.println("Basic DTW Accuracy: " + aveBasicDTW[0]/totalFolds);
        System.out.println("Basic DTW Sport Accuracy: " + aveBasicDTW[1]/totalFolds);
        System.out.println("");
        System.out.println("DTWD Accuracy: " + aveDTWD[0]/totalFolds);
        System.out.println("DTWD Sport Accuracy: " + aveDTWD[1]/totalFolds);
        System.out.println("");
        System.out.println("DTWI Accuracy: " + aveDTWI[0]/totalFolds);
        System.out.println("DTWI Sport Accuracy: " + aveDTWI[1]/totalFolds);
        System.out.println("");
        System.out.println("Enhanced DTWI Accuracy: " + aveEnhancedDTWI[0]/totalFolds);
        System.out.println("Enhanced DTWI Sport Accuracy: " + aveEnhancedDTWI[1]/totalFolds);
        System.out.println("");
        System.out.println("KNN_DTWI 3 Accuracy: " + aveKNNDTWI3[0]/totalFolds);
        System.out.println("KNN_DTWI 3 Sport Accuracy: " + aveKNNDTWI3[1]/totalFolds);
        System.out.println("");
        System.out.println("KNN_DTWI 5 Accuracy: " + aveKNNDTWI5[0]/totalFolds);
        System.out.println("KNN_DTWI 5 Sport Accuracy: " + aveKNNDTWI5[1]/totalFolds);
        System.out.println("");
        System.out.println("KNN_DTWI 7 Accuracy: " + aveKNNDTWI7[0]/totalFolds);
        System.out.println("KNN_DTWI 7 Sport Accuracy: " + aveKNNDTWI7[1]/totalFolds);
//        for (int i = 0; i < 32; i++) {
//            System.out.println("Average Enhanced DTWI " + i + " : " + aveEnhancedDTWI[i]/totalFolds);
//        }
        
    }
    public static double [] runKNNDTWI(double aveDTWI[], Instances test, Instances train, int fold, int k) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new KNN_DTWI(k), test, train);
        dtw.confusionMatrix();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
        aveDTWI[0] += dtw.getAccuracy();
        aveDTWI[1] += dtw.getSportAccuracy();
        return aveDTWI;
    }
    
    public static double [] runDTWD(double aveDTWD[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new DTWD(), test, train);
        dtw.confusionMatrix();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
        aveDTWD[0] += dtw.getAccuracy();
        aveDTWD[1] += dtw.getSportAccuracy();
        
        return aveDTWD;
    }
    
    public static double [] runDTWI(double aveDTWI[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper dtw = new ClassifierWrapper(new DTWI(), test, train);
        dtw.confusionMatrix();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
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
    
    public static double [] runRotationalForest(double aveRotation[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper rotation = new ClassifierWrapper(new RotationForest(), test, train);
        rotation.confusionMatrix();
        rotation.writeCsvFile("rotationTest" + fold, "ROTATIONALFOREST");
        aveRotation[0] += rotation.getAccuracy();
        aveRotation[1] += rotation.getSportAccuracy();
        return aveRotation;
    }
    
    public static double [] runRandomForest(double aveRF[], Instances test, Instances train, int fold, int numTrees, String testName) throws Exception{
        weka.classifiers.trees.RandomForest rf = new weka.classifiers.trees.RandomForest();
        rf.setNumTrees(numTrees);
        ClassifierWrapper randomForest = new ClassifierWrapper(rf,test,train);
        randomForest.confusionMatrix();
        randomForest.writeCsvFile("randomForestTest" + fold, "RANDOMFOREST/" + testName);
        aveRF[0] += randomForest.getAccuracy();
        aveRF[1] += randomForest.getSportAccuracy();
        return aveRF;
    }
    
    public static double [] runSMO(double aveSVM[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper svm = new ClassifierWrapper(new weka.classifiers.functions.SMO(), test, train);
        svm.confusionMatrix();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
        aveSVM[0] += svm.getAccuracy();
        aveSVM[1] += svm.getSportAccuracy();
        return aveSVM;
    }
    
     public static double [] runANN(double aveANN[], Instances test, Instances train, int fold) throws Exception{
        ClassifierWrapper ann = new ClassifierWrapper(new weka.classifiers.functions.SMO(), test, train);
        ann.confusionMatrix();
        //dtw.writeCsvFile("DTWITest" + fold, "DTWI");
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
    
    public static double runKNN(double aveKNN, Instances test, Instances train, int fold, int k, String testName) throws Exception{
        ClassifierWrapper knn = new ClassifierWrapper(new weka.classifiers.lazy.IBk(k),test,train);
        knn.confusionMatrix();
        knn.writeCsvFile("knnTest" + fold, "KNN/" + testName);
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
