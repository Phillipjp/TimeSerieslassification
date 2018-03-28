/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class KNN_DTWI extends Enhanced_DTWI{
    
    private int k;
    
    public KNN_DTWI(int k){
        super();
        this.k = k;
    }
    
    public KNN_DTWI(int k, int warp_size){
        super(warp_size);
        this.k = k;
    }
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainingInstances = data;
        Instance ins= data.instance(0);
        Instances split=ins.relationalValue(0);
        noClasses = data.numClasses();
        noAttributes = split.numAttributes();
        noInstances = data.numInstances();
        trainingData = makeDataMultiVariate(data, noInstances);
        
        setWarpSize();
        
    }
    
    private void setWarpSize(){
        
        Instances [] data = InstanceTools.resampleInstances(trainingInstances, 1, .5);
        double [][][] cvTrain = makeDataMultiVariate(data[0], data[0].numInstances());
        double [][][] cvTest = makeDataMultiVariate(data[1], data[1].numInstances());
        
        int maxWarpSize = cvTrain[0][0].length;
        int correct [] = new int [maxWarpSize];
        
        //for all possible values of the warping window size
        for(int i=0; i<maxWarpSize; i++){
            this.warp_size = i;
            
            //for all test data
            for(int test=0; test<cvTest.length; test++){
                DistanceAndClass [] dc = new DistanceAndClass [cvTrain.length];
                double temp =0;
                double minClass = 0;
                double minValue = Double.MAX_VALUE;
                //for all train data
                for(int train=0; train<cvTrain.length; train++){
                    //for each series in the train data
                    for(int series=0; series < cvTrain[train].length; series++ ){
                        temp+= distance(cvTrain[train][series], cvTest[test][series]);
                    }
                    dc[train] = new DistanceAndClass(temp, data[0].instance(train).classValue());
                    temp = 0;
                }
                minClass = nearestNeigbough(dc);
                if(minClass == data[1].instance(test).classValue()){
                    correct[i]++;
                }
                
            } 
        }
        
        //find what warp window size gave the most correct results
        int warp = 0;
        int biggest = correct[0];
//        for(int i=0;i<correct.length; i++){
//            System.out.print(i + "\t");
//        }
//         System.out.println("");
        for(int i=0; i<correct.length; i++){
            //System.out.print(correct[i] + "\t");
            if(biggest < correct[i]){
                biggest = correct[i];
                warp = i;
            }
        }
        //System.out.println("");
        //System.out.println(warp);
        this.warp_size =  warp;
    }
    
    //A nested class that stores the distance of a training instance from a test
    //instance and the class of the training instance
    public static class DistanceAndClass{
            public double distance;
            public double c;
            public DistanceAndClass (double distance, double c){
                this.distance = distance;
                this.c = c;
            } 
            
            //A method that returns the smaller distance
            public static class CompareAscending implements Comparator<DistanceAndClass>{
                @Override
                public int compare(DistanceAndClass a, DistanceAndClass b){
                    return (int)(a.distance-b.distance);     
                }
            }
    
        
    }
    
    //Sorts an array of DissanceAndClass objects in assending order accordinf to distance
    public static DistanceAndClass[] sortAscending(DistanceAndClass[] items) {
            Comparator compAsc = new DistanceAndClass.CompareAscending();
            Arrays.sort(items, compAsc);
            return items;
    }
    
    private double nearestNeigbough(DistanceAndClass [] dc){
        //sort dc according to distance
        dc = sortAscending(dc);
        //if k equals 1 return the class of the shorest distance
        if(k ==1 ){
            return dc[0].c;
        }
        double [] weights = new double [4];
        for (int i = 0; i < k; i++) {
            weights[(int)dc[i].c] += 1/(1+dc[i].distance);
        }
        
        double biggestWeight = -1;
        double predicted = -1;
        
        for (int i = 0; i < weights.length; i++) {
            if(weights[i]>biggestWeight){
                biggestWeight = weights[i];
                predicted = (double)i;
            }
        }
        return predicted;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        //put the test instance in an array
        Instances testSplit = instance.relationalValue(0);
        double[][] test = new double[testSplit.numInstances()][];
        for(int j=0;j<testSplit.numInstances();j++){
            test[j]=testSplit.instance(j).toDoubleArray();
        }
        //create an array containing the distance and class for each train instance
        DistanceAndClass [] dc = new DistanceAndClass [noInstances];
        
        //transpose array
       test = transposeArray(test);
        double distance = 0;
        //for all training instances find the distance
        for(int i=0; i<trainingInstances.numInstances(); i++){
            //for each series in the test data get the distance
            for(int j=0 ;j<trainingData[i].length; j++){
                distance += distance(trainingData[i][j], test[j]);
            }
            
            dc[i] = new DistanceAndClass(distance, trainingInstances.instance(i).classValue());
            distance = 0;
        }
        return nearestNeigbough(dc);
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double [] prob = new double [trainingInstances.numClasses()];
        //put the test instance in an array
        Instances testSplit = instance.relationalValue(0);
        double[][] test = new double[testSplit.numInstances()][];
        for(int j=0;j<testSplit.numInstances();j++){
            test[j]=testSplit.instance(j).toDoubleArray();
        }
        //create an array containing the distance and class for each train instance
        DistanceAndClass [] dc = new DistanceAndClass [noInstances];
        
        //transpose array
       test = transposeArray(test);
        double distance = 0;
        //for all training instsnces find the distance
        for(int i=0; i<trainingInstances.numInstances(); i++){
            //for each series in the test data get the distance
            for(int j=0 ;j<trainingData[i].length; j++){
                distance += distance(trainingData[i][j], test[j]);
            }
            
            dc[i] = new DistanceAndClass(distance, trainingInstances.instance(i).classValue());
            distance = 0;
        }
        
        //sort dc according to distance
        dc = sortAscending(dc);
        
        //find the number of predictions for each class of the smallest k predictions
        int [] predictions = new int [trainingInstances.numClasses()];
        for (int i = 0; i < k; i++) {
            predictions[(int)dc[i].c] ++;
        }
        
        for (int i = 0; i < prob.length; i++) {
            prob[i] = (double)predictions[i]/k;
        }
        return prob;
    }
}
