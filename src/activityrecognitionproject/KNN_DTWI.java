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
        //if k equals 1 return the class of the shorest distance
        if(k ==1 ){
            return dc[0].c;
        }
        //find the number of predictions for each class of the smallest k predictions
        int [] predictions = new int [4];
        for (int i = 0; i < k; i++) {
            predictions[(int)dc[i].c] ++;
        }
        
        //find the most commonly predicted class and check if there's a tie
        int highest = -1;
        HashSet classes = new HashSet();
        for (int i = 0; i < predictions.length; i++) {
            if(predictions[i] > highest){
                 classes.clear();
                 highest = predictions[i];
                 classes.add(i);
            }
            else{
                for (int j = 0; j < predictions.length; j++) {
                    if (i != j) {
                        if(predictions[i] == predictions[j]){
                            if(predictions[i] == highest){
                                classes.add(i);
                                highest = predictions[i];
                            }

                        }
                    }
                }
            }
        }
        ArrayList<Integer> classesList = new ArrayList<>(classes);
        //if there's no tie for the most popular class return predicted class
        if(classes.size()==1){
            return (double)classesList.get(0);
        }
        //else if there's a tie return the class with the instance that has the
        //shortest distance
        else{
            for(int i = 0; i < dc.length; i++) {
                for (int j = 0; j < classesList.size(); j++) {
                    if(dc[i].c == (double)classesList.get(j)){
                        return dc[i].c;
                    }
                }
                
            }
        }
        //should never happen
        return -1;
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
        
        for (int i = 0; i < k; i++) {
            prob[i] = predictions[i]/k;
        }
        
        return prob;
    }
}
