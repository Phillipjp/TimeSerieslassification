/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import java.util.Arrays;
import java.util.Comparator;
import utilities.ClassifierTools;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
/**
 *
 * @author phillipperks
 */
public class EuclideanDistance implements Classifier {
    double [][] trainingData;
    int noClasses;
    int noAttributes;
    int noInstances;
    Instances trainingInstances;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainingInstances = data;
        noClasses = data.numClasses();
        noAttributes = data.numAttributes();
        noInstances = data.numInstances();
        trainingData = new double [noInstances][noAttributes];
        for(int i=0; i<noInstances; i++){
            for(int a=0; a<noAttributes; a++){
                trainingData[i][a] = data.instance(i).value(a);
            }
        }
        
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
        //calculate all euclidean distances between the test intance and all
        //trainign instances
        double [] distances = new double [noInstances];
        for(int i=0; i<distances.length; i++){
            for(int a=0; a<noAttributes-1; a++){
                distances[i] += Math.pow(trainingData[i][a] - instance.value(a),2);
            }
        }
        //create an array containing the distance and class for each instance
        DistanceAndClass [] dc = new DistanceAndClass [noInstances];
        for(int i=0; i< noInstances; i++){
            dc[i] = new DistanceAndClass(distances[i], trainingInstances.instance(i).value(noAttributes-1));
        }
        
        //sort dc according to distance
        dc = sortAscending(dc);
        
        //return the class of the instance with the shortest distance from the
        //test instance
        return dc[0].c;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double prob [] = new double [noClasses];
        
        //Array that holds the number of times each class occurs
        int[] count = new int [noClasses];
        //Matrix that holds the mean value for each attribute for each class
        double [][] means = new double[noClasses][noAttributes];
        int classValue;
        //Calculate the means
        for(Instance i : trainingInstances){
            classValue = (int)i.classValue();
            count[classValue] ++;
            for(int j=0; j<noAttributes-1; j++){
                means[classValue][j] += i.value(j);
            }
        }
        for(int j=0; j<noClasses; j++){
            for(int k=0; k<noAttributes-1; k++){
                means[j][k]= means[j][k]/count[j];
            }
        }
        //Matrix that holds the standard deviation for each attribute for each class
        double [][] stdev = new double [noClasses][noAttributes];
        //calvulate standard deviations
        for(Instance i: trainingInstances){
            classValue = (int)i.classValue();
            for(int j=0; j<noAttributes-1; j++){
                stdev[classValue][j] += Math.pow(i.value(j)-means[classValue][j],2);
            }
        }
        for(int j=0; j<noClasses; j++){
            for(int k=0; k<noAttributes-1; k++){
                stdev[j][k]= Math.sqrt(stdev[j][k]);
            }
        }
        
        for(int i=0; i<noClasses; i++){
            prob[i]=1;
            for(int j=0; j<noAttributes-1; j++){
               prob[i]*=NormalDistribution.probability(instance.value(j), means[i][j],stdev[i][j]);
            }
        }
        return prob;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    public static void main(String[] args) throws Exception{
    }
}
