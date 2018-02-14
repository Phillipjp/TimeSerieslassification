/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class DTWD implements Classifier{
    
    double [][][] trainingData;
    int noClasses;
    int noAttributes;
    int noInstances;
    Instances trainingInstances;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainingInstances = data;
        Instance ins= data.instance(0);
        Instances split;
        noClasses = data.numClasses();
        noAttributes = ins.numAttributes();
        noInstances = data.numInstances();
        trainingData = new double [noInstances][][];
        for(int i=0;i<noInstances;i++){
           ins= data.instance(i);
           split=ins.relationalValue(0);
           double[][] train = new double[split.numInstances()][];
            for(int j=0;j<split.numInstances();j++){
                train[j]=split.instance(j).toDoubleArray();
            }
            trainingData[i]=train;
        }
    }

    //find the minimum of 3 values
    public double min(double a, double b, double c){
        double [] min = {a,b,c};
        Arrays.sort(min);
        return min[0];
    }
    
    //calculate euclidean distance between two points
    public double euclidean(double a, double b){
        return Math.pow(a-b,2);
    }
    
    public double distance(double [][] train, double [][] test, double currentMin){
        //initialise a matrix and set all valuse to max to prevent an incorrect 
        //distance being returned and used
        double [] [] matrix = new double [train.length][train.length];
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix.length; j++){
                matrix[i][j] = Double.MAX_VALUE;
            }
        }
        //calculate distances for the first column
        for(int i=0; i<matrix.length; i++){
            matrix[i][0] = 0;
            //sum of euclidean distances between all series values for train
            //and all values for the test sereis were time = 0 for test
            for(int j=0; j<train[i].length; j++){
                matrix[i][0] += euclidean(train[i][j], test[0][j]);
            }
        }
        //calculate distances for the first row
        for(int i=0; i<matrix.length; i++){
            matrix[0][i] = 0;
            //sum of euclidean distances between all series values for test
            //and all values for the train sereis were time = 0 for test
            for(int j=0; j<train[i].length; j++){
                matrix[0][i] += euclidean(train[0][j], test[i][j]);
            }
        }
        //set the bottom left value to 0
        matrix [0][0] = 0;
        //calculate distances for the rest of the matrix row by row
        double distance;
        //row
        for(int i=0; i<matrix.length; i++){
            //column
            for(int j=0; j<matrix.length; j++){
                //don't overwrite the values around the edge
                if(i!=0 && j!=0){
                    //Stop the path deviating too far from the diagonal
                    if(j<i+(matrix.length/4) && j>i-(matrix.length/4)){
                        distance = 0;
                        for(int k=0; k<train[i].length; k++){
                            //if the distance doesn't exceed the current min
                            if(distance < currentMin){
                                distance += euclidean(train[i][k], test[j][k]);
                            }
                            //else return the max value as this isn't the shotest distance
                            else{
                                return Double.MAX_VALUE;
                            }
                        }
                        //and the minimum value from the surrounding 3 values
                        distance += min(matrix[i][j-1], matrix[i-1][j], matrix[i-1][j-1]);
                        matrix[i][j]= distance;
                    }
                }
            }
        }
        //return top right of the matrix
        return matrix[matrix.length-1][matrix.length-1];
    }
    
    public double [][] transposeArray(double [][] array){
        int width = array.length;
        int height = array[0].length;
        double[][] array_new = new double[height][width];
        for (int w = 0; w < width; w++) {
            for (int h = 0; h < height; h++) {
              array_new[h][w] = array[w][h];
            }
        }
        return array_new;
    }
    
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double minValue = Double.MAX_VALUE;
        double minClass = -1;
        //put the test instance in an array
        Instances testSplit = instance.relationalValue(0);
        double[][] test = new double[testSplit.numInstances()][];
        for(int j=0;j<testSplit.numInstances();j++){
            test[j]=testSplit.instance(j).toDoubleArray();
        }
        double temp = 0;
        //for all training instances find the distance
        for(int i=0; i<trainingInstances.numInstances(); i++){
            
            temp = distance(trainingData[i], test, minValue);
            
            //check if the current distance is less then the current min distance
            if(temp<minValue){
                minValue = temp;
                minClass = trainingInstances.instance(i).classValue();
            }
        }
        return minClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
