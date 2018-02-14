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
public class DTWI implements Classifier {
    
    double [][][] trainingData;
    int noClasses;
    int noAttributes;
    int noInstances;
    Instances trainingInstances;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainingInstances = data;
        Instance ins= data.instance(0);
        Instances split=ins.relationalValue(0);
        noClasses = data.numClasses();
        noAttributes = split.numAttributes();
        noInstances = data.numInstances();
        trainingData = new double [noInstances][][];
        for(int i=0;i<noInstances;i++){
           ins= data.instance(i);
           split=ins.relationalValue(0);
           double[][] train = new double[split.numInstances()][];
            for(int j=0;j<split.numInstances();j++){
                train[j]=split.instance(j).toDoubleArray();
            }
            train = transposeArray(train);
            trainingData[i]=train;
        }
    }
    
    //calcualte the minimum of 3 values
    public double min(double a, double b, double c){
        double [] min = {a,b,c};
        Arrays.sort(min);
        return min[0];
    }
    
    //calculate euclidean distance between two points
    public double euclidean(double a, double b){
        return Math.pow(a-b,2);
    }
    
    public double distance(double [] train, double [] test){
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
            //euclidean distance between all the train values and test value
            //at 0 in time
            matrix[i][0]= euclidean(train[i], test[0]);
        }
        //calculate distances for the first row
        for(int j=0; j<matrix.length; j++){
            //euclidean distance between all the test values and train value
            //at 0 in time
            matrix[0][j]= euclidean(train[0], test[j]);
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
                        //calcualte euclidean disatnce between train and test instance attributes
                        distance = euclidean(train[i], test[j]);
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
        //transpose array
       test = transposeArray(test);
        double temp = 0;
        //for all training instsnces find the distance
        for(int i=0; i<trainingInstances.numInstances(); i++){
            //for each series in the test data get the distance
            for(int j=0 ;j<trainingData[i].length; j++){
                //only calculate distance if it's possible it might be the best
                if(temp<minValue){
                    temp += distance(trainingData[i][j], test[j]);
                }  
            }
            //check if the current distance is less then the current min distance
            if(temp<minValue){
                minValue = temp;
                minClass = trainingInstances.instance(i).classValue();
            }
            temp = 0;
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
