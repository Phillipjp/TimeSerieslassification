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
public class Basic_DTW implements Classifier{
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
    //calcualte the minimum of 3 values
    public double min(double a, double b, double c){
        double [] min = {a,b,c};
        Arrays.sort(min);
        return min[0];
    }
    
    public double euclidean(double a, double b){
        return Math.pow(a-b,2);
    }
    
    public double distance(Instance train, Instance test, Double currentMin){
        //initialise a matrix and set all valuse to max to prevent an incorrect 
        //distance being returned and used
        double [] [] matrix = new double [noAttributes][noAttributes];
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix.length; j++){
                matrix[i][j] = Double.MAX_VALUE;
            }
        }
        //calculate distances for the first column
        for(int i=0; i<matrix.length; i++){
            //euclidean distance between the current train value and test value
            //at 0 in time
            matrix[i][0] = euclidean(train.value(i),test.value(0));
        }
        //calculate distances for the first row
        for(int j=0; j<matrix.length; j++){
            //euclidean distance between the current train value and test value
            //at 0 in time
            matrix[j][0] = euclidean(train.value(0),test.value(j));
            
        }
        //set the bottom left value to 0
        matrix [0][0] = 0;
        //calculate distances for the rest of the matrix row by row
        double distance;
        //row
        for(int i=0; i<matrix.length; i++){
            //column
            for(int j=0; j<matrix.length; j++){
                //don't overwrite the valued around the edge
                if(i!=0 && j!=0){
                    //Stop the path deviating too far from the diagonal
                    if(j<i+(matrix.length/4) && j>i-(matrix.length/4)){
                        //the euclidean distance between the train and the test value
                        distance = euclidean(train.value(i),test.value(j));
                        //add the minimum value from the the the surounding squares
                        distance += min(matrix[i][j-1], matrix[i-1][j], matrix[i-1][j-1]);
                        //if the distance is already bigger than the currentMinimum value
                        //return MAX_VALUE as this can't be the class
                        if(distance > currentMin)
                            return Double.MAX_VALUE;
                        else
                            matrix[i][j]= distance;
                    }
                }
            }
        }
        //return the value at the top right corner
        return matrix[matrix.length-1][matrix.length-1];
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double minValue = Double.MAX_VALUE;
        double minClass = 0;
        
        double temp;
        for(Instance i: trainingInstances){
            temp = distance(i, instance, minValue);
            if(temp < minValue){
                minValue = temp;
                minClass = i.classValue();
            }
            
        }
        return minClass;
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
    
}
