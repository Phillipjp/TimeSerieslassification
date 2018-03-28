/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class Enhanced_DTWI extends DTWI {
    int warp_size;
    
    public Enhanced_DTWI(int warp_size){
        super();
        this.warp_size = warp_size;
    }
    
    public Enhanced_DTWI(){
        super();
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
    
    @Override
    protected double distance(double [] train, double [] test){
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
                    if(j<i+warp_size && j>i-warp_size){
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
                double temp =0;
                double minClass = 0;
                double minValue = Double.MAX_VALUE;
                //for all train data
                for(int train=0; train<cvTrain.length; train++){
                    //for each series in the train data
                    for(int series=0; series < cvTrain[train].length; series++ ){
                        if(temp<minValue){
                            temp+= distance(cvTrain[train][series], cvTest[test][series]);
                        }
                    }
                    if(temp<minValue){
                        minValue=temp;
                        minClass = data[0].instance(train).classValue();
                    }
                    temp = 0;
                }
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
        // System.out.println("");
        for(int i=0; i<correct.length; i++){
           // System.out.print(correct[i] + "\t");
            if(biggest <= correct[i]){
                biggest = correct[i];
                warp = i;
            }
        }
        
        this.warp_size =  warp;
    }
    

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    
    
}
