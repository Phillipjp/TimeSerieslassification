/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import java.util.ArrayList;
import java.util.Collections;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class Basic_DTWA implements Classifier {

    
    Classifier DTWA;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        setClassifier(data);
        this.DTWA.buildClassifier(data);
    }

    private void setClassifier(Instances data) throws Exception{
        int [] S_Success = findSuccesses(data);
        int S_iSuccess =  S_Success[0];
        int S_dSuccess =  S_Success[1];
        
        if(S_iSuccess>=S_dSuccess)
            this.DTWA = new DTWI();
        else
            this.DTWA = new DTWD();
        
    }
    
    private int [] findSuccesses(Instances data) throws Exception{
        int S_iSuccess = 0;
        int S_dSuccess = 0;
        Classifier DTWD = new DTWD();
        Classifier DTWI = new DTWI();
        double minD;
        double minI = 0;
        for(int i=0; i<data.numInstances(); i++){
            Instances cvData = data;
            cvData.delete(i);
            DTWD.buildClassifier(cvData);
            DTWI.buildClassifier(cvData);
            
            minD = nearestNeigboughDistanceD(data, i);
            minI = nearestNeigboughDistanceI(data, i);
            
            if(data.instance(i).classValue() == DTWD.classifyInstance(data.instance(i))){
                S_dSuccess++;
            }
            if(data.instance(i).classValue() == DTWI.classifyInstance(data.instance(i))){
                S_iSuccess++;
            }
            
            
        }
        int [] S_Success = {S_iSuccess, S_dSuccess};
        return S_Success;
    }
    
    private double nearestNeigboughDistanceD(Instances data, int test) throws Exception{
        //put training instances in an array
        double [][][] trainingData = new double [data.numInstances()-1][][];
        for(int i=0;i<data.numInstances();i++){
            if(i!=test){
                Instance ins= data.instance(i);
                Instances split=ins.relationalValue(0);
                double[][] train = new double[split.numInstances()][];
                 for(int j=0;j<split.numInstances();j++){
                     train[j]=split.instance(j).toDoubleArray();
                 }
                 trainingData[i]=train;
            }
        }
        
        //put test instrance in an array
        Instances testSplit = data.instance(test).relationalValue(0);
        double[][] testInstance = new double[testSplit.numInstances()][];
        for(int j=0;j<testSplit.numInstances();j++){
            testInstance[j]=testSplit.instance(j).toDoubleArray();
        }
        
        double minValue = Double.MAX_VALUE;
        double temp = 0;
        //for all training instances find the distance
        for(int i=0; i<trainingData.length; i++){
            temp = new DTWD().distance(trainingData[i], testInstance, minValue);
            //check if the current distance is less then the current min distance
            if(temp<minValue){
                minValue = temp;
            }
        }
        
        return minValue;
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
    
    private double nearestNeigboughDistanceI(Instances data, int test) throws Exception{
        //put training instances in an array
        double [][][] trainingData = new double [data.numInstances()-1][][];
        for(int i=0;i<data.numInstances();i++){
            if(i!=test){
                Instance ins= data.instance(i);
                Instances split=ins.relationalValue(0);
                double[][] train = new double[split.numInstances()][];
                 for(int j=0;j<split.numInstances();j++){
                     train[j]=split.instance(j).toDoubleArray();
                 }
                 train = transposeArray(train);
                 trainingData[i]=train;
            }
        }
        
        //put test instrance in an array
        Instances testSplit = data.instance(test).relationalValue(0);
        double[][] testInstance = new double[testSplit.numInstances()][];
        for(int j=0;j<testSplit.numInstances();j++){
            testInstance[j]=testSplit.instance(j).toDoubleArray();
        }
        testInstance = transposeArray(testInstance);
        
        double minValue = Double.MAX_VALUE;
        double temp = 0;
        //for all training instances find the distance
        for(int i=0; i<trainingData.length; i++){
            for(int j=0 ;j<trainingData[i].length; j++){
                //only calculate distance if it's possible it might be the best
                if(temp<minValue){
                    temp += new DTWI().distance(trainingData[i][j], testInstance[j]);
                }  
            }
            //check if the current distance is less then the current min distance
            if(temp<minValue){
                minValue = temp;
            }
        }
        
        return minValue;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return DTWA.classifyInstance(instance);
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
