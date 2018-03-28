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
public class DTWA extends Basic_DTW {

    
    Classifier DTWA;
    double [][][] trainingData;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainingInstances = data;
        noClasses = data.numClasses();
        noAttributes = data.numAttributes();
        noInstances = data.numInstances();
        trainingData = makeDataMultiVariate(data, noInstances);
    }

    protected double [][][] makeDataMultiVariate(Instances data, int length){
         double [][][] multiVariateData = new double [length][][];
         Instance ins;
         Instances split;
         for(int i=0;i<length;i++){
           ins= data.instance(i);
           split=ins.relationalValue(0);
           double[][] singleInstance = new double[split.numInstances()][];
            for(int j=0;j<split.numInstances();j++){
                singleInstance[j]=split.instance(j).toDoubleArray();
            }
            singleInstance = transposeArray(singleInstance);
            multiVariateData[i]=znorm(singleInstance);
        }
        
         return multiVariateData;
    }
    
    protected double [][] znorm(double [][] ts){
        for (int i = 0; i < ts.length; i++) {
            double sumNum = 0;
            double sumSqu = 0;
            double att = ts[i].length;
            for (int j = 0; j < att; j++) {
                sumNum += ts[i][j];
                sumSqu += Math.pow(ts[i][j], 2);
            }
        
            double mean = sumNum / att;
            double var = (att * sumSqu - sumNum * sumNum) / (att * att);
            double stdev = Math.sqrt(var);

            for (int j = 0; j < att; j++) {
                double z = (ts[i][j]-mean)/stdev;
                ts[i][j]=z;
            }
        }
        return ts;
    }
    private void setClassifier(Instances data, Instance test) throws Exception{
        //find out how to actually calculate minD and minI
        double minD = nearestNeigboughDistanceD(data, test);
        double minI = nearestNeigboughDistanceI(data, test);
        double S = minD/minI;
        
        double threshold = 0.7125;
        //System.out.println("S:\t" + S);
        if(S>threshold)
            this.DTWA = new DTWI();
        else
            this.DTWA = new DTWD();
        
    }
    
    private double learnThreshold(Instances data, Instance test) throws Exception{
        ArrayList<Double> [] S_Success = findScores(data, test);
        ArrayList<Double> S_iSuccess =  S_Success[0];
        ArrayList<Double> S_dSuccess =  S_Success[1];
        double threshold = 0;
        if(S_iSuccess.isEmpty() && S_dSuccess.isEmpty()){
            threshold = 1;
        }
        else if(S_iSuccess.isEmpty() && !S_dSuccess.isEmpty()){
            Collections.sort(S_dSuccess);
            threshold = S_dSuccess.get(0);
        }
        else if(!S_iSuccess.isEmpty() && S_dSuccess.isEmpty()){
            Collections.sort(S_iSuccess);
            threshold = S_iSuccess.get(0);
        }
        else{
            //threshold = decisionTree(S_iSuccess, S_dSuccess);
        }
        
        return threshold;
        
    }
    
    private ArrayList [] findScores(Instances data, Instance test) throws Exception{
        ArrayList<Double> S_iSuccess = new ArrayList<>();
        ArrayList<Double> S_dSuccess = new ArrayList<>();
        Classifier DTWD = new DTWD();
        Classifier DTWI = new DTWI();
        double minD;
        double minI = 0;
        for(int i=0; i<data.numInstances(); i++){
            Instances cvData = data;
            cvData.delete(i);
            DTWD.buildClassifier(cvData);
            DTWI.buildClassifier(cvData);
            
            minD = nearestNeigboughDistanceD(data, test);
            minI = nearestNeigboughDistanceI(data, test);
            
            if(data.instance(i).classValue() == DTWD.classifyInstance(data.instance(i))
                    && data.instance(i).classValue() != DTWI.classifyInstance(data.instance(i))){
                S_dSuccess.add(minD/minI);
            }
            if(data.instance(i).classValue() == DTWI.classifyInstance(data.instance(i))
                    && data.instance(i).classValue() != DTWD.classifyInstance(data.instance(i))){
                S_iSuccess.add(minD/minI);
            }
            
            
        }
        ArrayList [] S_Success = {S_iSuccess, S_dSuccess};
        return S_Success;
    }
    
    private double nearestNeigboughDistanceD(Instances data, Instance test) throws Exception{
        //put test instrance in an array
        Instances testSplit = test.relationalValue(0);
        double[][] testInstance = new double[testSplit.numInstances()][];
        for(int j=0;j<testSplit.numInstances();j++){
            testInstance[j]=testSplit.instance(j).toDoubleArray();
        }
        testInstance = transposeArray(testInstance);
        
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
    
    private double nearestNeigboughDistanceI(Instances data, Instance test) throws Exception{
        //put test instrance in an array
        Instances testSplit = test.relationalValue(0);
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
        setClassifier( trainingInstances,  instance);
        this.DTWA.buildClassifier(trainingInstances);
        return this.DTWA.classifyInstance(instance);
    }


    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
