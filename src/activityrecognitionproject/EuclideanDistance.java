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
            data.set(0, znorm(data.instance(i)));
            for(int a=0; a<noAttributes; a++){
                trainingData[i][a] = data.instance(i).value(a);
            }
        }
    }
    
    protected Instance znorm(Instance ts){
        double sumNum = 0;
        double sumSqu = 0;
        double att = noAttributes-1;
        for (int i = 0; i < att; i++) {
            sumNum += ts.value(i);
            sumSqu += Math.pow(ts.value(i), 2);
        }
        
        double mean = sumNum / att;
        double var = (att * sumSqu - sumNum * sumNum) / (att * att);
        double stdev = Math.sqrt(var);
       
        for (int i = 0; i < att; i++) {
            double z = (ts.value(i)-mean)/stdev;
            ts.setValue(i, z);
        }
       return ts;
    }
    
    protected double euclidean(double a, double b){
        return Math.pow(a-b,2);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        //calculate all euclidean distances between the test intance and all
        //trainign instances
        double distance = 0;
        double currentSmallest = Double.MAX_VALUE;
        double predictedClass = -1;
        for(int i=0; i<noInstances; i++){
            for(int a=0; a<noAttributes-1; a++){
                distance += euclidean(trainingData[i][a],instance.value(a));
            }
            if(distance<currentSmallest){
                currentSmallest = distance;
                predictedClass = trainingInstances.get(i).classValue();
            }
            distance = 0;
        }
        
        return predictedClass;
        
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
                Instances all = ClassifierTools.loadData("/Users/phillipperks/Desktop/3rd-Year-Project/ARFF_Files/Cross Validation/Combination/MVMotionUni.arff");
                    
                EuclideanDistance ed = new EuclideanDistance();
                ed.buildClassifier(all);
                for(int i=0; i<all.instance(0).numAttributes();i++){
                    System.out.print(all.instance(0).value(i) + "\t\t");
                }
                System.out.println("");
                all.set(0, ed.znorm(all.instance(0)));
                for(int i=0; i<all.instance(0).numAttributes();i++){
                    System.out.print(all.instance(0).value(i) + "\t");
                }
    }
}
