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
            data.set(i, znorm(data.instance(i)));
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
        instance =znorm(instance);
        double distance = 0;
        double currentSmallest = Double.MAX_VALUE;
        double predictedClass = -1;
        for(int i=0; i<noInstances; i++){
            for(int a=0; a<noAttributes-1; a++){
                distance += euclidean(trainingInstances.get(i).value(a),instance.value(a));
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
        double [] prob = new double [trainingInstances.numClasses()];
        double result = classifyInstance(instance);
        prob[(int)result] = 1;
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
