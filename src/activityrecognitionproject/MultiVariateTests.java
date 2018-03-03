/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;


/*
Multivariate data can be stored in Wekas "multi instance" format
https://weka.wikispaces.com/Multi-instance+classification

for TSC, the basic univariate syntax is 

 */


import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import utilities.ClassifierTools;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A getting started with relational attributes in Weka. Once you have the basics
 * there are a range of tools for manipulating them in 
 * package utilities.multivariate_tools 
 * 
 * See https://weka.wikispaces.com/Multi-instance+classification
 * for more 
 * @author ajb
 */
public class MultiVariateTests {
    
    
    public static void main(String[] args) throws Exception {
//Load a multivariate data set
//        String path="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\Multivariate\\univariateConcatExample";
//        Instances train = ClassifierTools.loadData(path);
//        System.out.println(" univariate data = "+train);
        String path="/Users/phillipperks/Desktop/3rd-Year-Project/ARFF_Files/Cross Validation/Combination/MVMotionMulti.arff";
        Instances train = ClassifierTools.loadData(path);
        System.out.println(" multivariate data = "+train);
//Recover the first instance
        Instance first=train.instance(0);
        
//Split into separate dimensions
        Instances split=first.relationalValue(0);
        System.out.println(" A single multivariate case split into 3 instances with no class values= "+split);
        for(Instance ins:split)
            System.out.println("Dimension of first case =" +ins);
//Extract as arrays
        double[][] d = new double[split.numInstances()][];
        for(int i=0;i<split.numInstances();i++){
           d[i]=split.instance(i).toDoubleArray();
           
        }
        
        System.out.println("print out all columns");
        for(int i=0; i<d[0].length; i++){
            System.out.println(d[0][i]);
        }
        
        System.out.println("");
        System.out.println(train.instance(0).classValue());
        
        System.out.println("rows: " + d.length);
        System.out.println("columns: " + d[0].length);
        
        for(int i=0; i<d.length; i++){
            for(int j=0; j<d[i].length; j++){
                System.out.print(d[i][j] + "\t");
            }
            System.out.println("");
        }
        
        //transpose array
        int width = d.length;
        int height = d[0].length;
        double[][] dTranspose = new double[height][width];
        for (int w = 0; w < width; w++) {
          for (int h = 0; h < height; h++) {
            dTranspose[h][w] = d[w][h];
          }
        }
        
        System.out.println("transposed array: \n");
        for(int i=0; i<dTranspose.length; i++){
            for(int j=0; j<dTranspose[i].length; j++){
                System.out.print(dTranspose[i][j] + "\t");
            }
            System.out.println("");
        }
        
        
        System.out.println("rows: " + dTranspose.length);
        System.out.println("columns: " + dTranspose[0].length);
        
        for(int i=0; i<dTranspose[0].length; i++){
            System.out.print(dTranspose[0][i] + "\t");
        }
        
        System.out.println("");
        System.out.println("Class value: " + first.classValue());
        System.out.println("");
        
        int noInstances = train.numInstances();
        double [][][] trainingData = new double [noInstances][][];
        for(int i=0;i<noInstances;i++){
           Instance ins= train.instance(i);
           Instances splitTest=ins.relationalValue(0);
           double[][] trainTest = new double[splitTest.numInstances()][];
            for(int j=0;j<splitTest.numInstances();j++){
                trainTest[j]=splitTest.instance(j).toDoubleArray();
            }
            trainTest = transposeArray(trainTest);
            trainingData[i]=trainTest;
        }
        
        for(int i=0; i< trainingData.length; i++){
            System.out.println("Instance " + i + ":");
            for(int j=0; j<trainingData[i].length; j++){
                for(int k=0; k<trainingData[i][j].length; k++){
                    System.out.print(trainingData[i][j][k] + "\t");
                }
                System.out.println("");
            }
            System.out.println("");
        }
        
        System.out.println(trainingData.length);
        System.out.println(trainingData[0].length);
        System.out.println(trainingData[0][0].length);
        
        int [] predictions = {1, 1, 2, 2, 3};
        
        int currentTie = -1;
        HashSet<Integer> classes = new HashSet<>();
        for (int i = 0; i < predictions.length; i++) {
            if(predictions[i] > currentTie){
                 classes.clear();
                 currentTie = predictions[i];
                 classes.add(i);
            }
            else{
                for (int j = 0; j < predictions.length; j++) {
                    if (i != j) {
                        if(predictions[i] == predictions[j]){
                            if(predictions[i] == currentTie){
                                classes.add(i);
                                currentTie = predictions[i];
                            }

                        }
                    }
                }
            }
        }
        int predictedClass = -1;
         ArrayList<Integer> classesList = new ArrayList<>(classes);
        if(classesList.size()==1){
            predictedClass = classesList.get(0);
        }
        System.out.println("DUPLICATES TEST");
        System.out.println(classes);
        System.out.println("PREDICTED CLASS: \t" + predictedClass);
        
//        Instances all = ClassifierTools.loadData("/Users/phillipperks/Desktop/3rd-Year-Project/ARFF_Files/Cross Validation/Combination/MVMotionUni.arff");
//
//        
//        Instances newInstances = new Instances(all, 5);
//        System.out.println(newInstances.numInstances());
//        newInstances.add(all.instance(0));
//        System.out.println(newInstances.numInstances());
    }
    public static double [][] transposeArray(double [][] array){
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
}