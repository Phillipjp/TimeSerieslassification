/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import java.io.FileWriter;
import java.io.IOException;


/**
 *
 * @author phillipperks
 * @param <C>
 */
public class ClassifierWrapper <C extends weka.classifiers.Classifier> {
    
    private C classifier;
    private int correct;
    private double accuracy;
    private Instances test;
    private Instances train;
    
    public ClassifierWrapper(C classifier, Instances test, Instances train ) throws Exception{
        this.classifier = classifier;
        this.correct = 0;
        this.train = train;
        this.test = test;
        
        this.classifier.buildClassifier(this.train);
        for(Instance i: test){
            if(this.classifier.classifyInstance(i) == i.value(this.test.numAttributes()-1)){
                this.correct++;
            }
        }
        this.accuracy = (double)this.correct/test.numInstances();
    }
    
    //returns the classifier
    public C getClassifier(){
        return classifier;
    }
    
    //classifies Instance i using this classifier
    public double classifyInstance(Instance i) throws Exception{
        return classifier.classifyInstance(i);
    }
    
    //returns the training Instances
    public Instances getTrain(){
        return train;
    }
    
    //returns the test Instances
    public Instances getTest(){
        return test;
    }
    
    //returns accuracy of classifier
    public double getAccuracy(){
        return accuracy;
    }
    
    //classifies all instances and prints the accuracy as well as a confusion matrix
    public void classifyAllInstances(){
        StringBuilder str = new StringBuilder();
        for(int i=0; i<50; i++){
            str.append("=");
        }
        
        str.append("\n\n").append("Class Name: ");
        str.append(classifier.getClass().getName()).append("\n");
        
        str.append("Accuracy: ").append((double)correct/test.numInstances()*100);
        str.append("% \n\n");
        
        int classes = test.numClasses();
        int[][] matrix = new int [classes][classes];
        for(Instance i: test){
            try {
                matrix[(int)classifier.classifyInstance(i)][(int)i.classValue()] += 1;
            } catch (Exception ex) {
                Logger.getLogger(ClassifierWrapper.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        str.append("\t");
        
        for(int i=0; i< classes; i++){
            str.append((double)i).append("\t");
        }
        
        str.append("\n");
        for(int i=0; i<20; i++){
            str.append("-");
        }
         str.append("\n");
        
        for(int i=0; i< classes; i++){
            str.append((double)i).append("|").append("\t");
            for(int j=0; j< classes; j++){
                str.append(matrix[i][j]).append("\t");
            }
            if(i < classes-1){
                str.append("\n   |\n"); 
            }
            else{
                str.append("\n\n");
            }
        }
         System.out.println(str.toString());
    }
    
    

    public void writeCsvFile(String fileName, String classifierName) {
        
        FileWriter fileWriter = null;
        final String NEW_LINE_SEPARATOR = "\n";
        final String COMMA_DELIMITER = ",";
        StringBuilder results = new StringBuilder();
        try {
                fileWriter = new FileWriter("/Users/phillipperks/Desktop/3rd-Year-Project/results/" + classifierName + "/" + fileName + ".csv");
                long startTime = System.currentTimeMillis();
                for(Instance i: test){
                    try {
                        results.append(i.classValue() + COMMA_DELIMITER + classifier.classifyInstance(i) + COMMA_DELIMITER + COMMA_DELIMITER );
                        for(int j=0; j<classifier.distributionForInstance(i).length; j++){
                            if(j != classifier.distributionForInstance(i).length - 1){
                                results.append(classifier.distributionForInstance(i)[j] + COMMA_DELIMITER);
                            }
                            else{
                                 results.append(Double.toString(classifier.distributionForInstance(i)[j]));
                            }
                        }
                        results.append(NEW_LINE_SEPARATOR);
                        
                    } catch (Exception ex) {
                        Logger.getLogger(ClassifierWrapper.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                //Write the CSV file header
                //write the type of data, the classifier and the Instaces type
                fileWriter.append("racketSports," + classifierName + ",test ");
                fileWriter.append(NEW_LINE_SEPARATOR);
                fileWriter.append("BuildTime," + duration + COMMA_DELIMITER + classifier.getClass());
                fileWriter.append(NEW_LINE_SEPARATOR);
                fileWriter.append(Double.toString((double)correct/test.numInstances()));
                fileWriter.append(NEW_LINE_SEPARATOR);
                fileWriter.append(results);
                
                





                System.out.println("CSV file was created successfully !!!");

        } catch (Exception e) {
                System.out.println("Error in CsvFileWriter !!!");
                e.printStackTrace();
        } finally {

                try {
                        fileWriter.flush();
                        fileWriter.close();
                } catch (IOException e) {
                        System.out.println("Error while flushing/closing fileWriter !!!");
                        e.printStackTrace();
                }

        }
    }
    
}
