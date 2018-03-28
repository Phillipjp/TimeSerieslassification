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
    private int correctSport;
    private double accuracy;
    private double sportAccuracy;
    private Instances test;
    private Instances train;
    private double [][] matrix;
    private double balancedAccuracy;
    
    public ClassifierWrapper(C classifier, Instances test, Instances train ) throws Exception{
        this.classifier = classifier;
        this.correct = 0;
        this.correctSport = 0;
        this.train = train;
        this.test = test;
        
        this.classifier.buildClassifier(this.train);
        matrix = new double[train.numClasses()][train.numClasses()];
        for(Instance i: test){
            double result = this.classifier.classifyInstance(i);
            matrix[(int)result][(int)i.classValue()] += 1;
            if(result == i.value(this.test.numAttributes()-1)){
                this.correct++;
            }
            if(result == 0 && i.value(this.test.numAttributes()-1) == 1 || 
                    result == 1 && i.value(this.test.numAttributes()-1) == 0 ||
                    result == 2 && i.value(this.test.numAttributes()-1) == 3 ||
                    result == 3 && i.value(this.test.numAttributes()-1) == 2 ||
                    result == i.value(this.test.numAttributes()-1)){
                this.correctSport++; 
            }
            
        }
        
        
        this.accuracy = (double)this.correct/test.numInstances();
        this.sportAccuracy = (double)this.correctSport/test.numInstances();
        balancedAccuracy = 0;
        for (int i = 0; i < train.numClasses(); i++) {
               double trp = 0;
               double total = 0;
               for (int j = 0; j < train.numClasses(); j++) {
                if(i == j){
                   trp = matrix[j][i];
                }
                total += matrix[j][i];
            }
            balancedAccuracy += trp/total;
        }
        balancedAccuracy /= train.numClasses();
        
        
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
    
    //returns accuracy of classifier
    public double getSportAccuracy(){
        return sportAccuracy;
    }
    
    //returns accuracy of classifier
    public double getBalancedAccuracy(){
        return balancedAccuracy;
    }
    
    //returns number of times the correct sport was classified
    public double getCorrectSport(){
        return correctSport;
    }
    
    //returns number of times the correct action was classified
    public double getCorrect(){
        return correct;
    }
    
    //classifies all instances and prints the accuracy as well as a confusion matrix
    public void confusionMatrix(){
        StringBuilder str = new StringBuilder();
        for(int i=0; i<50; i++){
            str.append("=");
        }
        
        str.append("\n\n").append("Class Name: ");
        str.append(classifier.getClass().getName()).append("\n");
        
        str.append("Accuracy: ").append(accuracy*100);
        str.append("% \n");
        str.append("Sport Accuracy: ").append(sportAccuracy*100);
        str.append("% \n");
        str.append("Balanced Accuracy: ").append(balancedAccuracy*100);
        str.append("% \n\n");
        
        
        
        str.append("\t");
        
        int classes = test.numClasses();
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
                fileWriter = new FileWriter("FinalResults/" + classifierName + "/" + fileName + ".csv");
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
                fileWriter.append(Double.toString(accuracy)).append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(sportAccuracy)).append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(balancedAccuracy)).append(COMMA_DELIMITER);
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
