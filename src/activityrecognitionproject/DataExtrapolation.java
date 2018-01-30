/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class DataExtrapolation {
    public static void main(String[] args) throws Exception {
        
        //Instances all = InstancesloadData("/Users/phillipperks/Desktop/3rd Year Project/ARFF_Files/allData.arff");
        Instances train = null;
        FileReader reader = new FileReader("/Users/phillipperks/Desktop/3rd Year Project/ARFF_Files/allData.arff");
            train = new Instances(reader);
        instanceToCSV("badmintonSmash", 3, train);
        //instanceToCSV("badmintonGraph", 37, all);
        
    }
    
    public static void instanceToCSV(String fileName, int instancePos, Instances instances){
         FileWriter fileWriter = null;
        final String NEW_LINE_SEPARATOR = "\n";
        final String COMMA_DELIMITER = ",";
        double [] instancesArray = new double[instances.get(instancePos).numAttributes()];
        System.out.println(instances.get(instancePos).numAttributes());
        for(int i=0; i<instances.get(instancePos).numAttributes(); i++) {
            instancesArray[i] = instances.get(instancePos).value(i);
        }
        
        try {
                fileWriter = new FileWriter("/Users/phillipperks/Desktop/3rd Year Project/graphs/"  + fileName + ".csv");
                
                try {
                    for(int i=0; i<instancesArray.length; i++){
                        if(i != instancesArray.length-1)
                            fileWriter.append(instancesArray[i] + COMMA_DELIMITER );
                        else
                            fileWriter.append(instancesArray[i] + "");
                    }
                    

                } catch (Exception ex) {
                    Logger.getLogger(ClassifierWrapper.class.getName()).log(Level.SEVERE, null, ex);
                }
                





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
