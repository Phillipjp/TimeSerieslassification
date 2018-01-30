/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;


import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class MultivariateConverter {
    
    public static void convert(String newPathName, Instances data) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter(newPathName,"UTF-8");
        
        writer.write("@relation input\n");
        writer.write("@attribute input relational\n");
        
        for (int i = 1; i < 33; i++) {
            writer.write("\t@attribute t" + i + " numeric\n");
        }
        
        writer.write("@end input\n");
        writer.write("@attribute activity {Badminton_Smash, Badminton_Clear, Squash_ForehandBoast, Squash_BackhandBoast}\n");
        writer.write("@end data\n\n");
        
        for(Instance i: data){
            writer.write("\"");
            for(int j=0; j<32; j++){
                writer.write(i.toString(j, 10));
                writer.write(",");
                writer.write(i.toString(j+32, 10));
                writer.write(",");
                writer.write(i.toString(j+64, 10));
                writer.write(",");
                writer.write(i.toString(j+96, 10));
                writer.write(",");
                writer.write(i.toString(j+128, 10));
                writer.write(",");
                writer.write(i.toString(j+160, 10));
                if(j!=31){
                    writer.write("\\n");
                }
                
            }
            writer.write("\",");
            writer.write(i.toString(i.numAttributes()-1,1));
            writer.write("\n");
        }
        
        writer.close();
        
    }
    
    public static Instances loadData(String filePath){
        String dataLocation=filePath;
        Instances i = null;
        try{
            FileReader reader = new FileReader(dataLocation);
            i = new Instances(reader);
            i.setClassIndex(i.numAttributes()-1);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        return i;
    }
    
    public static void main(String[] args) throws IOException {
        
        Instances data = loadData("/Users/phillipperks/Desktop/3rd Year Project/ARFF_Files/format.arff");
        convert("/Users/phillipperks/Desktop/3rd Year Project/ARFF_Files/formatMULTI.arff", data);
        
    }
    
}
