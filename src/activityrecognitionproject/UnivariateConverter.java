import java.io.*;

/**
 *
 * @author JFeatherstone
 */
public class UnivariateConverter {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.io.UnsupportedEncodingException
     */
    public static void main(String[] args) throws IOException {
        convert("/Users/phillipperks/Desktop/3rd-Year-Project/ARFF_Files/Cross Validation/simonBadminton.arff", "simonBadmintonUni");
    }
        
    public static void convert(String dataLocation, String newName) throws IOException{
        
        File inputFile = new File(dataLocation);
        FileReader reader = new FileReader(inputFile);
        BufferedReader br = new BufferedReader(reader); 
        PrintWriter writer = new PrintWriter("/Users/phillipperks/Desktop/3rd-Year-Project/ARFF_Files/Cross Validation/" + newName + ".arff","UTF-8");

        writer.write("@relation MVMotion \n");

        for (int i = 0; i < 32; i++) {
            writer.write("@attribute Ax" + i + " numeric\n");
        }
        for (int i = 0; i < 32; i++) {
            writer.write("@attribute Ay" + i + " numeric\n");
        }
        for (int i = 0; i < 32; i++) {
            writer.write("@attribute Az" + i + " numeric\n");
        }
        for (int i = 0; i < 32; i++) {
            writer.write("@attribute Gx" + i + " numeric\n");
        }
        for (int i = 0; i < 32; i++) {
            writer.write("@attribute Gy" + i + " numeric\n");
        }
        for (int i = 0; i < 32; i++) {
            writer.write("@attribute Gz" + i + " numeric\n");
        }

        for (int i = 0; i < 33; i++) {
            br.readLine();
        }
        
        writer.write(br.readLine());

        writer.write("\n@data\n\n");

        br.readLine();
        br.readLine();

        String lineString ;
        String[] lengthCheck;
        int total = 0;
        int written = 0;
        //int [] writtenArray = new int [70];
        int count = 0;
        while((lineString = br.readLine()) != null){
            lengthCheck = lineString.split(",");
            total++;
            System.out.println(lengthCheck.length);
            //if(lengthCheck.length<1796 && lengthCheck.length>1794){
                written++;
                //writtenArray[count] = lengthCheck.length;
                count++;

                char[] chars = lineString.toCharArray();
                
                int commaCount = 0;
                for(char c : chars){  
                    
                    if(c == ','){
                        commaCount++;
                    }
                    if(c != '"' && c != '\\'){
                        if(c == 'n' && commaCount < 192)
                            c =',';
                        writer.write(c);
                    }
                }
                    
                
                writer.write("\n");
            //}
        }

        writer.close();
        br.close();
        System.out.println("Total: " + total);
        System.out.println("Written: " + written);
        System.out.println("Written Array");
//        for(int i= 0; i<writtenArray.length; i++){
//            System.out.println(writtenArray[i]);
//        }
    }
    
    private static void toLines(String dataLocation) throws IOException {
        File inputFile = new File(dataLocation);
        FileReader reader = new FileReader(inputFile);
        BufferedReader br = new BufferedReader(reader);

        String lineString = br.readLine();

        char[] chars = lineString.toCharArray();

        for (char c : chars){
            if(c == ',')
                c = '\n';

            System.out.print(c);
        }
    }
    
}
