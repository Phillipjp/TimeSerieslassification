/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activityrecognitionproject;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class Basic_DTWA extends Basic_DTW {

    
    Classifier DTWA;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        setClassifier(data);
        this.DTWA.buildClassifier(data);
    }

    private void setClassifier(Instances data) throws Exception{
        int [] S_Success = findSuccesses(data);
        int S_iSuccess =  S_Success[0];
        int S_dSuccess =  S_Success[3];
        int S_iSuccess_Badminton = S_Success[2];
        int S_dSuccess_Badminton = S_Success[5];
        int S_iSuccess_Squash = S_Success[1];
        int S_dSuccess_Squash = S_Success[4];
        //if DTWI has more successfull classifications
        if(S_iSuccess>S_dSuccess){
            this.DTWA = new DTWI();
        }
        //if DTWD has more successfull classifications
        else if(S_dSuccess>S_iSuccess){
            this.DTWA = new DTWD();
        }
        //if both classsifiers make the same number of correct classifications
        else{
            //if DTWI predicted the coreect sport  more
            if(S_iSuccess_Badminton+S_iSuccess_Squash>S_dSuccess_Badminton+S_dSuccess_Squash){
               this.DTWA = new DTWI(); 
            }
            //if DTWD predicted the correct sport more
            else if(S_dSuccess_Badminton+S_dSuccess_Squash>S_iSuccess_Badminton+S_iSuccess_Squash){
                this.DTWA = new DTWD();
            }
            //else use DTWI as a default
            else{
                this.DTWA = new DTWI(); 
            }
        }
        
    }
    
    private int [] findSuccesses(Instances data) throws Exception{
        int S_iSuccess = 0;
        int S_dSuccess = 0;
        int S_iSuccess_Badminton = 0;
        int S_dSuccess_Badminton = 0;
        int S_iSuccess_Squash = 0;
        int S_dSuccess_Squash = 0;
        Classifier DTWD = new DTWD();
        Classifier DTWI = new DTWI();
        for(int i=0; i<data.numInstances(); i++){
            Instances cvData = new Instances (data);
            cvData.delete(i);
            DTWD.buildClassifier(cvData);
            DTWI.buildClassifier(cvData);
            if(data.instance(i).classValue() == DTWD.classifyInstance(data.instance(i))){
                S_dSuccess++;
            }
            else if((data.instance(i).classValue() == 0 && DTWD.classifyInstance(data.instance(i)) == 1)
                    || (data.instance(i).classValue() == 1 && DTWD.classifyInstance(data.instance(i)) == 0) ){
                S_dSuccess_Squash++;
            }
            else if((data.instance(i).classValue() == 2 && DTWD.classifyInstance(data.instance(i)) == 3)
                    || (data.instance(i).classValue() == 3 && DTWD.classifyInstance(data.instance(i)) == 2) ){
                S_dSuccess_Badminton++;
            }
            
            if(data.instance(i).classValue() == DTWI.classifyInstance(data.instance(i))){
                S_iSuccess++;
            }
            else if((data.instance(i).classValue() == 0 && DTWD.classifyInstance(data.instance(i)) == 1)
                    || (data.instance(i).classValue() == 1 && DTWD.classifyInstance(data.instance(i)) == 0) ){
                S_iSuccess_Squash++;
            }
            else if((data.instance(i).classValue() == 2 && DTWD.classifyInstance(data.instance(i)) == 3)
                    || (data.instance(i).classValue() == 3 && DTWD.classifyInstance(data.instance(i)) == 2) ){
                S_iSuccess_Badminton++;
            }
        }
        int [] S_Success = {S_iSuccess, S_iSuccess_Squash, S_iSuccess_Badminton,
            S_dSuccess, S_dSuccess_Squash, S_dSuccess_Badminton};
        return S_Success;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return DTWA.classifyInstance(instance);
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return DTWA.distributionForInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
