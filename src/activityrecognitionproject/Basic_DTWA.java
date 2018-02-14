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
public class Basic_DTWA implements Classifier {

    
    Classifier DTWA;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        setClassifier(data);
        this.DTWA.buildClassifier(data);
    }

    private void setClassifier(Instances data) throws Exception{
        int [] S_Success = findSuccesses(data);
        int S_iSuccess =  S_Success[0];
        int S_dSuccess =  S_Success[1];
        
        if(S_iSuccess>=S_dSuccess)
            this.DTWA = new DTWI();
        else
            this.DTWA = new DTWD();
        
    }
    
    private int [] findSuccesses(Instances data) throws Exception{
        int S_iSuccess = 0;
        int S_dSuccess = 0;
        Classifier DTWD = new DTWD();
        Classifier DTWI = new DTWI();
        for(int i=0; i<data.numInstances(); i++){
            Instances cvData = data;
            cvData.delete(i);
            DTWD.buildClassifier(cvData);
            DTWI.buildClassifier(cvData);
            if(data.instance(i).classValue() == DTWD.classifyInstance(data.instance(i))){
                S_dSuccess++;
            }
            if(data.instance(i).classValue() == DTWI.classifyInstance(data.instance(i))){
                S_iSuccess++;
            }
            
            
        }
        int [] S_Success = {S_iSuccess, S_dSuccess};
        return S_Success;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return DTWA.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
