package activityrecognitionproject;

import weka.core.Statistics;

public class NormalDistribution {
    
    public static double probability(double x, double m, double std){
        double y=(x-m)/std;
        double p=Statistics.normalProbability(y);
        if(p>0.5) p=1-p;
        return p;
    }
    public static void main(String[] args){
        System.out.println(" mean 0, std dev = 1");
        for(double x=-1;x<=1;x+=0.1)
            System.out.println(" x = "+x+" p ="+probability(x,0,1));
            
    }
}