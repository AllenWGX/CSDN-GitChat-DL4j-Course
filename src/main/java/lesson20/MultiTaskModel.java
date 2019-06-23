package lesson20;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MultiTaskModel {
    private static ComputationGraph getMultiTaskModel(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                        .updater(new Adam(0.01))
                        .graphBuilder()
                        .addInputs("input")
                        .addLayer("dense-1", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input")
                        .addLayer("dense-2", new DenseLayer.Builder().nIn(4).nOut(4).build(), "dense-1")
                        .addLayer("out1", new OutputLayer.Builder()
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4).nOut(2).build(), "dense-2")
                        .addLayer("out2", new OutputLayer.Builder()
                                .activation(Activation.IDENTITY)
                                .lossFunction(LossFunctions.LossFunction.MSE)
                                .nIn(4).nOut(1).build(), "dense-2")
                        .setOutputs("out1","out2")
                        .build();
        
        return new ComputationGraph(conf);
    }
    
    public static void main(String[] args) {
        double[][] features = new double[][]{{0.9,0.7,0.6},{0.9,0.7,0.6},{0.9,0.7,0.6},{0.9,0.7,0.6}};
        double[][] labels1 = new double[][]{{1.0,0.0},{0.0,1.0},{1.0,0.0},{0.0,1.0}};
        double[][] labels2 = new double[][]{{0.8},{0.8},{0.8},{0.8}};
        //
        INDArray featuresND = Nd4j.create(features);
        INDArray labelND1 = Nd4j.create(labels1);
        INDArray labelND2 = Nd4j.create(labels2);
        //
        ComputationGraph model = getMultiTaskModel();
        model.init();
        model.fit(new INDArray[]{featuresND}, new INDArray[]{labelND1, labelND2});

    }

}
