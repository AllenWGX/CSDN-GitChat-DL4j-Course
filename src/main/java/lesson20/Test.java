package lesson20;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Test {
    private static ComputationGraph getMultiTaskModel(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                        .updater(new Adam(0.01))
                        .graphBuilder()
                        .addInputs("input")
                        .addLayer("dense-1", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input")
                        .addLayer("dense-2", new DenseLayer.Builder().nIn(4).nOut(4).build(), "dense-1")
                        //
                        .addLayer("dense-out-pctr", new DenseLayer.Builder().nIn(4).nOut(2).build(), "dense-2")
                        .addLayer("dense-out-pcvr", new DenseLayer.Builder().nIn(4).nOut(2).build(), "dense-2")
                        .addVertex("dense-out-pcvctr", new ElementWiseVertex(ElementWiseVertex.Op.Product), "dense-out-pctr", "dense-out-pcvr")
                        //
                        .addLayer("out1", new OutputLayer.Builder()
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2).nOut(2).build(), "dense-out-pctr")
                        .addLayer("out2", new OutputLayer.Builder()
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2).nOut(2).build(), "dense-out-pcvr")
                        .addLayer("out3", new OutputLayer.Builder()
                                        .activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nIn(2).nOut(2).build(), "dense-out-pcvctr")
                        .setOutputs("out1","out2","out3")
                        .build();
        
        return new ComputationGraph(conf);
    }
    
    private static ComputationGraph getMultiTaskModel2(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                        .updater(new Adam(0.01))
                        .graphBuilder()
                        .addInputs("input1", "input1-1","input2", "input2-1")
                        .addLayer("embedding1", new EmbeddingLayer.Builder().nIn(10).nOut(12).build(), "input1")
                        .addLayer("embedding1-1", new EmbeddingLayer.Builder().nIn(10).nOut(12).build(), "input1-1")
                        .addLayer("embedding2", new EmbeddingLayer.Builder().nIn(10).nOut(12).build(), "input2")
                        .addLayer("embedding2-1", new EmbeddingLayer.Builder().nIn(10).nOut(12).build(), "input2-1")
                        .addVertex("ElementwiseAdd1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "embedding1","embedding1-1")
                        .addVertex("ElementwiseAdd2", new ElementWiseVertex(ElementWiseVertex.Op.Add), "embedding2","embedding2-1")
                        //
                        .addLayer("out1", new OutputLayer.Builder()
                                        .activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nIn(12).nOut(2).build(), "ElementwiseAdd1")
                        .addLayer("out2", new OutputLayer.Builder()
                                        .activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nIn(12).nOut(2).build(), "ElementwiseAdd2")
                        //
                        .setOutputs("out1","out2")
                        .build();
        
        return new ComputationGraph(conf);
    }
    
    public static void main(String[] args){
        double[][] features = new double[][]{{0.9,0.7,0.6},{0.9,0.7,0.6},{0.9,0.7,0.6},{0.9,0.7,0.6}};
        double[][] labels1 = new double[][]{{1.0,0.0},{0.0,1.0},{1.0,0.0},{0.0,1.0}};
        double[][] labels2 = new double[][]{{1.0,0.0},{0.0,1.0},{1.0,0.0},{0.0,1.0}};
        double[][] labels3 = new double[][]{{1.0,0.0},{0.0,1.0},{1.0,0.0},{0.0,1.0}};
        //
        INDArray featuresND = Nd4j.create(features);
        INDArray labelND1 = Nd4j.create(labels1);
        INDArray labelND2 = Nd4j.create(labels2);
        INDArray labelND3 = Nd4j.create(labels3);
        //
        ComputationGraph model = getMultiTaskModel();
        model.init();
        System.out.println(model.summary());
        model.fit(new INDArray[]{featuresND}, new INDArray[]{labelND1, labelND2, labelND3});
        //
        ComputationGraph model2 = getMultiTaskModel2();
        model2.init();
        System.out.println(model2.summary());

    }
}
