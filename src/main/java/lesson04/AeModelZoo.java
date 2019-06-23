package lesson04;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class AeModelZoo {
    public static MultiLayerNetwork mlp(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345L)
                        .weightInit(WeightInit.XAVIER)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Adam(0.01))
                        .list()
                        .layer(0, new DenseLayer.Builder().activation(Activation.RELU)
                                        .nIn(28*28).nOut(1000).build())
                        .layer(1, new DenseLayer.Builder().activation(Activation.RELU)
                                        .nIn(1000).nOut(500).build())
                        .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                                        .nIn(500).nOut(250).build())
                        .layer(3, new DenseLayer.Builder().activation(Activation.RELU)
                                        .nIn(250).nOut(500).build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                                        .nIn(500).nOut(1000).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)  
                                        .nIn(1000)  
                                        .nOut(28*28)  
                                        .activation(Activation.RELU)  
                                        .build())
                        .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        return model;
    }
    
    public static MultiLayerNetwork dae(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Adam(0.01))
                        .list()
                        .layer(0, new AutoEncoder.Builder().nIn(28 * 28).nOut(1000)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .lossFunction(LossFunction.KL_DIVERGENCE)
                                .corruptionLevel(0.3)
                                .build())
                        .layer(1, new AutoEncoder.Builder().nIn(1000).nOut(500)
                                     .activation(Activation.RELU)
                                     .weightInit(WeightInit.XAVIER)
                                     .lossFunction(LossFunction.KL_DIVERGENCE)
                                     .corruptionLevel(0.3)
                                     .build())
                        .layer(2, new AutoEncoder.Builder().nIn(500).nOut(250)
                                        .activation(Activation.RELU)
                                        .weightInit(WeightInit.XAVIER)
                                        .lossFunction(LossFunction.KL_DIVERGENCE)
                                        .corruptionLevel(0.3)
                                        .build())
                        .layer(3, new AutoEncoder.Builder().nIn(250).nOut(500)
                                        .activation(Activation.RELU)
                                        .weightInit(WeightInit.XAVIER)
                                        .lossFunction(LossFunction.KL_DIVERGENCE)
                                        .corruptionLevel(0.3)
                                        .build())
                        .layer(4, new AutoEncoder.Builder().nIn(500).nOut(1000)
                                        .activation(Activation.RELU)
                                        .weightInit(WeightInit.XAVIER)
                                        .lossFunction(LossFunction.KL_DIVERGENCE)
                                        .corruptionLevel(0.3)
                                        .build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                     .nIn(1000)
                                     .nOut(28 * 28)
                                     .activation(Activation.RELU)
                                     .build())
                        .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        return model;
    }
    
    public static MultiLayerNetwork vae(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(1234)
                        .updater(new Adam())
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(0, new VariationalAutoencoder.Builder()
                            .activation(Activation.LEAKYRELU)
                            .encoderLayerSizes(1000, 500)        
                            .decoderLayerSizes(500, 1000)        
                            .pzxActivationFunction(Activation.IDENTITY) 
                            .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction())) 
                            .nIn(28 * 28)              
                            .nOut(250)                            
                            .build())
                        .pretrain(true).backprop(false)
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }
}
