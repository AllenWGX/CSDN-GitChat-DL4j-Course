package lesson05.classification;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FashionMnistClassify {
    private static Logger logger = LoggerFactory.getLogger(FashionMnistClassify.class);
    public static MultiLayerNetwork getModel(){
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .weightInit(WeightInit.XAVIER)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Adam(0.01))
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                .nIn(1)
                                .stride(1, 1)
                                .nOut(32)
                                .activation(Activation.LEAKYRELU)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                .stride(1, 1)
                                .nOut(64)
                                .activation(Activation.LEAKYRELU)
                                .build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.LEAKYRELU)
                                .nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(10)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutionalFlat(28, 28, 1));
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        return model; 
}
    
    
    public static void main(String[] args) throws IOException {
        if( args.length != 3 ){
            logger.error("Error Parameter");
            return;
        }
        final int BATCH_SIZE = Integer.parseInt(args[0]);
        final int NUM_EPOCHS = Integer.parseInt(args[1]);
        final String MODEL_SAVE_PATH = args[2];
        //
        DataSetIterator mnistTrain = new MnistDataSetIterator(BATCH_SIZE, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(BATCH_SIZE, false, 12345);
        MultiLayerNetwork model = getModel();
        //
        for( int i = 0; i < NUM_EPOCHS; ++i ){
            model.fit(mnistTrain);
        }
        Evaluation eval = (Evaluation)model.evaluate(mnistTest);
        logger.info(eval.stats());
        ModelSerializer.writeModel(model, MODEL_SAVE_PATH, true);

    }

}
