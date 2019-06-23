package lesson06.generation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TextGeneration {
    private static final String[] stringArray = new String[]{"我","来自","苏宁","易购","。"};
    
    public static MultiLayerNetwork generateModel(LinkedHashSet<String> words){
        MultiLayerConfiguration netconf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .miniBatch(false)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .list()
                .layer(0, new LSTM.Builder().nIn(words.size()).nOut(128).activation(Activation.TANH).build())
                .layer(1, new LSTM.Builder().nIn(128).nOut(128).activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(128).nOut(words.size()).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(netconf);
        return net;
    }
    
    public static void main(String[] args) {
        LinkedHashSet<String> vocabulary = new LinkedHashSet<>();
        List<String> allWords = new ArrayList<>();
        for (String str : stringArray)
            vocabulary.add(str);
        allWords.addAll(vocabulary);
        //
        MultiLayerNetwork net = generateModel(vocabulary);
        net.setListeners(new ScoreIterationListener(1));
        //
        INDArray input = Nd4j.zeros(1, allWords.size(), stringArray.length);
        INDArray labels = Nd4j.zeros(1, allWords.size(), stringArray.length);

        int samplePos = 0;
        for (String currentWord : stringArray) {
            String nextWord = stringArray[(samplePos + 1) % (stringArray.length)];
            input.putScalar(new int[] { 0, allWords.indexOf(currentWord), samplePos }, 1);
            labels.putScalar(new int[] { 0, allWords.indexOf(nextWord), samplePos }, 1);
            samplePos++;
        }
        DataSet trainingData = new DataSet(input, labels);

        for (int epoch = 0; epoch < 50; epoch++) {

            System.out.println("Epoch " + epoch);

            net.fit(trainingData);
            net.rnnClearPreviousState();

            INDArray testInit = Nd4j.zeros(allWords.size());
            testInit.putScalar(allWords.indexOf(stringArray[0]), 1);
            System.out.print(stringArray[0] + " ");

            INDArray output = net.rnnTimeStep(testInit);

            for (int step = 0; step < 4; ++step ) {
                int sampledCharacterIdx = Nd4j.getExecutioner().exec(new IMax(output), 1).getInt(0);
                System.out.print(allWords.get(sampledCharacterIdx) + " ");
                INDArray nextInput = Nd4j.zeros(allWords.size());
                nextInput.putScalar(sampledCharacterIdx, 1);
                output = net.rnnTimeStep(nextInput);
            }
            System.out.print("\n");
        }
    }

}
