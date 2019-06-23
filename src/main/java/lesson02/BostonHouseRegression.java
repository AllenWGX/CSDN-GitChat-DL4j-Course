package lesson02;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class BostonHouseRegression {
    public static List<DataSet> loadHousePrice(File file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = null;
        List<DataSet> totalDataSetList = new LinkedList<DataSet>();
        while( (line = br.readLine()) != null ){
            String[] token = line.split(",");
            double[] featureArray = new double[token.length - 1];
            double[] labelArray = new double[1];
            for( int i = 0; i < token.length - 1; ++i ){
                featureArray[i] = Double.parseDouble(token[i]);
            }
            labelArray[0] = Double.parseDouble(token[token.length - 1]);
            //
            INDArray featureNDArray = Nd4j.create(featureArray);
            INDArray labelNDArray = Nd4j.create(labelArray);
            totalDataSetList.add(new DataSet(featureNDArray, labelNDArray));
        }
        br.close();
        return totalDataSetList;
}
    
    public static MultiLayerNetwork model(){
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .seed(12345L)
                        .updater(new Adam(0.01))
                        .weightInit(WeightInit.XAVIER)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .list()
                        .layer(0, new DenseLayer.Builder().activation(Activation.LEAKYRELU)
                                        .nIn(13).nOut(10).build())
                        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                        .activation(Activation.IDENTITY)
                                        .nIn(10).nOut(1).build());
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;      
}
    
    
    public static void main(String[] args) throws IOException {
        final int batchSize = 4;
        final long SEED = 1234L;
        final int trainSize = 400;
        List<DataSet> housePriceList = loadHousePrice(new File("house_price.csv"));
        //获取全部数据并且打乱顺序
        DataSet allData = DataSet.merge(housePriceList);
        allData.shuffle(SEED);
        //划分训练集和验证集
        SplitTestAndTrain split = allData.splitTestAndTrain(trainSize);
        DataSet dsTrain = split.getTrain();
        DataSet dsTest = split.getTest();
        DataSetIterator trainIter = new ListDataSetIterator(dsTrain.asList() , batchSize);
        DataSetIterator testIter = new ListDataSetIterator(dsTest.asList() , batchSize);
        //归一化处理
        DataNormalization scaler = new NormalizerMinMaxScaler(0,1);
        scaler.fit(trainIter);
        scaler.fit(testIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);
        //声明多层感知机
        MultiLayerNetwork mlp = model();
        mlp.setListeners(new ScoreIterationListener(1));
        //训练 200 个 Epoch
        for( int i = 0; i < 200; ++i ){
            mlp.fit(trainIter);
            trainIter.reset();
        }
        //利用 Deeplearning4j 内置的回归模型分析器进行模型评估
        RegressionEvaluation eval = mlp.evaluateRegression(testIter);
        System.out.println(eval.stats());
        testIter.reset();
        //输出验证集的真实值和预测值
        while( testIter.hasNext() ){
            System.out.println(testIter.next().getLabels());
        }
        System.out.println();
        testIter.reset();
        System.out.println(mlp.output(testIter));
    }

}
