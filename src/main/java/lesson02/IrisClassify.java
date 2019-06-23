package lesson02;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class IrisClassify {
    private static final int numClasses = 3;
    
    public static MultiLayerNetwork model(){
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .weightInit(WeightInit.XAVIER)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Adam(0.01))
                        .list()
                        .layer(0, new DenseLayer.Builder().activation(Activation.LEAKYRELU)
                                        .nIn(4).nOut(2).build())
                        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                            .activation(Activation.SOFTMAX)
                                        .nIn(2).nOut(3).build());
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;      
    }  
    
    public static List<DataSet> loadIrisSeq(File file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = null;
        List<DataSet> trainDataSetList = new LinkedList<DataSet>();
        while( (line = br.readLine()) != null ){
             String[] token = line.split(",");
             double[] featureArray = new double[token.length - 1];
             double[] labelArray = new double[numClasses];
             for( int i = 0; i < token.length - 1; ++i ){
                 featureArray[i] = Double.parseDouble(token[i]);
             }
             labelArray[Integer.parseInt(token[token.length - 1])] = 1.0;
             //
             INDArray featureNDArray = Nd4j.create(featureArray);
             INDArray labelNDArray = Nd4j.create(labelArray);
             trainDataSetList.add(new DataSet(featureNDArray, labelNDArray));
        }
        br.close();
        return trainDataSetList;
   }
    
    public static void main(String[] args) throws IOException {
        /*--------------超参数常量声明------------------*/
        final int batchSize = 3;
        final long SEED = 1234L;
        final int trainSize = 120;
        /*--------------数据集构建------------------*/
        List<DataSet> irisList = loadIrisSeq(new File("Iris-data-num.csv"));//该方法参考上一节课程的实现
        DataSet allData = DataSet.merge(irisList);
        allData.shuffle(SEED);
        SplitTestAndTrain split = allData.splitTestAndTrain(trainSize);
        DataSet dsTrain = split.getTrain();
        DataSet dsTest = split.getTest();
        DataSetIterator trainIter = new ListDataSetIterator(dsTrain.asList() , batchSize);
        DataSetIterator testIter = new ListDataSetIterator(dsTest.asList() , batchSize);
        //
        MultiLayerNetwork mlp = model();
        mlp.setListeners(new ScoreIterationListener(1));    //loss score 监听器
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        mlp.setListeners(new StatsListener(statsStorage));
        for( int i = 0; i < 20; ++i ){
             mlp.fit(trainIter);    //训练模型
             trainIter.reset();
             testIter.reset();
        }
        Evaluation eval = mlp.evaluate(testIter);    //在验证集上进行准确性测试
        System.out.println(eval.stats());
        uiServer.stop();                             //停止ui server
    }

}
