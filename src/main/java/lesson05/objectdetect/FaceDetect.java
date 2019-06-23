package lesson05.objectdetect;

import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.deeplearning4j.zoo.model.helper.DarknetHelper.addLayers;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FaceDetect {
    private static Logger logger = LoggerFactory.getLogger(FaceDetect.class);
    
    public static ComputationGraph tinyYOLO(int nBoxes, int numLabels, INDArray priors, double lambdaNoObj, double lambdaCoord){
        GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                        .seed(123456L)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                        .gradientNormalizationThreshold(1.0)
                        .updater(new Adam(1e-3))
                        .activation(Activation.IDENTITY)
                        //.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                        .graphBuilder()
                        .addInputs("input")
                        .setInputTypes(InputType.convolutional(416, 416, 3));
                addLayers(graphBuilder, 1, 3, 3, 16, 2, 2);
                addLayers(graphBuilder, 2, 3, 16, 32, 2, 2);
                addLayers(graphBuilder, 3, 3, 32, 64, 2, 2);
                addLayers(graphBuilder, 4, 3, 64, 128, 2, 2);
                addLayers(graphBuilder, 5, 3, 128, 256, 2, 2);
                addLayers(graphBuilder, 6, 3, 256, 512, 2, 1);
                addLayers(graphBuilder, 7, 3, 512, 1024, 0, 0);
                addLayers(graphBuilder, 8, 3, 1024, 1024, 0, 0);

                int layerNumber = 9;
                graphBuilder
                        .addLayer("convolution2d_" + layerNumber,
                                new ConvolutionLayer.Builder(1,1)
                                        .nIn(1024)
                                        .nOut(nBoxes * (5 + numLabels))
                                        .weightInit(WeightInit.XAVIER)
                                        .stride(1,1)
                                        .convolutionMode(ConvolutionMode.Same)
                                        .weightInit(WeightInit.RELU)
                                        .activation(Activation.IDENTITY)
                                        .build(),
                                "activation_" + (layerNumber - 1))
                        .addLayer("outputs",
                                new Yolo2OutputLayer.Builder()
                                        .lambbaNoObj(lambdaNoObj)
                                        .lambdaCoord(lambdaCoord)
                                        .boundingBoxPriors(priors)
                                        .build(),
                                "convolution2d_" + layerNumber)
                        .setOutputs("outputs");
        ComputationGraphConfiguration conf = graphBuilder.build();
        ComputationGraph net = new ComputationGraph(conf);
        return net;
}
    
    
    public static void main(String[] args) throws IOException {
        if( args.length != 4 ){
            logger.error("Error Parameter");
            return;
        }
        final int width = 416;
        final int height = 416;
        final int nChannels = 3;
        final int gridWidth = 13;
        final int gridHeight = 13;
        //
        final String inputPath = args[0];
        final String inputTestPath = args[1];
        final int batchSize = Integer.parseInt(args[2]);
        final int nEpochs = Integer.parseInt(args[3]);
        final Random rng = new Random(12345L);
        //
        int nClasses = 1;
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 1.0;
        double[][] priorBoxes = {{1.08, 1.19}, {3.42, 4.41}, {6.63, 11.38}, {9.42, 5.11}, {16.62, 10.52}};    
        
        //读取训练和测试数据集
        FileSplit trainData = new FileSplit(new File(inputPath), NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit testData = new FileSplit(new File(inputTestPath), NativeImageLoader.ALLOWED_FORMATS, rng);

        //读取 VOC 格式的标注数据
        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new VocLabelProvider(inputPath));
        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new VocLabelProvider(inputTestPath));
        recordReaderTest.initialize(testData);

        //归一化处理
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        //
        ComputationGraph model = tinyYOLO(nBoxes, nClasses, Nd4j.create(priorBoxes), lambdaNoObj, lambdaCoord);

        logger.info("Train model...");
        model.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < nEpochs; i++) {
              train.reset();
              while (train.hasNext()) {
                   model.fit(train.next());
               }
        }
        ModelSerializer.writeModel(model, "face_detect.mod", true);

    }

}
