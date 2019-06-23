package lesson04;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class MnistCompress {
    
    private static NativeImageLoader imageLoader = new NativeImageLoader(28,28,1);
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    
    public static void saveReconstructImg(INDArray recontruct,String path) throws IOException{
        int[] arrays = recontruct.data().asInt();
        int[][] matrix = new int[28][28];
        for( int row = 0; row < 28; ++row ){
            for( int col = 0; col < 28; ++col ){
                 matrix[col][row]= arrays[row * 28 + col];
            }
        }
        BufferedImage bufferImage2=new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        for(int y = 0; y < 28; ++y){
            for(int x = 0; x < 28; ++x){
                int Pixel=matrix[x][y]<<16 | matrix[x][y] << 8 | matrix[x][y];
                bufferImage2.setRGB(x, y,Pixel);
            }
        }
        File outputfile = new File(path);
        ImageIO.write(bufferImage2, "jpg", outputfile);
    }
    
    public static DataSetIterator genDataSetIter(String path, int batchSize) throws IOException{
        List<DataSet> datasetLst = new LinkedList<>();
        // 
        File dir = new File(path);  //"D:/data/MNIST"
        File[] subdirs = dir.listFiles();
        for( File subdir : subdirs ){
            File[] files = subdir.listFiles();
            for( File file : files ){
                INDArray feature = imageLoader.asRowVector(file);
                scaler.transform(feature);
                datasetLst.add(new org.nd4j.linalg.dataset.DataSet(feature, feature));
            }
        }
        Collections.shuffle(datasetLst);
        DataSetIterator trainIter = new ListDataSetIterator(datasetLst, batchSize);
        return trainIter;
    }
    
    public static void reconstructImg(MultiLayerNetwork network, String testFilePath) throws IOException{
        File testDir = new File(testFilePath); //"D:/data/MNIST/test1"
        File[] testFiles = testDir.listFiles();
        for( File testFile :  testFiles){
            INDArray recontruct = imageLoader.asRowVector(testFile);
            scaler.transform(recontruct);
            INDArray _reconstruct = network.output(recontruct);
            _reconstruct = _reconstruct.mul(255);
            saveReconstructImg(_reconstruct, "test.jpg");
        }
    }
    
    private static void reconstructDaeImg(MultiLayerNetwork network, String testFilePath) throws IOException{
        File testDir = new File(testFilePath); //"D:/data/MNIST/test1"
        File[] testFiles = testDir.listFiles();
        LayerWorkspaceMgr workspaceMgr = LayerWorkspaceMgr.builder().defaultNoWorkspace().build();
        for( File testFile :  testFiles){
            INDArray recontruct = imageLoader.asRowVector(testFile);
            scaler.transform(recontruct);
            org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
            = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) network.getLayer(0);
            INDArray latent = vae.activate(recontruct,false,workspaceMgr);
            INDArray _recontruct = vae.generateAtMeanGivenZ(latent);
            _recontruct = _recontruct.mul(255);
            saveReconstructImg(_recontruct, "test.jpg");
        }
    }
    
    public static void main(String[] args) throws IOException {
        if(args.length != 1){
            System.err.println("Error Param");
            return;
        }
        final String modelType = args[0];
        final int batchSize = 32;
        final int numEpoch = 10;
        //
        DataSetIterator trainDataIter = genDataSetIter("data/MNIST", batchSize);
        //
        MultiLayerNetwork network = null;
        switch( modelType ){
            case "vae":network = AeModelZoo.vae();break;
            case "mlp":network = AeModelZoo.mlp();break;
            case "dae":network = AeModelZoo.dae();break;
        }
        //
        network.setListeners(new ScoreIterationListener(1));
        for( int i = 0; i < numEpoch; ++i ){
            if( "vae".equals(modelType) )network.pretrain(trainDataIter);
            else network.fit(trainDataIter);
            trainDataIter.reset();
        }
        //
        if( "vae".equals(modelType) )reconstructDaeImg(network, "data/MNIST/test1");
        else reconstructImg(network, "data/MNIST/test1");
    }

}
