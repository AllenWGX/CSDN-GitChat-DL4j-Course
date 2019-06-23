package lesson06.classification;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.util.Pair;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.FilteredSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class TextClassify {
    private static final Logger log = LoggerFactory.getLogger(TextClassify.class);
    public static String corpusPath = "comment/corpus.txt";
    public static String labelPath = "comment/label.txt";
    public static String testPath = "comment/corpus.txt";

    public static final int VOCAB_SIZE = 50000;
    public static final int FEATURE_VEC_SIZE = 64;
    
    private static class TextClassifyIterator implements DataSetIterator {

        private static final long serialVersionUID = -649505930720554358L;

        private int batchSize;
        private int vocabSize;
        private int maxLength;
        private SequenceIterator<VocabWord> iter1;
        private SequenceIterator<VocabWord> iter2;
        private AbstractCache<VocabWord> vocabCache1;
        private AbstractCache<VocabWord> vocabCache2;

        private boolean toTestSet;

        public TextClassifyIterator(int batchSize, int vocabSize, SequenceIterator<VocabWord> iter1, SequenceIterator<VocabWord> iter2,
                                          AbstractCache<VocabWord> vocabCache1, AbstractCache<VocabWord> vocabCache2) {
            this.batchSize = batchSize;
            this.vocabSize = vocabSize;
            this.iter1 = iter1;
            this.iter2 = iter2;
            this.vocabCache1 = vocabCache1;
            this.vocabCache2 = vocabCache2;

        }

        public AbstractCache<VocabWord> getVocabCache1() {
            return vocabCache1;
        }

        public AbstractCache<VocabWord> getVocabCache2() {
            return vocabCache2;
        }

        public int getMaxLen(){
            return maxLength;
        }

        @Override
        public DataSet next(int num) {

            if( toTestSet ){
                reset();
                batchSize = num;
            }

            /*--构建输入序列--*/
            List<List<VocabWord>> iter1List = new ArrayList<>(batchSize);       

            for (int i = 0; i < batchSize && iter1.hasMoreSequences(); i++) {
                iter1List.add(iter1.nextSequence().getElements());
            }
            /*--构建输入序列结束--*/
            /*-------------------------------------*/
            /*--构建输出序列--*/
            List<List<VocabWord>> iter2List = new ArrayList<>(batchSize);
            for (int i = 0; i < batchSize && iter2.hasMoreSequences(); i++) {
                iter2List.add(iter2.nextSequence().getElements());
            }
            /*--输出序列构建完毕--*/
            /*--------------------------------------*/
            int numExamples = Math.min(iter1List.size(), iter2List.size());     //保证语料及标注在数量上的一致性
            int in1Length = 0;
            int in2Length = 0;
            /*以下部分针对输入/输出序列长度，以当前 batch 里最长序列的长度作为全部序列的长度，以此兼容变长的场景*/
            for (int i = 0; i < numExamples; i++) {
                in1Length = Math.max(in1Length, iter1List.get(i).size());
            }
            for (int i = 0; i < numExamples; i++) {
                in2Length = Math.max(in2Length, iter2List.get(i).size());
            }
            maxLength = Math.max(in1Length, in2Length);
            /*完成输入/输出变长的支持*/
            /*--------------------------------------*/


            INDArray features = Nd4j.create(numExamples, 1, maxLength);  //当  batchSize=32，maxLength=39，则每条语料由每个词的哈夫曼编码构成，如：283.0，128.0，10.0，0.0，0.0……

            INDArray labels = Nd4j.create(numExamples, 2, maxLength);   //2 这里代表分类的类别数目
            //
            INDArray featuresMask = Nd4j.zeros(numExamples, maxLength);
            INDArray labelsMask = Nd4j.zeros(numExamples, maxLength);

            int[] origin = new int[3];
            int[] mask = new int[2];
            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = iter1List.get(i);
                origin[0] = i;                        //三维数组 origin 中，下标为 0 存储语料的批次（batch index）
                mask[0] = i;

                int j = 0;                          //每条语料中词的位置索引
                for (VocabWord vw : list) {         //遍历整条语料
                    origin[2] = j;
                    features.putScalar(origin, vw.getIndex());
                    //
                    mask[1] = j;
                    featuresMask.putScalar(mask, 1.0);  //对于掩码序列，如果当前位置有词出现，则为 1.0，否则为 0.0
                    ++j;
                }
                //
                int idx = iter2List.get(i).get(0).getIndex();
                int lastIdx = list.size();
                labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //设置标注信息，[1.0] 为正例，[0.1] 为负例
                labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //标注序列掩码保证只在最后一个 RNN Cell 输出标注信息，其余掩码都为 0.0
            }

            return new DataSet(features, labels, featuresMask, labelsMask);
        }


        @Override
        public void reset() {
            iter1.reset();
            iter2.reset();
        }

        @Override
        public boolean hasNext() {
            return iter1.hasMoreSequences() && iter2.hasMoreSequences();
        }

        @Override
        public DataSet next() {
            return next(batchSize);
        }

        @Override
        public int inputColumns() {
            // TODO Auto-generated method stub
            return 0;
        }

        @Override
        public int totalOutcomes() {
            // TODO Auto-generated method stub
            return 0;
        }

        @Override
        public boolean resetSupported() {
            // TODO Auto-generated method stub
            return false;
        }

        @Override
        public boolean asyncSupported() {
            // TODO Auto-generated method stub
            return false;
        }

        @Override
        public int batch() {
            // TODO Auto-generated method stub
            return 0;
        }

        @Override
        public void setPreProcessor(DataSetPreProcessor preProcessor) {
            // TODO Auto-generated method stub
            
        }

        @Override
        public DataSetPreProcessor getPreProcessor() {
            // TODO Auto-generated method stub
            return null;
        }

        @Override
        public List<String> getLabels() {
            // TODO Auto-generated method stub
            return null;
        }
    }
    
    private static Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> getIterator(boolean english, int maxVocabSize) throws Exception {
        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();
        File file;
        if (english) file = new File(corpusPath);
        else file = new File(labelPath);
        
        /*define line reader,tokenizer for initalizing SentenceTransformer*/
        /*-----------------------------------*/
        BasicLineIterator lineIter = new BasicLineIterator(file);       //read file line one by one
        TokenizerFactory t = new DefaultTokenizerFactory();             //default tokenizer(for english)
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                                                                .iterator(lineIter)
                                                                .tokenizerFactory(t)
                                                                .build();

        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<>(transformer)
                                                                   .build();
        /*finish defining SentenceTransformer*/
        /*-----------------------------------*/
        /*build vacabulary*/
        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                                                        .addSource(sequenceIterator, 1)
                                                        .setTargetVocabCache(vocabCache)
                                                        .setEntriesLimit(maxVocabSize)
                                                        .build();
        constructor.buildJointVocabulary(false, true);
        sequenceIterator.reset();
        /*finish building vacabulary*/
        SequenceIterator<VocabWord> filteredIterator = new FilteredSequenceIterator<>(sequenceIterator, vocabCache);
        return new Pair<>(filteredIterator, vocabCache);
    }
    
    private static TextClassifyIterator getTrainData(int batchSize, int vocabSize) throws Exception {

        Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> en = getIterator(true, vocabSize);
        Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> fr = getIterator(false, vocabSize);

        return new TextClassifyIterator(batchSize, vocabSize, en.getFirst(), fr.getFirst(), en.getSecond(), fr.getSecond());
    }
    
    private static MultiLayerNetwork textClassifyModel(final int VOCAB_SIZE){
        MultiLayerConfiguration netconf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(5 * 1e-4) 
                .updater(new Adam(0.01))
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(VOCAB_SIZE).nOut(100).activation(Activation.IDENTITY).build())
                .layer(1, new LSTM.Builder().nIn(100).nOut(100).activation(Activation.SOFTSIGN).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(100).nOut(2).build())
                .setInputType(InputType.recurrent(VOCAB_SIZE))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(netconf);
        net.init();
        return net;
    }
    
    public static void main(String[] args) throws Exception {
        TextClassifyIterator trainData = getTrainData(4, VOCAB_SIZE);
        MultiLayerNetwork network = textClassifyModel(VOCAB_SIZE);
        network.setListeners(new ScoreIterationListener(1));
        //
        for( int numEpoch = 0; numEpoch < 20; ++numEpoch){
            network.fit(trainData);
            trainData.reset();
            Evaluation evaluation = (Evaluation)network.evaluate(trainData);
            System.out.println(evaluation.accuracy());
        }
        //
    }

}
