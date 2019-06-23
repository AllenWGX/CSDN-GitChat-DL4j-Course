package lesson06.sequencelabeling;

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
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Segment {
    private static final Logger logger = LoggerFactory.getLogger(Segment.class);
    
    public static String corpusPath = "segment/corpus.txt";
    public static String labelPath = "segment/label.txt";
    public static String testPath = "segment/corpus.txt";

    public static final int VOCAB_SIZE = 50;
    private static final int BatchSize = 2;
    
    private static class SegmentIterator implements DataSetIterator {

        /**
         * 
         */
        private static final long serialVersionUID = -649505930720554358L;
        
        private int batchSize;
        private int vocabSize;
        private SequenceIterator<VocabWord> wordsIter;
        private SequenceIterator<VocabWord> taggingIter;
        private AbstractCache<VocabWord> wordVocab;
        private AbstractCache<VocabWord> taggingVocab;
        
        private boolean toTestSet;

        public SegmentIterator(int batchSize, int vocabSize, SequenceIterator<VocabWord> wordsIter, SequenceIterator<VocabWord> taggingIter,
                                          AbstractCache<VocabWord> wordVocab, AbstractCache<VocabWord> taggingVocab) {
            this.batchSize = batchSize;
            this.vocabSize = vocabSize;
            this.wordsIter = wordsIter;
            this.taggingIter = taggingIter;
            this.wordVocab = wordVocab;
            this.taggingVocab = taggingVocab;
        }

        public AbstractCache<VocabWord> getWordVocabulary() {
            return wordVocab;
        }

        public AbstractCache<VocabWord> getTaggingVocabulary() {
            return taggingVocab;
        }

        /***
         *Generate Test Data 
         */
        public DataSet generateTest(int testSize) {
            toTestSet = true;
            DataSet testData = next(testSize);
            return testData;
        }
        
        @Override
        public DataSet next(int num) { 
            if( toTestSet ){
                reset();
                batchSize = num;
            } 
            /*--Contruct Input Sequence--*/
            List<List<VocabWord>> wordIterList = new ArrayList<>(batchSize);   //Innner List means each train sentence
                                                                            //List of List store a @batchSize train data
            for (int i = 0; i < batchSize && wordsIter.hasMoreSequences(); i++) {
                wordIterList.add(wordsIter.nextSequence().getElements());
            }
            /*--Finish Contructing Input Sequence--*/
            /*-------------------------------------*/
            /*--Contruct Output Sequence--*/
            List<List<VocabWord>> taggingIterList = new ArrayList<>(batchSize);
            for (int i = 0; i < batchSize && taggingIter.hasMoreSequences(); i++) {
                taggingIterList.add(taggingIter.nextSequence().getElements());
            }
            /*--Finish Contructing Output Sequence--*/
            /*--------------------------------------*/
            final int numExamples = Math.min(wordIterList.size(), taggingIterList.size()); //ensure input/output have same number
            int wordIterMaxLength = 0;
            int taggingIterMaxLength = 0;
            /*reserve maximum capacity of input/output sentence*/
            for (int i = 0; i < numExamples; i++) {
                wordIterMaxLength = Math.max(wordIterMaxLength, wordIterList.get(i).size());
            }
            for (int i = 0; i < numExamples; i++) {
                taggingIterMaxLength = Math.max(taggingIterMaxLength, taggingIterList.get(i).size());
            }
            /*finish reserving maximum capacity of input/output sentence*/
            /*--------------------------------------*/
            INDArray featureTensor = Nd4j.create(numExamples, 1, wordIterMaxLength);
            int[] loc = new int[3];
            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = wordIterList.get(i);
                loc[0] = i;                        //arr(0) store the index of batch sentence

                int j = 0;                          //index of the word in the sentence
                for (VocabWord vw : list) {         //traverse the list which store an entire sentence
                    loc[2] = j++;
                    featureTensor.putScalar(loc, vw.getIndex());
                }
            }
            
            //Using a one-hot representation here. Can't use indexes line for input
            INDArray taggingTensor = Nd4j.create(numExamples, 4, taggingIterMaxLength);
            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = taggingIterList.get(i);
                loc[0] = i;

                int j = 0;
                for (VocabWord vw : list) {
                    loc[1] = vw.getIndex();
                    loc[2] = j++;
                    taggingTensor.putScalar(loc, 1.0);   //one hot representation
                }
            }
            return new DataSet(featureTensor, taggingTensor);
        }


        @Override
        public void reset() {
            wordsIter.reset();
            taggingIter.reset();
        }

        @Override
        public boolean hasNext() {
            return wordsIter.hasMoreSequences() && taggingIter.hasMoreSequences();
        }

        @Override
        public DataSet next() {
            return next(batchSize);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public boolean resetSupported() {
            // TODO Auto-generated method stub
            return true;
        }

        @Override
        public boolean asyncSupported() {
            // TODO Auto-generated method stub
            return true;
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
    
    /***
    *单条语料的预测
    */
    private static List<String> predict(String sentence, MultiLayerNetwork net,
                        AbstractCache<VocabWord> wordVocabulary,
                        AbstractCache<VocabWord> taggingVocabulary){
            char[] charArray = sentence.toCharArray();
            double[] wordIndices = new double[charArray.length];
            for(int idx = 0; idx < charArray.length; ++idx){
                int wordIndex = wordVocabulary.indexOf(String.valueOf(charArray[idx]));
                wordIndices[idx] = wordIndex;
            }
            INDArray featureTensor = Nd4j.create(1, 1, charArray.length);
            featureTensor.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.interval(0, charArray.length)}, Nd4j.create(wordIndices));
            INDArray out = net.output(featureTensor,false);
            System.out.println("Test");
            System.out.println(out);
            INDArray argMax = Nd4j.argMax(out, 1);
            StringBuilder labels = new StringBuilder();
            for( int col = 0; col < argMax.columns(); ++col ){
                double labelIndex = argMax.getDouble(col);
                String labelWord = taggingVocabulary.wordAtIndex((int)labelIndex);
                labels.append(labelWord);
            }
            String labelStr = labels.toString();
            List<String> result = new ArrayList<>();
            for( int index = 0; index < sentence.length();  ){
                if( labelStr.charAt(index) == 's'){
                    result.add(Character.toString(sentence.charAt(index)));
                    ++index;
                }else if( labelStr.charAt(index) == 'b' ){
                    StringBuilder tempWord = new StringBuilder();
                    do{
                        tempWord.append(Character.toString(sentence.charAt(index)));
                        ++index;
                    }while( (  index < sentence.length() && labelStr.charAt(index) != 'e') );
                    tempWord.append(Character.toString(sentence.charAt(index)));
                    ++index;
                    result.add(tempWord.toString());
                }
            }
            return result;
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
    
    private static SegmentIterator getTrainData(int batchSize, int vocabSize) throws Exception {

        Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> en = getIterator(true, vocabSize);
        Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> fr = getIterator(false, vocabSize);

        return new SegmentIterator(batchSize, vocabSize, en.getFirst(), fr.getFirst(), en.getSecond(), fr.getSecond());
    }
    
    public static MultiLayerNetwork getSegModel(final int VOCAB_SIZE){
        MultiLayerConfiguration netconf = new NeuralNetConfiguration.Builder()
              .seed(1234L)
              .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) 
              .updater(new Adam(0.01))
              .list()
              .layer(0, new EmbeddingLayer.Builder().nIn(VOCAB_SIZE).nOut(128).activation(Activation.IDENTITY).build())
              .layer(1, new LSTM.Builder().nIn(128).nOut(128).activation(Activation.SOFTSIGN).build())
              .layer(2, new RnnOutputLayer.Builder().nIn(128).nOut(4).activation(Activation.SOFTMAX).build())
              .setInputType(InputType.recurrent(VOCAB_SIZE))
              .build();

      MultiLayerNetwork net = new MultiLayerNetwork(netconf);
      net.init();
      return net;
    }
    
    
    
    public static void main(String[] args) throws Exception {
        SegmentIterator trainData = getTrainData(BatchSize, VOCAB_SIZE);
        AbstractCache<VocabWord> taggingVocabulary = trainData.getTaggingVocabulary();
        AbstractCache<VocabWord> wordVocabulary = trainData.getWordVocabulary();
        MultiLayerNetwork net = getSegModel(VOCAB_SIZE);
        net.setListeners(new ScoreIterationListener(1));
        //
        for( int numEpoch = 0; numEpoch < 20; ++numEpoch){
            net.fit(trainData);
            trainData.reset();
            INDArray out = net.output(trainData);
            System.out.println(out);
            trainData.reset();
            Evaluation evaluation = (Evaluation)net.evaluate(trainData);
            System.out.println(evaluation.accuracy());
        }
        //
        System.out.println(predict("中国的清华大学是著名学府", net, wordVocabulary,taggingVocabulary));
        return;

    }

}
